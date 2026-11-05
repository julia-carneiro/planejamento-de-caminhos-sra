import sys
import math
import time
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from identificar_formas import (abrir_camera, detectar_robo, detectar_obstaculos,
                                expandir_obstaculos_minkowski, detectar_mesa,
                                obstaculos_fora_da_mesa, escala_pela_mesa)
from calibracao import calibrar_camera, pixel_para_real
from planejamento import (build_grid, plan_on_grid, subdivide, plan_on_quadtree,
                          simplificar_caminho, converter_caminho_para_waypoints)
import Remote_Position_Control_TEC220 as rpc

RESOLUCAO = 15
MAX_DEPTH = 7
PORTA_SERIAL = "/dev/ttyUSB0"
BAUD_RATE = 57600
TESTE_SEM_ROBO = False
MAX_WAYPOINTS = 10
RAIO_ROBO_PX = 50
RAIO_EXCLUSAO_ROBO_PX = 90 # contornos dentro desse raio do robô são ignorados como obstáculo
THETA_INICIAL = 0.0
# Para onde a FRENTE do robô aponta na imagem ao iniciar, (u, v): (-1,0)=esquerda, (0,-1)=cima, (1,0)=direita
FRENTE_ROBO_TELA = (-1.0, 0.0)
FATOR_ODOMETRIA = 0.60 / 0.80

destino_pixel = None
inicio_pixel = None


def capturar_clique(event, x, y, flags, param):
    global destino_pixel, inicio_pixel
    if event == cv2.EVENT_LBUTTONDOWN:
        destino_pixel = (x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        inicio_pixel = (x, y)

def monitorar_ate_parar(ser, poll_s=0.5, timeout_s=120.0, leituras_paradas=3, n_waypoints=None):
    t0 = time.time()
    paradas, ultima = 0, None
    while time.time() - t0 < timeout_s:
        t = rpc.get_telemetry(ser)
        if t is not None:
            ultima = t
            print(f"  [{time.time() - t0:5.1f}s] {rpc.format_telemetry(t)}")
            if t["state"] not in (0, 1) and abs(t["v"]) < 0.01:
                paradas += 1
                if paradas >= leituras_paradas:
                    if n_waypoints is not None and t["wp"] < n_waypoints - 1:
                        print(f"  Aviso: firmware em DONE no wp={t['wp']} de {n_waypoints - 1}; "
                              f"missão encerrou antes do último waypoint.")
                    return True, ultima
            else:
                paradas = 0
        plt.pause(poll_s)  # mantém a janela do plot viva durante o monitoramento
    return False, ultima


def enviar_para_robo(waypoints, porta, baud, theta_inicial=0.0):
    """
    Protocolo do firmware:
      <C>            zera a lista
      <O,x,y,th>     define a pose inicial (origem = posição do robô)
      <A,x,y,th>     um waypoint por comando
      <R>            liga + auto-equilíbrio + navega
    x,y em metros (3 casas); th em graus (1 casa).
    """
    def normalizar(wp):
        x, y, th = float(wp[0]), float(wp[1]), float(wp[2])
        th = (th + 180.0) % 360.0 - 180.0
        return round(x, 3), round(y, 3), round(th, 1)

    pts = [normalizar(wp) for wp in waypoints]
    if len(pts) < 2:
        print("Caminho curto demais para enviar.")
        return

    inicio, alvos = pts[0], pts[1:]   # o 1º ponto é a posição atual do robô

    if TESTE_SEM_ROBO:
        print("\n[MODO TESTE] Mensagens que seriam enviadas:")
        print("  <C>")
        print(f"  <O,{inicio[0]:.3f},{inicio[1]:.3f},{theta_inicial:.1f}>")
        for x, y, th in alvos:
            print(f"  <A,{x:.3f},{y:.3f},{th:.1f}>")
        print("  <R>")
        return

    try:
        ser = rpc.connect(port=porta, baud=baud)
        try:
            print(f"<C> -> {rpc.stop(ser)}")

            resp = rpc.set_pose(ser, inicio[0], inicio[1], theta_inicial)
            print(f"<O> -> {resp}")
            if resp is None or not resp.startswith("<OK"):
                print("Falha ao redefinir a pose.")
                return

            for i, (x, y, th) in enumerate(alvos):
                resp = rpc.add_waypoint(ser, x, y, th)
                print(f"<A> WP {i}: x={x:.3f} y={y:.3f} th={th:.1f} -> {resp}")
                if resp is None or not resp.startswith("<OK"):
                    print(f"Falha ao enviar waypoint {i}.")
                    return

            lista = rpc.list_waypoints(ser)
            print(f"<L> -> robô tem {0 if lista is None else len(lista)} waypoints (enviados: {len(alvos)})")
            if lista is None or len(lista) != len(alvos):
                print("Lista do robô difere da enviada; abortando.")
                return

            ok, resp, msg = rpc.start(ser, wait_auto_eq=True)
            print(f"<R> -> {resp} ({msg})")
            if not ok:
                return

            concluido, ultima = monitorar_ate_parar(ser, n_waypoints=len(alvos))
            print("Status da missao:", "missão concluída" if concluido else "timeout de monitoramento")
            if ultima is not None:
                print(f"Posicao final (odometria): x={ultima['x']:+.3f} m  y={ultima['y']:+.3f} m  "
                      f"th={ultima['th_deg']:+.1f} graus")
        finally:
            ser.close()
    except Exception as e:
        print(f"Falha na comunicacao serial: {e}")

def _desenhar(frame, inicio, destino):
    tela = frame.copy()
    if inicio:
        cv2.circle(tela, inicio, 8, (0, 255, 0), -1)
    if destino:
        cv2.circle(tela, destino, 8, (0, 0, 255), -1)
    cv2.imshow("Camera / Imagem", tela)
    return cv2.waitKey(1) & 0xFF


def interface_captura():
    global destino_pixel, inicio_pixel
    cv2.namedWindow("Camera / Imagem")
    cv2.setMouseCallback("Camera / Imagem", capturar_clique)

    args_img = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args_img:
        f = cv2.imread(args_img[0])
        if f is None:
            return None, None
        inicio_pixel = detectar_robo(f) or inicio_pixel
        while True:
            tecla = _desenhar(f, inicio_pixel, destino_pixel)
            if inicio_pixel and destino_pixel:
                cv2.destroyAllWindows()
                return f, inicio_pixel
            if tecla == ord('q'):
                cv2.destroyAllWindows()
                return None, None

    cap = abrir_camera()
    if not cap.isOpened():
        return None, None
    frame_capturado = None
    while True:
        ret, f = cap.read()
        if not ret or f is None:
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            continue
        inicio_pixel = detectar_robo(f) or inicio_pixel
        tecla = _desenhar(f, inicio_pixel, destino_pixel)
        if inicio_pixel and destino_pixel:
            frame_capturado = f
            break
        if tecla == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame_capturado, inicio_pixel

def main():
    frame, start = interface_captura()
    if frame is None or start is None or destino_pixel is None:
        return
    goal = destino_pixel

    contorno_mesa = detectar_mesa(frame)
    obstaculos_brutos = detectar_obstaculos(frame, inicio=start, destino=goal,
                                           raio_exclusao_robo=RAIO_EXCLUSAO_ROBO_PX)
    escala_mesa = None

    if contorno_mesa is not None:
        obstaculos_brutos = [p for p in obstaculos_brutos
                             if cv2.pointPolygonTest(contorno_mesa, tuple(map(float, p.mean(axis=0))), False) >= 0]

    obstaculos = expandir_obstaculos_minkowski(obstaculos_brutos, frame.shape, RAIO_ROBO_PX)

    if contorno_mesa is not None:
        obstaculos += obstaculos_fora_da_mesa(contorno_mesa, frame.shape, RAIO_ROBO_PX)
        ppm_comp, ppm_larg = escala_pela_mesa(contorno_mesa)
        escala_mesa = ppm_larg
        print(f"Escala pela mesa: {ppm_comp / 100:.2f} px/cm (comprimento), "
              f"{ppm_larg / 100:.2f} px/cm (largura)")
        for nome, pt in [("inicio", start), ("destino", goal)]:
            if cv2.pointPolygonTest(contorno_mesa, (float(pt[0]), float(pt[1])), True) < RAIO_ROBO_PX:
                print(f"Aviso: {nome} fora da mesa ou perto demais da borda.")
    else:
        print("Aviso: mesa nao detectada; planejando sem limite de bordas.")

    h, w = frame.shape[:2]
    bounds = (0, 0, w, h)

    grid = build_grid(bounds, obstaculos, resolution=RESOLUCAO)
    caminho_grade = plan_on_grid(grid, start, goal, resolution=RESOLUCAO, bounds=bounds)

    tree = subdivide(bounds, obstaculos, MAX_DEPTH)
    caminho_qt, folhas_livres = plan_on_quadtree(tree, obstaculos, start, goal)

    waypoints = None
    caminho_wp = None
    if caminho_qt:
        px_por_m = calibrar_camera(frame, escala_mesa=escala_mesa, recalibrar="--recalibrar" in sys.argv)
        if px_por_m is not None:
            print(f"Escala usada: {px_por_m / 100:.2f} px/cm")
            mascara_obs = np.zeros(frame.shape[:2], np.uint8)
            cv2.fillPoly(mascara_obs, [np.round(p).astype(np.int32) for p in obstaculos], 255)
            caminho = simplificar_caminho(caminho_qt, mascara_obs)
            caminho_wp = caminho
            print(f"Waypoints: {len(caminho_qt)} nós da quadtree -> {len(caminho)} pontos (sem cruzar obstáculos)")
            if len(caminho) - 1 > MAX_WAYPOINTS:
                print(f"Aviso: {len(caminho) - 1} waypoints > MAX_WAYPOINTS={MAX_WAYPOINTS}. "
                      f"Enviando só os {MAX_WAYPOINTS} primeiros (trecho livre); o robô para antes do destino. "
                      f"Rode de novo da nova posição.")
                caminho = caminho[:MAX_WAYPOINTS + 1]
                caminho_wp = caminho
        if px_por_m is not None and caminho:
            caminho_metros = pixel_para_real(caminho, px_por_m, origem=start, frente=FRENTE_ROBO_TELA)
            caminho_metros = [(x * FATOR_ODOMETRIA, y * FATOR_ODOMETRIA) for x, y in caminho_metros]
            waypoints = converter_caminho_para_waypoints(caminho_metros)
        elif px_por_m is None:
            print("Aviso: sem calibracao salva, sem tabuleiro e sem mesa; waypoints nao enviados.")
    else:
        print("Nenhum caminho encontrado pela quadtree.")

    # plot aparece primeiro (sem bloquear), fica aberto durante a missão
    _plotar(frame, obstaculos, obstaculos_brutos, caminho_grade, caminho_qt, folhas_livres, start, goal, caminho_wp)

    if waypoints:
        enviar_para_robo(waypoints, PORTA_SERIAL, BAUD_RATE, theta_inicial=THETA_INICIAL)
    plt.show()  # ao fim da missão, mantém o plot até você fechar


def _plotar(frame, obstaculos, obstaculos_brutos, caminho_grade, caminho_qt, folhas_livres, start, goal, caminho_wp=None):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for ax, titulo, caminho in [(ax1, f"Grade regular ({RESOLUCAO}px)", caminho_grade),
                                (ax2, f"Quadtree ({len(folhas_livres)} folhas livres)", caminho_qt)]:
        ax.imshow(rgb)
        for poly in obstaculos:
            ax.fill(poly[:, 0], poly[:, 1], color="red", alpha=0.3)
        for poly in obstaculos_brutos:
            ax.fill(poly[:, 0], poly[:, 1], color="darkred", alpha=0.7)
        if caminho:
            ax.plot(*zip(*caminho), "g-", linewidth=2)
        if caminho_wp:
            ax.plot(*zip(*caminho_wp), "o-", color="orange", linewidth=2, markersize=6, label="waypoints enviados")
        ax.plot(*start, "bo", markersize=8)
        ax.plot(*goal, "yo", markersize=8)
        ax.set_title(titulo)
    for f in folhas_livres:
        ax2.add_patch(plt.Rectangle((f[0], f[1]), f[2] - f[0], f[3] - f[1],
                                    fill=False, edgecolor="gray", linewidth=0.3))
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)  # desenha a janela antes de enviar ao robô


if __name__ == "__main__":
    main()
