import sys
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from identificar_formas import abrir_camera, detectar_robo, obter_deteccoes, montar_obstaculos, expandir_obstaculos_minkowski
from calibracao import calibrar_camera, pixel_para_real
from planejamento import build_grid, plan_on_grid, subdivide, plan_on_quadtree, limitar_waypoints, converter_caminho_para_waypoints
import Remote_Position_Control_TEC220 as rpc

RESOLUCAO = 15
MAX_DEPTH = 7
PORTA_SERIAL = "COM3"
BAUD_RATE = 57600
TESTE_SEM_ROBO = False
MAX_WAYPOINTS = 10
RAIO_ROBO_PX = 25

destino_pixel = None
inicio_pixel = None

def capturar_clique(event, x, y, flags, param):
    global destino_pixel, inicio_pixel
    if event == cv2.EVENT_LBUTTONDOWN:
        destino_pixel = (x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        inicio_pixel = (x, y)

def enviar_para_robo(waypoints, porta, baud, theta_inicial=0.0):
    """
    Envia os waypoints seguindo o protocolo do firmware:
      <C>            zera a lista
      <O,x,y,th>     alinha a odometria do robô ao referencial do tabuleiro
      <A,x,y,th>     um waypoint por comando (mensagens curtas)
      <R>            liga + auto-equilíbrio + navega
    Formato numérico: x,y em metros com 3 casas; th em graus com 1 casa.
    """

    # Converte np.float32 -> float nativo e normaliza theta para [-180, 180)
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
            resp = rpc.stop(ser)
            print(f"<C> -> {resp}")

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

            ok, resp, msg = rpc.start(ser, wait_auto_eq=True)
            print(f"<R> -> {resp} ({msg})")
            if not ok:
                return

            concluido = rpc.wait_until_done_telemetry(ser, poll_s=0.5, timeout_s=120.0)
            print("Status da missao:", "missão concluída" if concluido else "timeout de monitoramento")
        finally:
            ser.close()
    except Exception as e:
        print(f"Falha na comunicacao serial: {e}")

def interface_captura():
    global destino_pixel, inicio_pixel
    frame_capturado = None
    if len(sys.argv) > 1:
        f = cv2.imread(sys.argv[1])
        if f is None:
            return None, None
        cv2.namedWindow("Camera / Imagem")
        cv2.setMouseCallback("Camera / Imagem", capturar_clique)
        pos_robo = detectar_robo(f)
        if pos_robo: inicio_pixel = pos_robo
        while True:
            frame_display = f.copy()
            if inicio_pixel: cv2.circle(frame_display, inicio_pixel, 8, (0, 255, 0), -1)
            if destino_pixel: cv2.circle(frame_display, destino_pixel, 8, (0, 0, 255), -1)
            cv2.imshow("Camera / Imagem", frame_display)
            tecla = cv2.waitKey(1) & 0xFF
            if inicio_pixel and destino_pixel:
                frame_capturado = f
                break
            elif tecla == ord('q'):
                break
        cv2.destroyAllWindows()
        return frame_capturado, inicio_pixel
    else:
        cap = abrir_camera()
        if not cap.isOpened():
            return None, None
        cv2.namedWindow("Camera / Imagem")
        cv2.setMouseCallback("Camera / Imagem", capturar_clique)
        while True:
            ret, f = cap.read()
            if not ret or f is None:
                if cv2.waitKey(30) & 0xFF == ord('q'): break
                continue
            frame_display = f.copy()
            pos_robo = detectar_robo(f)
            if pos_robo:
                inicio_pixel = pos_robo
                cv2.circle(frame_display, pos_robo, 8, (0, 255, 0), -1)
            if destino_pixel:
                cv2.circle(frame_display, destino_pixel, 8, (0, 0, 255), -1)
            cv2.imshow("Camera / Imagem", frame_display)
            tecla = cv2.waitKey(1) & 0xFF
            if inicio_pixel and destino_pixel:
                frame_capturado = f
                break
            elif tecla == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        return frame_capturado, inicio_pixel

def main():
    frame, start = interface_captura()
    if frame is None or start is None or destino_pixel is None:
        return

    deteccoes = obter_deteccoes(frame, inicio=start, destino=destino_pixel)
    obstaculos_brutos = montar_obstaculos(deteccoes)
    obstaculos_expandidos = expandir_obstaculos_minkowski(obstaculos_brutos, frame.shape, RAIO_ROBO_PX)

    h, w = frame.shape[:2]
    bounds = (0, 0, w, h)
    goal = destino_pixel

    grid = build_grid(bounds, obstaculos_expandidos, resolution=RESOLUCAO)
    caminho_grade = plan_on_grid(grid, start, goal, resolution=RESOLUCAO, bounds=bounds)

    tree = subdivide(bounds, obstaculos_expandidos, MAX_DEPTH)
    caminho_qt, folhas_livres = plan_on_quadtree(tree, obstaculos_expandidos, start, goal)

    if caminho_qt:
        caminho_qt_limitado = limitar_waypoints(caminho_qt, max_pts=MAX_WAYPOINTS)
        H = calibrar_camera(frame)
        if H is not None:
            caminho_metros = pixel_para_real(caminho_qt_limitado, H)

            # Detecta se o referencial do tabuleiro é de mão esquerda (como a imagem)
            p0, px, py = pixel_para_real([start, (start[0] + 1, start[1]), (start[0], start[1] + 1)], H)
            du = (px[0] - p0[0], px[1] - p0[1])
            dv = (py[0] - p0[0], py[1] - p0[1])
            if du[0] * dv[1] - du[1] * dv[0] > 0:
                caminho_metros = [(x, -y) for x, y in caminho_metros]

            waypoints = converter_caminho_para_waypoints(caminho_metros)
            enviar_para_robo(waypoints, PORTA_SERIAL, BAUD_RATE)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, titulo, caminho in [(ax1, f"Grade regular ({RESOLUCAO}px)", caminho_grade),
                                (ax2, f"Quadtree ({len(folhas_livres)} folhas livres)", caminho_qt)]:
        ax.imshow(rgb)
        for poly in obstaculos_expandidos:
            ax.fill(poly[:, 0], poly[:, 1], color="red", alpha=0.3)
        for poly in obstaculos_brutos:
            ax.fill(poly[:, 0], poly[:, 1], color="darkred", alpha=0.7)
        if caminho:
            ax.plot(*zip(*caminho), "g-", linewidth=2)
        ax.plot(*start, "bo", markersize=8)
        ax.plot(*goal, "yo", markersize=8)
        ax.set_title(titulo)

    for f in folhas_livres:
        ax2.add_patch(plt.Rectangle((f[0], f[1]), f[2] - f[0], f[3] - f[1],
                                    fill=False, edgecolor="gray", linewidth=0.3))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
