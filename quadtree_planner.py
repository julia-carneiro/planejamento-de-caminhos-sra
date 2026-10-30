import sys
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from identificar_formas import abrir_camera, detectar_robo, obter_deteccoes, montar_obstaculos, expandir_obstaculos_minkowski
from calibracao import calibrar_camera, pixel_para_real
from planejamento import build_grid, plan_on_grid, subdivide, plan_on_quadtree, limitar_waypoints, converter_caminho_para_waypoints
from Remote_Position_Control_TEC220 import connect, run_mission

RESOLUCAO = 15
MAX_DEPTH = 7
PORTA_SERIAL = "/dev/rfcomm0"
BAUD_RATE = 115200
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

def enviar_para_robo(waypoints, porta, baud):
    if TESTE_SEM_ROBO:
        return
    try:
        ser = connect(port=porta, baud=baud)
        ok, msg = run_mission(ser, waypoints)
    except Exception as e:
        pass

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