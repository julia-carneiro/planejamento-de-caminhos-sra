import sys
import math
import heapq
import cv2
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.transforms import Bbox

from identificar_formas import (MIN_AREA, EPSILON_FACTOR, classificar_forma, abrir_camera)
from Remote_Position_Control_TEC220 import connect, run_mission

RESOLUCAO = 15
MAX_DEPTH = 7
PORTA_SERIAL = "/dev/rfcomm0"
BAUD_RATE = 115200
TESTE_SEM_ROBO = False
MAX_WAYPOINTS = 10

destino_pixel = None
inicio_pixel = None

def limitar_waypoints(caminho, max_pts=10):
    """
    Reduz a quantidade de pontos do caminho para no máximo 'max_pts',
    garantindo que o início e o final sejam preservados.
    """
    if not caminho or len(caminho) <= max_pts:
        return caminho

    # Gera índices distribuídos igualmente do primeiro ao último ponto
    indices = np.linspace(0, len(caminho) - 1, max_pts, dtype=int)
    return [caminho[i] for i in indices]

def capturar_clique(event, x, y, flags, param):
    global destino_pixel, inicio_pixel
    if event == cv2.EVENT_LBUTTONDOWN:
        destino_pixel = (x, y)
        print(f"Destino definido: {destino_pixel}")
    elif event == cv2.EVENT_RBUTTONDOWN:
        inicio_pixel = (x, y)
        print(f"Inicio manual definido: {inicio_pixel}")

def detectar_robo(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 140, 100]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 140, 100]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    possiveis_robos = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 30 < area < 3000:
            peri = cv2.arcLength(cnt, True)
            if peri > 0:
                circ = (4 * math.pi * area) / (peri ** 2)
                if circ > 0.4:
                    possiveis_robos.append(cnt)

    if possiveis_robos:
        melhor_contorno = max(possiveis_robos, key=cv2.contourArea)
        M = cv2.moments(melhor_contorno)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
    return None

def calibrar_camera(frame, tamanho_quadrado_m=0.025):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    padroes = [(8, 6), (8, 5), (9, 6), (7, 6)]

    for padrao in padroes:
        ret, corners = cv2.findChessboardCorners(gray, padrao, None)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            obj_points = np.zeros((padrao[0] * padrao[1], 2), np.float32)
            obj_points[:, :2] = np.mgrid[0:padrao[0], 0:padrao[1]].T.reshape(-1, 2)
            obj_points *= tamanho_quadrado_m

            H, _ = cv2.findHomography(corners, obj_points)
            print(f"Calibracao bem-sucedida usando padrao {padrao}")
            return H

    return None

def pixel_para_real(pontos_pixels, H):
    if H is None or not pontos_pixels: return []
    pts = np.array(pontos_pixels, dtype=np.float32).reshape(-1, 1, 2)
    pts_reais = cv2.perspectiveTransform(pts, H)
    return [tuple(p[0]) for p in pts_reais]

def converter_caminho_para_waypoints(caminho_metros):
    waypoints = []
    tamanho = len(caminho_metros)
    for i in range(tamanho):
        cx, cy = caminho_metros[i]
        if i < tamanho - 1:
            nx, ny = caminho_metros[i+1]
            theta = math.degrees(math.atan2(ny - cy, nx - cx))
        else:
            theta = 0.0
        waypoints.append((cx, cy, theta))
    return waypoints

def enviar_para_robo(waypoints, porta, baud):
    if TESTE_SEM_ROBO:
        print("\n[MODO TESTE] Waypoints gerados (x, y, theta):")
        for i, wp in enumerate(waypoints):
            print(f" WP {i}: x={wp[0]:.3f}m, y={wp[1]:.3f}m, th={wp[2]:.1f} graus")
        return

    try:
        ser = connect(port=porta, baud=baud)
        ok, msg = run_mission(ser, waypoints)
        print(f"Status da missao: {msg}")
    except Exception as e:
        print(f"Falha na comunicacao serial: {e}")

def obter_deteccoes(frame, inicio=None, destino=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    deteccoes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w > frame.shape[1] * 0.40 or h > frame.shape[0] * 0.40:
            continue

        if inicio is not None and cv2.pointPolygonTest(cnt, inicio, False) >= 0:
            continue
        if destino is not None and cv2.pointPolygonTest(cnt, destino, False) >= 0:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, EPSILON_FACTOR * peri, True)
        if len(approx) > 12 or len(approx) < 3:
            continue

        nome, _ = classificar_forma(cnt, approx)
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        deteccoes.append((approx, (cx, cy), nome))

    return deteccoes

def montar_obstaculos(deteccoes):
    obstaculos = []
    for approx, _, _ in deteccoes:
        obstaculos.append(approx.reshape(-1, 2).astype(float))
    return obstaculos

def build_grid(bounds, obstacles, resolution=0.1):
    xmin, ymin, xmax, ymax = bounds
    nx = int((xmax - xmin) / resolution)
    ny = int((ymax - ymin) / resolution)
    grid = np.zeros((nx, ny), dtype=np.uint8)

    for i in range(nx):
        for j in range(ny):
            cx = xmin + (i + 0.5) * resolution
            cy = ymin + (j + 0.5) * resolution
            if point_in_any_polygon((cx, cy), obstacles):
                grid[i, j] = 1
    return grid

def plan_on_grid(grid, start, goal, resolution=0.1, bounds=None):
    xmin, ymin = (bounds[0], bounds[1]) if bounds is not None else (0.0, 0.0)
    nx, ny = grid.shape

    def to_idx(p):
        i = min(max(int((p[0] - xmin) / resolution), 0), nx - 1)
        j = min(max(int((p[1] - ymin) / resolution), 0), ny - 1)
        return (i, j)

    s, g = to_idx(start), to_idx(goal)
    if grid[s] or grid[g]:
        return None

    vizinhanca_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    custo = {s: 0.0}
    pai = {s: None}
    fila = [(0.0, s)]
    fechados = set()
    while fila:
        _, atual = heapq.heappop(fila)
        if atual in fechados:
            continue
        fechados.add(atual)
        if atual == g:
            break
        for di, dj in vizinhanca_8:
            viz = (atual[0] + di, atual[1] + dj)
            if not (0 <= viz[0] < nx and 0 <= viz[1] < ny) or grid[viz] == 1:
                continue
            c = custo[atual] + np.hypot(di, dj)
            if c < custo.get(viz, np.inf):
                custo[viz] = c
                pai[viz] = atual
                h = np.hypot(viz[0] - g[0], viz[1] - g[1])
                heapq.heappush(fila, (c + h, viz))
    if g not in pai:
        return None

    caminho = []
    no = g
    while no is not None:
        caminho.append((xmin + (no[0] + 0.5) * resolution, ymin + (no[1] + 0.5) * resolution))
        no = pai[no]
    return caminho[::-1]

def subdivide(cell, obstacles, max_depth, depth=0):
    if depth >= max_depth or not cell_intersects_obstacles(cell, obstacles):
        return cell
    children = split_into_quadrants(cell)
    return [subdivide(c, obstacles, max_depth, depth + 1) for c in children]

def cell_intersects_obstacles(cell, obstacles):
    caixa = Bbox.from_extents(*cell)
    return any(Path(np.vstack([poly, poly[:1]]), closed=True).intersects_bbox(caixa, filled=True) for poly in obstacles)

def split_into_quadrants(cell):
    xmin, ymin, xmax, ymax = cell
    xm, ym = (xmin + xmax) / 2, (ymin + ymax) / 2
    return [(xmin, ymin, xm, ym), (xm, ymin, xmax, ym),
            (xm, ym, xmax, ymax), (xmin, ym, xm, ymax)]

def plan_on_quadtree(tree, obstacles, start, goal):
    folhas = collect_leaves(tree)
    livres = [f for f in folhas if not cell_intersects_obstacles(f, obstacles)]
    centros = [((f[0] + f[2]) / 2, (f[1] + f[3]) / 2) for f in livres]

    def contem(f, p):
        return f[0] <= p[0] <= f[2] and f[1] <= p[1] <= f[3]

    s = next((k for k, f in enumerate(livres) if contem(f, start)), None)
    g = next((k for k, f in enumerate(livres) if contem(f, goal)), None)
    if s is None or g is None:
        return None, livres

    def adjacentes(f1, f2):
        sobre_x = min(f1[2], f2[2]) - max(f1[0], f2[0])
        sobre_y = min(f1[3], f2[3]) - max(f1[1], f2[1])
        return (sobre_x > 0 and sobre_y == 0) or (sobre_y > 0 and sobre_x == 0)

    custo = {s: 0.0}
    pai = {s: None}
    fila = [(0.0, s)]
    fechados = set()
    while fila:
        _, atual = heapq.heappop(fila)
        if atual in fechados:
            continue
        fechados.add(atual)
        if atual == g:
            break
        for viz in range(len(livres)):
            if viz in fechados or not adjacentes(livres[atual], livres[viz]):
                continue
            c = custo[atual] + np.linalg.norm(np.subtract(centros[viz], centros[atual]))
            if c < custo.get(viz, np.inf):
                custo[viz] = c
                pai[viz] = atual
                h = np.linalg.norm(np.subtract(centros[viz], centros[g]))
                heapq.heappush(fila, (c + h, viz))
    if g not in pai:
        return None, livres

    caminho = [tuple(goal)]
    no = pai[g]
    while no is not None and no != s:
        caminho.append(centros[no])
        no = pai[no]
    caminho.append(tuple(start))
    return caminho[::-1], livres

def collect_leaves(tree):
    if isinstance(tree, tuple):
        return [tree]
    return [folha for filho in tree for folha in collect_leaves(filho)]

def point_in_polygon(pt, poly):
    return Path(poly).contains_point(pt)

def point_in_any_polygon(pt, obstacles):
    return any(point_in_polygon(pt, poly) for poly in obstacles)

def interface_captura():
    global destino_pixel, inicio_pixel
    frame_capturado = None

    if len(sys.argv) > 1:
        f = cv2.imread(sys.argv[1])
        if f is None:
            print(f"Erro ao carregar imagem: {sys.argv[1]}")
            return None, None
        cv2.namedWindow("Camera / Imagem")
        cv2.setMouseCallback("Camera / Imagem", capturar_clique)

        pos_robo = detectar_robo(f)
        if pos_robo:
            inicio_pixel = pos_robo

        while True:
            frame_display = f.copy()
            if inicio_pixel:
                cv2.circle(frame_display, inicio_pixel, 8, (0, 255, 0), -1)
            if destino_pixel:
                cv2.circle(frame_display, destino_pixel, 8, (0, 0, 255), -1)

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
            print("Erro ao abrir a câmera.")
            return None, None

        cv2.namedWindow("Camera / Imagem")
        cv2.setMouseCallback("Camera / Imagem", capturar_clique)

        while True:
            ret, f = cap.read()
            if not ret or f is None:
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
                continue

            frame_display = f.copy()
            pos_robo = detectar_robo(f)

            if pos_robo:
                inicio_pixel = pos_robo
                cv2.circle(frame_display, pos_robo, 8, (0, 255, 0), -1)

            if destino_pixel:
                cv2.circle(frame_display, destino_pixel, 8, (0, 0, 255), -1)

            # Exibe o frame atualizado e permite ler o clique do mouse e a tecla 'q'
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
        print("Cancelado ou dados incompletos.")
        return

    deteccoes = obter_deteccoes(frame, inicio=start, destino=destino_pixel)
    obstaculos = montar_obstaculos(deteccoes)

    h, w = frame.shape[:2]
    bounds = (0, 0, w, h)
    goal = destino_pixel

    grid = build_grid(bounds, obstaculos, resolution=RESOLUCAO)
    caminho_grade = plan_on_grid(grid, start, goal, resolution=RESOLUCAO, bounds=bounds)

    tree = subdivide(bounds, obstaculos, MAX_DEPTH)
    caminho_qt, folhas_livres = plan_on_quadtree(tree, obstaculos, start, goal)

    if caminho_qt:
        caminho_qt_limitado = limitar_waypoints(caminho_qt, max_pts=MAX_WAYPOINTS)
        H = calibrar_camera(frame)
        if H is not None:
            caminho_metros = pixel_para_real(caminho_qt_limitado, H)
            waypoints = converter_caminho_para_waypoints(caminho_metros)
            enviar_para_robo(waypoints, PORTA_SERIAL, BAUD_RATE)
        else:
            print("Aviso: Quadriculado de calibracao nao encontrado. Waypoints nao convertidos.")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, titulo, caminho in [(ax1, f"Grade regular ({RESOLUCAO}px)", caminho_grade),
                                (ax2, f"Quadtree ({len(folhas_livres)} folhas livres)", caminho_qt)]:
        ax.imshow(rgb)
        for poly in obstaculos:
            ax.fill(poly[:, 0], poly[:, 1], color="red", alpha=0.4)
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
