import heapq
import sys

## O listing que tem em alguns comentários é pra indicar a qual pseudocodigo aquele trecho é referente

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.transforms import Bbox

from identificar_formas import (MIN_AREA, EPSILON_FACTOR, USAR_WEBCAM, IMAGEM_PATH,
                                classificar_forma, abrir_camera)

# --- Parâmetros do planejamento ---
RESOLUCAO = 15         # tamanho da célula da grade regular (pixels)
MAX_DEPTH = 7          # profundidade máxima da quadtree
START_FRAC = (0.05, 0.05)  # start/goal como fração da imagem (x, y)
GOAL_FRAC = (0.95, 0.95)

def obter_deteccoes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    deteccoes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, EPSILON_FACTOR * peri, True)
        if len(approx) > 12 or len(approx) < 3:  # < 3 vértices não é polígono
            continue
        nome, _ = classificar_forma(cnt, approx)
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        deteccoes.append((approx, (cx, cy), nome))
    return deteccoes

## Ponte entre os dois roteiros (Listing 1) ##########################################################
def regular_polygon(center, n, radius, rotation=0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])

def montar_obstaculos(deteccoes):
    # cada item contendo (approx, centroide, nome)
    obstaculos = []

    for approx, (cx, cy), nome in deteccoes:
        # Opcao 1: usar o proprio contorno aproximado como obstaculo (poligono irregular)
        obstaculos.append(approx.reshape(-1, 2).astype(float))

        # Opcao 2: reconstruir como poligono regular a partir do centroide e do raio
        # raio = max(np.linalg.norm(approx.reshape(-1, 2) - [cx, cy], axis=1))
        # obstaculos.append(regular_polygon((cx, cy), n=len(approx), radius=raio))
    return obstaculos

## Método 3: Implementação grade regular (Listing 6) ##########################################################
def build_grid(bounds, obstacles, resolution=0.1):
    xmin, ymin, xmax, ymax = bounds
    nx = int((xmax - xmin) / resolution)
    ny = int((ymax - ymin) / resolution)
    grid = np.zeros((nx, ny), dtype=np.uint8)  # 0=livre, 1=ocupado

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

    # Converte coordenadas para indices
    def to_idx(p):
        i = min(max(int((p[0] - xmin) / resolution), 0), nx - 1)
        j = min(max(int((p[1] - ymin) / resolution), 0), ny - 1)
        return (i, j)

    s, g = to_idx(start), to_idx(goal)
    if grid[s] or grid[g]:
        return None

    # Aplica A* sobre celulas livres com vizinhanca-8
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

    # Retorna sequencia de centros de celulas
    caminho = []
    no = g
    while no is not None:
        caminho.append((xmin + (no[0] + 0.5) * resolution, ymin + (no[1] + 0.5) * resolution))
        no = pai[no]
    return caminho[::-1]

## Método 3: Refinamento com Quadtree (Listing 7) ##########################################################
# Celula = (xmin, ymin, xmax, ymax). Folha = tupla; no misto = lista com os 4 filhos.
def subdivide(cell, obstacles, max_depth, depth=0):
    if depth >= max_depth or not cell_intersects_obstacles(cell, obstacles):
        return cell  # folha: livre ou ocupada
    # Divide em 4 quadrantes e recursivamente
    children = split_into_quadrants(cell)
    return [subdivide(c, obstacles, max_depth, depth + 1) for c in children]

def cell_intersects_obstacles(cell, obstacles):
    # A celula toca algum obstaculo (borda ou interior)?
    caixa = Bbox.from_extents(*cell)
    return any(Path(np.vstack([poly, poly[:1]]), closed=True).intersects_bbox(caixa, filled=True) for poly in obstacles)

def split_into_quadrants(cell):
    # Ordem do artigo: A (sup. esq.), B (sup. dir.), C (inf. dir.), D (inf. esq.)
    xmin, ymin, xmax, ymax = cell
    xm, ym = (xmin + xmax) / 2, (ymin + ymax) / 2
    return [(xmin, ymin, xm, ym), (xm, ymin, xmax, ym),
            (xm, ym, xmax, ymax), (xmin, ym, xm, ymax)]

def plan_on_quadtree(tree, obstacles, start, goal):
    # Folhas livres = nos do grafo de conectividade (artigo, secao 3)
    folhas = collect_leaves(tree)
    livres = [f for f in folhas if not cell_intersects_obstacles(f, obstacles)]
    centros = [((f[0] + f[2]) / 2, (f[1] + f[3]) / 2) for f in livres]

    def contem(f, p):
        return f[0] <= p[0] <= f[2] and f[1] <= p[1] <= f[3]

    s = next((k for k, f in enumerate(livres) if contem(f, start)), None)
    g = next((k for k, f in enumerate(livres) if contem(f, goal)), None)
    if s is None or g is None:
        return None, livres

    # Arestas = folhas livres que compartilham um lado; peso = distancia euclidiana entre centros
    def adjacentes(f1, f2):
        sobre_x = min(f1[2], f2[2]) - max(f1[0], f2[0])
        sobre_y = min(f1[3], f2[3]) - max(f1[1], f2[1])
        return (sobre_x > 0 and sobre_y == 0) or (sobre_y > 0 and sobre_x == 0)

    # A* sobre o grafo (artigo, secao 4: f = (1-w)g + w*h, w = 0.5 -> A* classico)
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

    # Retorna sequencia de centros das celulas (start -> ... -> goal)
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

## Integração com o mapa de polígonos // Funções auxiliares (Listing 9) ##########################################################
def point_in_polygon(pt, poly):
    return Path(poly).contains_point(pt)

def point_in_any_polygon(pt, obstacles):
    return any(point_in_polygon(pt, poly) for poly in obstacles)

def capturar_frame():
    # Imagem passada na linha de comando: python quadtree_planner.py testes/1.png
    if len(sys.argv) > 1:
        return cv2.imread(sys.argv[1])
    if not USAR_WEBCAM:
        return cv2.imread(IMAGEM_PATH)
    cap = abrir_camera()
    if not cap.isOpened():
        return None
    print("Pressione 'p' para capturar o frame e planejar, 'q' para sair.")
    frame = None
    while True:
        ret, f = cap.read()
        if not ret:  # câmeras via Wi-Fi podem falhar nas primeiras leituras
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            continue
        cv2.imshow("Camera do Robo", f)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('p'):
            frame = f
            break
        if tecla == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame


def main():
    frame = capturar_frame()
    if frame is None:
        print("Erro: nenhum frame capturado.")
        return

    # Camera -> deteccao de poligonos -> lista de obstaculos
    deteccoes = obter_deteccoes(frame)
    obstaculos = montar_obstaculos(deteccoes)
    print(f"{len(obstaculos)} obstaculos: {[nome for _, _, nome in deteccoes]}")

    h, w = frame.shape[:2]
    bounds = (0, 0, w, h)
    start = (START_FRAC[0] * w, START_FRAC[1] * h)
    goal = (GOAL_FRAC[0] * w, GOAL_FRAC[1] * h)

    # Planejamento: grade regular e quadtree
    grid = build_grid(bounds, obstaculos, resolution=RESOLUCAO)
    caminho_grade = plan_on_grid(grid, start, goal, resolution=RESOLUCAO, bounds=bounds)

    tree = subdivide(bounds, obstaculos, MAX_DEPTH)
    caminho_qt, folhas_livres = plan_on_quadtree(tree, obstaculos, start, goal)

    # Validacao visual com matplotlib
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for ax, titulo, caminho in [(ax1, f"Grade regular ({RESOLUCAO}px)", caminho_grade),
                                (ax2, f"Quadtree ({len(folhas_livres)} folhas livres)", caminho_qt)]:
        ax.imshow(rgb)
        for poly in obstaculos:
            ax.fill(poly[:, 0], poly[:, 1], color="red", alpha=0.4)
        if caminho:
            ax.plot(*zip(*caminho), "g-", linewidth=2)
        else:
            print(f"{titulo}: nenhum caminho encontrado")
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
