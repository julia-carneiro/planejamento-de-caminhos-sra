import math
import heapq
import numpy as np
from matplotlib.path import Path
from matplotlib.transforms import Bbox

def limitar_waypoints(caminho, max_pts=10):
    if not caminho or len(caminho) <= max_pts:
        return caminho
    indices = np.linspace(0, len(caminho) - 1, max_pts, dtype=int)
    return [caminho[i] for i in indices]

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
        waypoints.append((float(cx), float(cy), float(theta)))
    return waypoints

def point_in_polygon(pt, poly):
    return Path(poly).contains_point(pt)

def point_in_any_polygon(pt, obstacles):
    return any(point_in_polygon(pt, poly) for poly in obstacles)

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
        if atual in fechados: continue
        fechados.add(atual)
        if atual == g: break
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

def collect_leaves(tree):
    if isinstance(tree, tuple):
        return [tree]
    return [folha for filho in tree for folha in collect_leaves(filho)]

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
        if atual in fechados: continue
        fechados.add(atual)
        if atual == g: break
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