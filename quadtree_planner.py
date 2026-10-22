import numpy as np
from matplotlib.path import Path

## Representação do mapa com polígonos regulares ##########################################################
def regular_polygon(center, n, radius, rotation=0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation

    return np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])

obstaculos = []

for aprox, (cx, cy), nome in deteccoes:
    #opcao 1: usar o proprio contorno aproximado como obstaculo
    #obstaculos.append(approx.reshape(-1, 2).astype(float))

    #opcao 2: reconstruir como poligono regular a partir do centroide e do raio
    raio = max(np.linalg.norm(approx - [cx, cy], axis=1))
    obstaculos.append(regular_polygon((cx, cy), n=len(approx), radius=raio))

## Implementação grade regular ##########################################################

def build_grid(bounds, obstacles, resolution=0.1):
    xmin, ymin, xmax, ymax = bounds
    nx = int((xmax - xmin) / resolution)
    ny = int((ymax - ymin) / resolution)
    grid = np.zeros((nx, ny), dtype=np.uint8) # 0=livre, 1=ocupado

    for i in range(nx):
        for j in range(ny):
            cx = xmin + (i + 0.5) * resolution
            cy = ymin + (j + 0.5) * resolution
            if point_in_any_polygon((cx, cy), obstacles):
                grid[i, j] = 1
    return grid

def plan_on_grid(grid, start, goal, resolution=0.1, bounds=None):
    # Converte coordenadas para indices
    # Aplica BFS/A* sobre celulas livres com vizinhança -8
    # Retorna sequencia de centros de celulas
    pass

def subdivide(cell, obstacles, max_depth, depth=0):
    if depth >= max_depth or not cell_intersects_obstacles(cell, obstacles):
        return cell # folha: livre ou ocupada
    children = split_into_quandrants(cell)
    return [subdivide(c, obstacles, max_depth, depth+1) for c in children]

def cell_intersects_obstacles():
    pass

def split_into_quandrants():
    pass

## Integração com o mapa de polígonos regulares // Funções auxiliares ##########################################################

def point_in_polygon(pt, poly):
    return Path(poly).contains_point(pt)

def point_in_any_polygon(pt, obstacles):
    return any(point_in_polygon(pt, poly) for poly in obstacles)

def dist_point_polygon(pt, poly):
    #Distancia minima do ponto ao poligono
    #(0, se dentro; senao, distancia ao segmento mais proximo)
    if point_in_polygon():
        return 0.0
    dmin = np.inf
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        d = dist_point_segment(pt, a, b)
        dmin = min(dmin, d)
    return dmin

def dist_point_segment(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
    proj = a + t * ab
    return np.linalg.norm(p - proj)
