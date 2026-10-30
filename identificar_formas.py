import cv2
import numpy as np
import math
import os
import platform

MIN_AREA = 500
EPSILON_FACTOR = 0.018
USAR_WEBCAM = True
CAM_INDEX = 0

GRID_SIZE = (128, 128)
IMAGEM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "placeholder_formas.jpeg")

CORES = {
    "amarelo": (0, 255, 255), "azul": (255, 0, 0), "verde": (0, 255, 0),
    "laranja": (0, 165, 255), "magenta": (255, 0, 255), "vermelho": (0, 0, 255),
    "roxo": (128, 0, 128), "ciano": (255, 255, 0), "cinza": (128, 128, 128)
}

def abrir_camera(indice=CAM_INDEX, largura=640, altura=480):
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(indice, cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def classificar_forma(cnt, approx):
    v = len(approx)
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return f"poligono ({v})", CORES["cinza"]
    circ = (4 * math.pi * area) / (peri ** 2)
    if v >= 8 and circ > 0.85: return "circulo", CORES["amarelo"]
    elif v == 3: return "triangulo", CORES["azul"]
    elif v == 4:
        x, y, w, h = cv2.boundingRect(approx)
        prop = w / float(h)
        if 0.95 <= prop <= 1.05: return "quadrado", CORES["verde"]
        else: return "retangulo", CORES["laranja"]
    elif v == 5: return "pentagono", CORES["magenta"]
    elif v == 6: return "hexagono", CORES["vermelho"]
    elif v == 7: return "heptagono", CORES["roxo"]
    else: return f"poligono ({v})", CORES["cinza"]

def gerar_mapa_ocupacao(shape_original, contornos_obstaculos):
    mascara_vazia = np.zeros(shape_original[:2], dtype=np.uint8)
    cv2.drawContours(mascara_vazia, contornos_obstaculos, -1, 255, thickness=cv2.FILLED)
    mapa_redimensionado = cv2.resize(mascara_vazia, GRID_SIZE, interpolation=cv2.INTER_NEAREST)
    matriz_ocupacao = (mapa_redimensionado > 0).astype(np.uint8)
    return matriz_ocupacao, mapa_redimensionado

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

def expandir_obstaculos_minkowski(obstaculos_originais, shape, raio_px):
    if not obstaculos_originais:
        return []
    mask = np.zeros(shape[:2], dtype=np.uint8)
    int_obstaculos = [np.array(poly, dtype=np.int32) for poly in obstaculos_originais]
    cv2.fillPoly(mask, int_obstaculos, 255)
    kernel_size = int(raio_px * 2 + 1)
    kernel_disco = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_expandida = cv2.dilate(mask, kernel_disco, iterations=1)
    contours, _ = cv2.findContours(mask_expandida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obstaculos_expandidos = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
        if len(approx) >= 3:
            obstaculos_expandidos.append(approx.reshape(-1, 2).astype(float))
    return obstaculos_expandidos