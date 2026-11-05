import cv2
import numpy as np
import math
import platform

MIN_AREA = 500
EPSILON_FACTOR = 0.018
CAM_INDEX = 0

COMPRIMENTO_MESA_M = 2.5
LARGURA_MESA_M = 1.2

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

def detectar_robo(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 140, 100]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 140, 100]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidatos = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 30 < area < 3000:
            peri = cv2.arcLength(cnt, True)
            if peri > 0 and (4 * math.pi * area) / (peri ** 2) > 0.4:
                candidatos.append(cnt)
    if candidatos:
        M = cv2.moments(max(candidatos, key=cv2.contourArea))
        if M["m00"] > 0:
            return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return None

def detectar_obstaculos(frame, inicio=None, destino=None, raio_exclusao_robo=0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    obstaculos = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
        _, _, w, h = cv2.boundingRect(cnt)
        if w > frame.shape[1] * 0.40 or h > frame.shape[0] * 0.40:
            continue
        if inicio is not None and cv2.pointPolygonTest(cnt, inicio, False) >= 0:
            continue
        # contorno dentro/tocando o disco do robô -> é o próprio robô, não obstáculo
        if inicio is not None and raio_exclusao_robo > 0:
            if cv2.pointPolygonTest(cnt, inicio, True) >= -raio_exclusao_robo:
                continue
            if np.min(np.linalg.norm(cnt.reshape(-1, 2) - np.array(inicio), axis=1)) <= raio_exclusao_robo:
                continue
        if destino is not None and cv2.pointPolygonTest(cnt, destino, False) >= 0:
            continue
        approx = cv2.approxPolyDP(cnt, EPSILON_FACTOR * cv2.arcLength(cnt, True), True)
        if 3 <= len(approx) <= 12:
            obstaculos.append(approx.reshape(-1, 2).astype(float))
    return obstaculos

def expandir_obstaculos_minkowski(obstaculos_originais, shape, raio_px):
    if not obstaculos_originais:
        return []
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(p, dtype=np.int32) for p in obstaculos_originais], 255)
    disco = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * raio_px + 1, 2 * raio_px + 1))
    mask = cv2.dilate(mask, disco, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    expandidos = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) >= 3:
            expandidos.append(approx.reshape(-1, 2).astype(float))
    return expandidos

def detectar_mesa(frame, area_min_frac=0.2):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, claro = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    claro = cv2.morphologyEx(claro, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    contours, _ = cv2.findContours(claro, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    maior = max(contours, key=cv2.contourArea)
    if cv2.contourArea(maior) < area_min_frac * frame.shape[0] * frame.shape[1]:
        return None
    return cv2.convexHull(maior)

def obstaculos_fora_da_mesa(contorno_mesa, shape, margem_px=0):
    h, w = shape[:2]
    mesa = np.zeros((h, w), np.uint8)
    cv2.drawContours(mesa, [contorno_mesa], -1, 255, cv2.FILLED)
    if margem_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margem_px + 1, 2 * margem_px + 1))
        mesa = cv2.erode(mesa, k)
    fora = cv2.bitwise_not(mesa)

    linhas = np.where(mesa.any(axis=1))[0]
    if len(linhas) == 0:
        return [np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)]
    cy = int((linhas[0] + linhas[-1]) / 2)

    poligonos = []
    for y0, y1 in [(0, cy + 3), (cy - 3, h)]:
        parte = np.zeros_like(fora)
        parte[y0:y1] = fora[y0:y1]
        cnts, _ = cv2.findContours(parte, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < 20:
                continue
            approx = cv2.approxPolyDP(c, 2.0, True)
            if len(approx) >= 3:
                poligonos.append(approx.reshape(-1, 2).astype(float))
    return poligonos

def escala_pela_mesa(contorno_mesa):
    _, (a, b), _ = cv2.minAreaRect(contorno_mesa)
    return max(a, b) / COMPRIMENTO_MESA_M, min(a, b) / LARGURA_MESA_M
