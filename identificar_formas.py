import cv2
import numpy as np
import math

# --- Parâmetros Críticos ---
MIN_AREA = 500

EPSILON_FACTOR = 0.018
USAR_WEBCAM = True
CAM_INDEX = 2

GRID_SIZE = (128, 128)

CORES = {
    "amarelo": (0, 255, 255),
    "azul": (255, 0, 0),
    "verde": (0, 255, 0),
    "laranja": (0, 165, 255),
    "magenta": (255, 0, 255),
    "vermelho": (0, 0, 255),
    "roxo": (128, 0, 128),
    "ciano": (255, 255, 0),
    "cinza": (128, 128, 128)
}

def classificar_forma(cnt, approx):
    """Algoritmo 2: Classificação da forma a partir do contorno e aproximação."""
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
    """Gera matriz binarizada de ocupação e visualização do mapa."""
    mascara_vazia = np.zeros(shape_original[:2], dtype=np.uint8)
    cv2.drawContours(mascara_vazia, contornos_obstaculos, -1, 255, thickness=cv2.FILLED)

    mapa_redimensionado = cv2.resize(mascara_vazia, GRID_SIZE, interpolation=cv2.INTER_NEAREST)
    matriz_ocupacao = (mapa_redimensionado > 0).astype(np.uint8)

    return matriz_ocupacao, mapa_redimensionado

def processar_frame(frame):
    """Algoritmo 1: Pre-processamento, segmentação e anotação."""
    # Pré-processamento
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Segmentação
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos_validos = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, EPSILON_FACTOR * peri, True)

        if len(approx) > 12:
            continue

        contornos_validos.append(approx)

        # Classificação
        nome, cor = classificar_forma(cnt, approx)
        v = len(approx)

        cv2.drawContours(frame, [cnt], -1, CORES["cinza"], 1)
        cv2.drawContours(frame, [approx], -1, cor, 3)

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 4, (0, 0, 0), -1)
            rotulo = f"{nome} ({v}v)"
            cv2.putText(frame, rotulo, (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

    matriz, mapa_visual = gerar_mapa_ocupacao(frame.shape, contornos_validos)
    return frame, mapa_visual, matriz

def main():
    if not USAR_WEBCAM:
        imagem_path = "/home/jcgsouza/UEFS/SRA_p2/placeholder_formas.jpeg"
        frame = cv2.imread(imagem_path)
        if frame is None:
            print(f"Erro: Não foi possível carregar a imagem {imagem_path}")
            return

        frame_anotado, mapa_visual, _ = processar_frame(frame)
        cv2.imshow("Câmera do Robô - Formas", frame_anotado)
        cv2.imshow("Mapa de Ocupacao Binarizado (128x128)", mapa_visual)
        cv2.waitKey(0)
    else:
        cap = cv2.VideoCapture(CAM_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        if not cap.isOpened():
            print("Erro: Não foi possível abrir a câmera.")
            return

        print("Pressione 'q' para sair.")
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame_anotado, mapa_visual, _ = processar_frame(frame)
            cv2.imshow("Câmera do Robô - Formas", frame_anotado)
            cv2.imshow("Mapa de Ocupacao Binarizado (128x128)", mapa_visual)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
