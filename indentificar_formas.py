import cv2
import numpy as np
import math

# --- Configurações ---
MIN_AREA = 500
EPSILON_FACTOR = 0.02
USAR_WEBCAM = True
CAM_INDEX = 2

def classificar_forma(cnt, approx):
    v = len(approx)
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)

    # Evita divisão por zero
    if peri == 0:
        return "desconhecido", (128, 128, 128)

    circ = (4 * math.pi * area) / (peri ** 2)

    # Cores em formato BGR (Azul, Verde, Vermelho)
    if v >= 8 and circ > 0.85:
        return "circulo", (0, 255, 255)       # Amarelo
    if v == 3:
        return "triangulo", (255, 0, 0)       # Azul
    elif v == 4:
        x, y, w, h = cv2.boundingRect(approx)
        prop = w / float(h)
        if 0.95 <= prop <= 1.05:
            return "quadrado", (0, 255, 0)    # Verde
        else:
            return "retangulo", (0, 165, 255) # Laranja
    elif v == 5:
        return "pentagono", (255, 0, 255)     # Magenta
    elif v == 6:
        return "hexagono", (0, 0, 255)        # Vermelho
    elif v == 7:
        return "heptagono", (128, 0, 128)     # Roxo
    elif v == 8:
        return "octogono", (255, 255, 0)      # Ciano
    else:
        return f"poligono({v})", (128, 128, 128) # Cinza

def processar_frame(frame):
    # Converter para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur Gaussiano
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny
    edges = cv2.Canny(blur, 50, 150)

    # Dilatação e Fechamento
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, EPSILON_FACTOR * peri, True)

        if len(approx) > 12:
            continue

        nome, cor = classificar_forma(cnt, approx)

        # Desenhar contorno (cinza, linha fina)
        cv2.drawContours(frame, [cnt], -1, (128, 128, 128), 1)

        # Desenhar polígono aproximado (cor classificada, linha grossa)
        cv2.drawContours(frame, [approx], -1, cor, 3)

        # Calcular centroide
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Desenhar círculo no centroide e rótulo
            cv2.circle(frame, (cx, cy), 4, (0, 0, 0), -1)
            rotulo = f"{nome} ({len(approx)}v)"
            cv2.putText(frame, rotulo, (cx - 20, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, rotulo, (cx - 20, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1)

    return frame

def main():
    if not USAR_WEBCAM:
        # Modo Imagem
        imagem_path = "placeholder_formas.jpeg"
        frame = cv2.imread(imagem_path)
        if frame is None:
            print(f"Erro: Não foi possível carregar a imagem {imagem_path}")
            return

        frame_processado = processar_frame(frame)
        cv2.imshow("Classificacao de Formas", frame_processado)
        cv2.waitKey(0)

    else:
        # Modo Webcam
        cap = cv2.VideoCapture(CAM_INDEX)

        # Configurar resolução (ex: 1920x1080)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        if not cap.isOpened():
            print("Erro: Não foi possível abrir a câmera.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_processado = processar_frame(frame)
            cv2.imshow("Classificacao de Formas", frame_processado)

            # Pressione 'q' para sair
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
