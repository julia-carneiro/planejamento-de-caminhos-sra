import json
import os
import cv2
import numpy as np

TAMANHO_QUADRADO_M = 0.024

PADROES = [(9, 6), (6, 9), (8, 6), (7, 6), (8, 5), (7, 5), (6, 5), (9, 7), (7, 4)]


def _achar_cantos(gray, padrao):
    if hasattr(cv2, "findChessboardCornersSB"):
        ret, corners = cv2.findChessboardCornersSB(gray, padrao)
        if ret:
            return corners
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, padrao, flags)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        return cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    return None


def medir_px_por_metro(frame, tamanho_quadrado_m=TAMANHO_QUADRADO_M):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for padrao in PADROES:
        corners = _achar_cantos(gray, padrao)
        if corners is None:
            continue
        grade = corners.reshape(padrao[1], padrao[0], 2)
        dist_h = np.linalg.norm(np.diff(grade, axis=1), axis=2)
        dist_v = np.linalg.norm(np.diff(grade, axis=0), axis=2)
        px_por_quadrado = np.mean(np.concatenate([dist_h.ravel(), dist_v.ravel()]))
        px_por_m = px_por_quadrado / tamanho_quadrado_m
        print(f"Calibracao pelo tabuleiro (padrao {padrao}): {px_por_m / 100:.2f} px/cm")
        return float(px_por_m)
    return None


ARQUIVO_CALIBRACAO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibracao.json")

def _carregar_calibracao(shape):
    try:
        with open(ARQUIVO_CALIBRACAO) as f:
            d = json.load(f)
        if d.get("shape") == list(shape[:2]):
            return float(d["px_por_m"])
        print("Aviso: calibracao salva e de outra resolucao; recalibrando.")
    except (OSError, ValueError, KeyError):
        pass
    return None


def calibrar_camera(frame, escala_mesa=None, recalibrar=False):
    """Usa a calibracao salva (calibracao.json); so mede o tabuleiro se nao existir
    ou se recalibrar=True. A escala da mesa e so fallback e nao e salva."""
    if not recalibrar:
        px_por_m = _carregar_calibracao(frame.shape)
        if px_por_m is not None:
            print(f"Calibracao carregada de {os.path.basename(ARQUIVO_CALIBRACAO)}: {px_por_m / 100:.2f} px/cm")
            return px_por_m
    px_por_m = medir_px_por_metro(frame)
    if px_por_m is not None:
        with open(ARQUIVO_CALIBRACAO, "w") as f:
            json.dump({"px_por_m": px_por_m, "shape": list(frame.shape[:2])}, f)
        print(f"Calibracao salva em {os.path.basename(ARQUIVO_CALIBRACAO)}.")
        return px_por_m
    if escala_mesa is not None:
        print(f"Aviso: tabuleiro nao encontrado; usando escala da mesa ({escala_mesa / 100:.2f} px/cm).")
        return float(escala_mesa)
    return None


def pixel_para_real(pontos_pixels, px_por_m, origem, frente=(-1.0, 0.0)):
    """
    Converte pixels -> metros no referencial do robô (X = frente, Y = esquerda).
    `frente` = vetor unitário (u, v) na imagem para onde o robô aponta no início
    (v cresce para baixo). Ex.: (-1, 0) esquerda da tela, (0, -1) cima, (1, 0) direita.
    Esquerda do robô na imagem = (fv, -fu).
    """
    u0, v0 = origem
    fu, fv = frente
    n = (fu ** 2 + fv ** 2) ** 0.5
    fu, fv = fu / n, fv / n
    lu, lv = fv, -fu
    return [(((u - u0) * fu + (v - v0) * fv) / px_por_m,
             ((u - u0) * lu + (v - v0) * lv) / px_por_m) for u, v in pontos_pixels]
