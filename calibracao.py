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


def calibrar_camera(frame, escala_mesa=None):
    px_por_m = medir_px_por_metro(frame)
    if px_por_m is not None:
        return px_por_m
    if escala_mesa is not None:
        print(f"Aviso: tabuleiro nao encontrado; usando escala da mesa ({escala_mesa / 100:.2f} px/cm).")
        return float(escala_mesa)
    return None


def pixel_para_real(pontos_pixels, px_por_m, origem):
    u0, v0 = origem
    return [((u - u0) / px_por_m, -(v - v0) / px_por_m) for u, v in pontos_pixels]
