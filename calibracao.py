import os
import cv2
import numpy as np

def carregar_calibracao(arquivo="calibracao.npy"):
    if os.path.exists(arquivo):
        return np.load(arquivo)
    return None

def salvar_calibracao(H, arquivo="calibracao.npy"):
    np.save(arquivo, H)

def calibrar_camera(frame, tamanho_quadrado_m=0.025):
    H_salvo = carregar_calibracao()
    if H_salvo is not None:
        return H_salvo
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
            salvar_calibracao(H)
            return H
    return None

def pixel_para_real(pontos_pixels, H):
    if H is None or not pontos_pixels: return []
    pts = np.array(pontos_pixels, dtype=np.float32).reshape(-1, 1, 2)
    pts_reais = cv2.perspectiveTransform(pts, H)
    return [tuple(p[0]) for p in pts_reais]