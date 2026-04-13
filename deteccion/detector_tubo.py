"""
detector_tubo.py — Detección del tubo amarillo (acceso mundo submarino) en Banana Kong
Resolución BlueStacks: 960x540

Estrategia: Template matching con alpha.
El tubo es amarillo intenso y muy distintivo visualmente.
"""

import cv2
import numpy as np
import pygetwindow as gw
from mss import mss
import time
import os

UMBRAL        = 0.8
ROI           = (0, 300, 960, 510)
ESCALAS       = [0.8, 0.9, 1.0, 1.1, 1.2]
FACTOR_RESIZE = 0.5
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")


class DetectorTubo:
    def __init__(self):
        self.titulo  = "BlueStacks"
        self.ventana = None
        self.sct     = mss()
        self.actualizar_ventana()

        ruta = os.path.join(TEMPLATES_DIR, "tubo-bg.png")
        img  = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"No se encontró: {ruta}")
        if img.shape[2] == 4:
            self.alpha    = img[:, :, 3]
            self.template = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            self.template = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.alpha    = None
        self.template = cv2.resize(self.template, (0,0), fx=FACTOR_RESIZE, fy=FACTOR_RESIZE)
        if self.alpha is not None:
            self.alpha = cv2.resize(self.alpha, (0,0), fx=FACTOR_RESIZE, fy=FACTOR_RESIZE)
        print(f"✅ Template tubo: {self.template.shape[1]}x{self.template.shape[0]}px")

    def actualizar_ventana(self):
        ventanas = gw.getWindowsWithTitle(self.titulo)
        if ventanas:
            self.ventana = ventanas[0]
            self.monitor = {
                "top": self.ventana.top, "left": self.ventana.left,
                "width": self.ventana.width, "height": self.ventana.height,
            }
            return True
        return False

    def capturar_pantalla(self):
        if self.ventana is None:
            if not self.actualizar_ventana():
                return None
        try:
            self.monitor = {
                "top": self.ventana.top, "left": self.ventana.left,
                "width": self.ventana.width, "height": self.ventana.height,
            }
        except Exception:
            self.ventana = None
            return None
        screenshot = self.sct.grab(self.monitor)
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)

    def detectar_tubo(self, frame):
        """
        Retorna:
            hay_tubo  : bool
            cx, cy    : posición normalizada [0,1] o None
            rect      : (x, y, w, h) en píxeles o None
            frame_r   : frame anotado
        """
        if frame is None:
            return False, None, None, frame

        x0, y0, x1, y1 = ROI
        roi       = frame[y0:y1, x0:x1]
        roi_small = cv2.resize(roi, (0,0), fx=FACTOR_RESIZE, fy=FACTOR_RESIZE)
        roi_gris  = cv2.cvtColor(roi_small, cv2.COLOR_BGR2GRAY)
        frame_r   = frame.copy()
        factor_inv = 1.0 / FACTOR_RESIZE
        cv2.rectangle(frame_r, (x0, y0), (x1, y1), (255, 255, 0), 1)  # ROI visible

        mejor_val, mejor_loc, mejor_nw, mejor_nh = 0, None, 0, 0
        h_t, w_t = self.template.shape

        for escala in ESCALAS:
            nw, nh = int(w_t * escala), int(h_t * escala)
            if nw >= roi_gris.shape[1] or nh >= roi_gris.shape[0]:
                continue
            t_s = cv2.resize(self.template, (nw, nh))
            if self.alpha is not None:
                a_s = cv2.resize(self.alpha, (nw, nh))
                res = cv2.matchTemplate(roi_gris, t_s, cv2.TM_CCOEFF_NORMED, mask=a_s)
            else:
                res = cv2.matchTemplate(roi_gris, t_s, cv2.TM_CCOEFF_NORMED)
            _, val, _, loc = cv2.minMaxLoc(res)
            if np.isfinite(val) and val > mejor_val:
                mejor_val, mejor_loc, mejor_nw, mejor_nh = val, loc, nw, nh

        if mejor_val < UMBRAL or mejor_loc is None:
            cv2.putText(frame_r, "Tubo: NO", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
            return False, None, None, frame_r

        x_real  = int(x0 + mejor_loc[0] * factor_inv)
        y_real  = int(y0 + mejor_loc[1] * factor_inv)
        bw_real = int(mejor_nw * factor_inv)
        bh_real = int(mejor_nh * factor_inv)
        cx   = (x_real + bw_real/2) / frame.shape[1]
        cy   = (y_real + bh_real/2) / frame.shape[0]
        rect = (x_real, y_real, bw_real, bh_real)

        cv2.rectangle(frame_r, (x_real, y_real),
                      (x_real+bw_real, y_real+bh_real), (0, 255, 255), 2)
        cv2.putText(frame_r, f"TUBO {mejor_val:.2f}", (x_real, y_real-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return True, (cx, cy), rect, frame_r

    def probar(self):
        print("=== DETECTOR DE TUBO ===  q=salir | s=guardar")
        time.sleep(2)
        cv2.namedWindow("Tubo Detector")
        while True:
            frame = self.capturar_pantalla()
            if frame is None:
                time.sleep(1)
                continue
            hay_tubo, pos, rect, frame_r = self.detectar_tubo(frame)
            if hay_tubo:
                print(f"🟡 Tubo detectado en cx={pos[0]:.2f} cy={pos[1]:.2f}")
            cv2.imshow("Tubo Detector", frame_r)
            cv2.moveWindow("Tubo Detector", 100, 100)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                cv2.imwrite("tubo_frame.png", frame_r)
                print("Guardado")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = DetectorTubo()
    detector.probar()