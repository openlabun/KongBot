"""
detector_rocas.py — Detección de rocas en Banana Kong
Resolución BlueStacks: 960x540

Estrategia: Template matching con dos templates (roca1 y roca2).
"""

import cv2
import numpy as np
import pygetwindow as gw
from mss import mss
import time
import os

UMBRAL = 0.7
ROI = (140, 0, 960, 510)
ESCALAS = [0.9, 1.0, 1.1]
MIN_DISTANCIA = 50
FACTOR_RESIZE = 0.5  # trabajar a mitad de resolución
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

TEMPLATES_INFO = [
    ("roca1-bg.png", "roca"),
    ("roca2-bg.png", "roca_grande"),
]


class DetectorRocas:
    def __init__(self):
        self.titulo = "BlueStacks"
        self.ventana = None
        self.sct = mss()
        self.actualizar_ventana()

        self.templates = []
        for filename, tipo in TEMPLATES_INFO:
            ruta = os.path.join(TEMPLATES_DIR, filename)
            img = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"⚠️  No se encontró: {ruta}")
                continue
            if img.shape[2] == 4:
                alpha = img[:, :, 3]
                gris = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            else:
                gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                alpha = None
            # Pre-reducir templates al mismo factor que la ROI
            gris_small = cv2.resize(gris, (0,0), fx=FACTOR_RESIZE, fy=FACTOR_RESIZE)
            alpha_small = cv2.resize(alpha, (0,0), fx=FACTOR_RESIZE, fy=FACTOR_RESIZE) if alpha is not None else None
            self.templates.append((gris_small, alpha_small, tipo))

        print(f"✅ {len(self.templates)} templates de roca cargados")

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

    def detectar_rocas(self, frame):
        if frame is None:
            return [], frame

        x0, y0, x1, y1 = ROI
        roi = frame[y0:y1, x0:x1]
        roi_small = cv2.resize(roi, (0,0), fx=FACTOR_RESIZE, fy=FACTOR_RESIZE)
        roi_gris = cv2.cvtColor(roi_small, cv2.COLOR_BGR2GRAY)
        frame_resultado = frame.copy()

        detecciones = []
        factor_inv = 1.0 / FACTOR_RESIZE

        for template, alpha, tipo in self.templates:
            h_t, w_t = template.shape
            for escala in ESCALAS:
                nw, nh = int(w_t * escala), int(h_t * escala)
                if nw >= roi_gris.shape[1] or nh >= roi_gris.shape[0]:
                    continue
                t_scaled = cv2.resize(template, (nw, nh))
                if alpha is not None:
                    a_scaled = cv2.resize(alpha, (nw, nh))
                    res = cv2.matchTemplate(roi_gris, t_scaled, cv2.TM_CCOEFF_NORMED, mask=a_scaled)
                else:
                    res = cv2.matchTemplate(roi_gris, t_scaled, cv2.TM_CCOEFF_NORMED)
                ubicaciones = np.where(res >= UMBRAL)
                for pt in zip(*ubicaciones[::-1]):
                    val = res[pt[1], pt[0]]
                    if not np.isfinite(val):
                        continue
                    detecciones.append((val, pt[0], pt[1], nw, nh, tipo))

        # Non-maximum suppression
        detecciones.sort(key=lambda d: -d[0])
        rocas = []
        usados = []

        for val, bx, by, bw, bh, tipo in detecciones:
            cx_d, cy_d = bx + bw/2, by + bh/2
            if any(abs(cx_d-ux) < MIN_DISTANCIA and abs(cy_d-uy) < MIN_DISTANCIA for ux, uy in usados):
                continue
            usados.append((cx_d, cy_d))

            x_real = int(x0 + bx * factor_inv)
            y_real = int(y0 + by * factor_inv)
            bw_real = int(bw * factor_inv)
            bh_real = int(bh * factor_inv)
            cx = (x_real + bw_real/2) / frame.shape[1]
            cy = (y_real + bh_real/2) / frame.shape[0]
            rocas.append((cx, cy, tipo, (x_real, y_real, bw_real, bh_real)))
            #print(f"  Roca conf={val:.2f} | tipo={tipo} | area={bw_real*bh_real} | w={bw_real} h={bh_real} | ratio={round(bw_real/bh_real,2) if bh_real>0 else 0}")

            cv2.rectangle(frame_resultado, (x_real, y_real), (x_real+bw_real, y_real+bh_real), (0, 140, 255), 2)
            cv2.putText(frame_resultado, f"{tipo} {val:.2f}", (x_real, y_real-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1)

        cv2.putText(frame_resultado, f"Rocas: {len(rocas)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 140, 255) if rocas else (0, 200, 0), 2)

        return rocas, frame_resultado

    def probar(self):
        print("=== DETECTOR DE ROCAS ===")
        print(f"Umbral: {UMBRAL} — sube si hay falsos positivos, baja si no detecta")
        print("q=salir | s=guardar")
        time.sleep(2)

        cv2.namedWindow("Rocas Detector")

        while True:
            frame = self.capturar_pantalla()
            if frame is None:
                time.sleep(1)
                continue

            rocas, frame_resultado = self.detectar_rocas(frame)
            '''
            if rocas:
                print(f"🪨 {len(rocas)} roca(s): {rocas}")
            '''

            cv2.imshow("Rocas Detector", frame_resultado)
            cv2.moveWindow("Rocas Detector", 100, 100)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                cv2.imwrite("rocas_frame.png", frame_resultado)
                print("Guardado")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = DetectorRocas()
    detector.probar()