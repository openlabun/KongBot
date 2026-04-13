"""
detector_agua.py — Detección de agua en Banana Kong
Resolución BlueStacks: 960x540

Estrategia: HSV puro — el azul/celeste del agua es único en la escena.
Detecta si hay agua visible y su posición horizontal para que el agente
sepa si debe saltar o esquivar.
"""

import cv2
import numpy as np
import pygetwindow as gw
from mss import mss
import time


# ── Configuración ────────────────────────────────────────────────────

# ROI — el agua aparece en la parte inferior del frame
ROI = (0, 300, 960, 510)

# HSV del agua — azul/celeste calibrado desde pixels reales
AGUA_HSV_BAJO = np.array([85, 120, 150])
AGUA_HSV_ALTO = np.array([97, 210, 230])

# Área mínima para considerar que hay agua (evita ruido)
AGUA_AREA_MIN = 2000


class DetectorAgua:
    def __init__(self):
        self.titulo = "BlueStacks"
        self.ventana = None
        self.sct = mss()
        self.actualizar_ventana()

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

    # ─────────────────────────────────────────
    def detectar_agua(self, frame):
        """
        Detecta si hay agua visible y su posición.

        Retorna:
            hay_agua      : bool
            posicion_x    : float normalizado [0,1] del centro del agua, o None
            frame_resultado : frame con anotaciones
            mascara       : máscara HSV para depuración
        """
        if frame is None:
            return False, None, frame, None

        x0, y0, x1, y1 = ROI
        roi = frame[y0:y1, x0:x1]

        frame_resultado = frame.copy()
        cv2.rectangle(frame_resultado, (x0, y0), (x1, y1), (255, 255, 0), 1)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mascara = cv2.inRange(hsv, AGUA_HSV_BAJO, AGUA_HSV_ALTO)
        mascara = cv2.erode(mascara, None, iterations=1)
        mascara = cv2.dilate(mascara, None, iterations=2)

        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        zonas_agua = []
        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if area < AGUA_AREA_MIN:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            zonas_agua.append((x, y, w, h, area))

        if not zonas_agua:
            cv2.putText(frame_resultado, "Agua: NO", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
            return False, None, frame_resultado, mascara, []

        # Calcular posición horizontal del centro del agua
        total_area = sum(z[4] for z in zonas_agua)
        cx_pond = sum((z[0] + z[2]/2) * z[4] for z in zonas_agua) / total_area
        cx_norm = (cx_pond + x0) / frame.shape[1]

        # Dibujar zonas de agua
        for (ax, ay, aw, ah, area) in zonas_agua:
            x_real, y_real = ax + x0, ay + y0
            cv2.rectangle(frame_resultado,
                          (x_real, y_real), (x_real+aw, y_real+ah),
                          (0, 255, 0), 2)

        cv2.putText(frame_resultado,
                    f"AGUA detectada  cx={cx_norm:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

        rects = [(ax + x0, ay + y0, aw, ah) for (ax, ay, aw, ah, _) in zonas_agua]
        return True, cx_norm, frame_resultado, mascara, rects

    # ─────────────────────────────────────────
    def probar(self):
        print("=== DETECTOR DE AGUA ===")
        print("q=salir | s=guardar | m=máscara")
        time.sleep(2)

        mostrar_mascara = True
        cv2.namedWindow("Agua Detector")

        while True:
            frame = self.capturar_pantalla()
            if frame is None:
                time.sleep(1)
                continue

            hay_agua, cx, frame_resultado, mascara, _ = self.detectar_agua(frame)

            if hay_agua:
                print(f"💧 Agua detectada en cx={cx:.2f}")

            cv2.imshow("Agua Detector", frame_resultado)
            cv2.moveWindow("Agua Detector", 100, 100)

            if mostrar_mascara and mascara is not None:
                cv2.imshow("Mascara Agua", mascara)
                cv2.moveWindow("Mascara Agua", 800, 100)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                cv2.imwrite("agua_frame.png", frame_resultado)
                if mascara is not None:
                    cv2.imwrite("agua_mascara.png", mascara)
                print("Guardado")
            elif key == ord("m"):
                mostrar_mascara = not mostrar_mascara
                if not mostrar_mascara:
                    cv2.destroyWindow("Mascara Agua")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = DetectorAgua()
    detector.probar()