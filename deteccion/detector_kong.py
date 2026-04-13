"""
detector_kong.py — Detección de Kong: HSV + Template Matching + CSRT Tracker
Resolución BlueStacks: 960x540

Estrategia:
  1. HSV + Template Matching inicializa el tracker CSRT
  2. CSRT sigue a Kong frame a frame sin depender del color
  3. Si el tracker falla, se reinicializa con HSV + Template Matching
"""

import cv2
import numpy as np
import pygetwindow as gw
from mss import mss
import time
import os

# ── Configuración ────────────────────────────────────────────────────
ROI = (80, 0, 420, 510)

KONG_HSV_BAJO = np.array([5, 80, 50])
KONG_HSV_ALTO = np.array([25, 170, 180])

BLOB_AREA_MIN = 400
BLOB_AREA_MAX = 3500
BLOB_RATIO_MIN = 0.4
BLOB_RATIO_MAX = 2.5

UMBRAL = 0.65
ESCALAS = [0.9, 1.0, 1.1]
MARGEN_BLOB = 10

# Cuántos frames esperar antes de reintentar inicializar el tracker
FRAMES_REINTENTAR = 10

# Verificación periódica anti-drift
FRAMES_VERIFICAR  = 25   # cada 25 frames comprobar que el tracker no derivó
MAX_DIST_DRIFT    = 80   # px: distancia máxima tolerable entre tracker y template

# Tamaño fijo del bbox que se le entrega al CSRT — independiente de la pose
TRACKER_W = 55
TRACKER_H = 60

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

TEMPLATES_INFO = [
    ("kong_inicio-bg.png",     "inicio"),
    ("kong_corriendo1-bg.png", "corriendo"),
    ("kong_corriendo3-bg.png", "corriendo"),
    ("kong_saltando-bg.png",   "saltando"),
    ("kong_saltando2-bg.png",  "saltando"),
    ("kong_paracaidas-bg.png", "paracaidas"),
    ("kong_dash-bg.png",       "dash"),
    ("kong_liana-bg.png",      "liana"),
    ("kong_guacamaya-bg.png",  "guacamaya"),
]


class DetectorKong:
    def __init__(self):
        self.titulo = "BlueStacks"
        self.ventana = None
        self.sct = mss()
        self.actualizar_ventana()

        self.templates = []
        for filename, pose in TEMPLATES_INFO:
            ruta = os.path.join(TEMPLATES_DIR, filename)
            img = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"⚠️  No se encontró: {ruta}")
                continue
            if img.shape[2] == 4:
                alpha = img[:, :, 3]
                img_gris = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            else:
                img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                alpha = None
            self.templates.append((img_gris, alpha, pose))
        print(f"✅ {len(self.templates)} templates cargados")

        # Estado del tracker
        self.tracker = None
        self._tracker_activo = False
        self._frames_sin_deteccion = 0
        self._frames_desde_verificacion = 0   # anti-drift
        self._pose_anterior = "corriendo"
        self._rect_anterior = None
        self.posicion_anterior = None

    def actualizar_ventana(self):
        ventanas = gw.getWindowsWithTitle(self.titulo)
        if ventanas:
            self.ventana = ventanas[0]
            self.monitor = {
                "top": self.ventana.top,
                "left": self.ventana.left,
                "width": self.ventana.width,
                "height": self.ventana.height,
            }
            return True
        return False

    def capturar_pantalla(self):
        if not self.actualizar_ventana():
            return None
        screenshot = self.sct.grab(self.monitor)
        frame = np.array(screenshot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def _tiene_color_kong(self, frame, rect):
        """True si el bbox trackeado aún contiene suficiente color de Kong."""
        x, y, w, h = rect
        if w <= 0 or h <= 0:
            return False
        roi = frame[max(0,y):y+h, max(0,x):x+w]
        if roi.size == 0:
            return False
        hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask  = cv2.inRange(hsv, KONG_HSV_BAJO, KONG_HSV_ALTO)
        ratio = np.count_nonzero(mask) / mask.size
        return ratio > 0.08   # al menos 8% de píxeles con color de Kong

    def _encontrar_blobs_hsv(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mascara = cv2.inRange(hsv, KONG_HSV_BAJO, KONG_HSV_ALTO)
        mascara = cv2.erode(mascara, None, iterations=1)
        mascara = cv2.dilate(mascara, None, iterations=2)
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if not (BLOB_AREA_MIN < area < BLOB_AREA_MAX):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / h if h > 0 else 0
            if not (BLOB_RATIO_MIN < ratio < BLOB_RATIO_MAX):
                continue
            blobs.append((x, y, w, h))
        return blobs

    def _match_sobre_blob(self, roi_gris, x, y, w, h):
        h_roi, w_roi = roi_gris.shape
        bx0 = max(0, x - MARGEN_BLOB)
        by0 = max(0, y - MARGEN_BLOB)
        bx1 = min(w_roi, x + w + MARGEN_BLOB)
        by1 = min(h_roi, y + h + MARGEN_BLOB)
        recorte = roi_gris[by0:by1, bx0:bx1]
        mejor_val = 0
        mejor_pose = None
        for template, alpha, pose in self.templates:
            h_t, w_t = template.shape
            for escala in ESCALAS:
                nw = int(w_t * escala)
                nh = int(h_t * escala)
                if nw >= recorte.shape[1] or nh >= recorte.shape[0]:
                    continue
                t_scaled = cv2.resize(template, (nw, nh))
                if alpha is not None:
                    a_scaled = cv2.resize(alpha, (nw, nh))
                    res = cv2.matchTemplate(recorte, t_scaled, cv2.TM_SQDIFF_NORMED, mask=a_scaled)
                else:
                    res = cv2.matchTemplate(recorte, t_scaled, cv2.TM_SQDIFF_NORMED)
                min_val, _, _, _ = cv2.minMaxLoc(res)
                val = 1.0 - min_val
                if val > mejor_val:
                    mejor_val = val
                    mejor_pose = pose
        return mejor_val, mejor_pose

    def _detectar_con_hsv_template(self, frame):
        """Intenta detectar a Kong con HSV + Template. Retorna (rect_px, pose, conf) o None."""
        x0, y0, x1, y1 = ROI
        roi = frame[y0:y1, x0:x1]
        roi_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blobs = self._encontrar_blobs_hsv(roi)
        if not blobs:
            return None

        mejor_val_global = 0
        mejor_blob = None
        mejor_pose = None
        for blob in blobs:
            bx, by, bw, bh = blob
            val, pose = self._match_sobre_blob(roi_gris, bx, by, bw, bh)
            if val > mejor_val_global:
                mejor_val_global = val
                mejor_blob = blob
                mejor_pose = pose

        if mejor_val_global < UMBRAL or mejor_blob is None:
            return None

        bx, by, bw, bh = mejor_blob
        x_real = x0 + bx
        y_real = y0 + by
        return (x_real, y_real, bw, bh), mejor_pose, mejor_val_global

    def reset(self):
        """Llamar al reiniciar el juego — el CSRT empieza desde cero."""
        self.tracker               = None
        self._tracker_activo       = False
        self._frames_sin_deteccion = 0
        self._frames_desde_verificacion = 0
        self._rect_anterior        = None
        self.posicion_anterior     = None

    def _inicializar_tracker(self, frame, rect):
        """Inicializa el CSRT tracker con bbox de tamaño fijo centrado en el blob."""
        x, y, w, h = rect
        cx = x + w // 2
        cy = y + h // 2
        tx = max(0, cx - TRACKER_W // 2)
        ty = max(0, cy - TRACKER_H // 2)
        # Asegurar que no se sale del frame
        tx = min(tx, frame.shape[1] - TRACKER_W)
        ty = min(ty, frame.shape[0] - TRACKER_H)
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, (tx, ty, TRACKER_W, TRACKER_H))
        self._tracker_activo = True

    def detectar_kong(self, frame):
        if frame is None:
            return None, frame, None, None, 0.0

        frame_resultado = frame.copy()
        x0, y0, x1, y1 = ROI
        cv2.rectangle(frame_resultado, (x0, y0), (x1, y1), (255, 255, 0), 1)

        # ── Intentar con tracker CSRT ─────────────────────────────────
        if self._tracker_activo and self.tracker is not None:
            ok, bbox = self.tracker.update(frame)
            if ok:
                bx, by, bw, bh = [int(v) for v in bbox]
                # CSRT puede cambiar el tamaño del bbox con el tiempo.
                # Usamos solo el centro que devuelve y forzamos tamaño fijo.
                cx_t = bx + bw // 2
                cy_t = by + bh // 2
                x = max(0, cx_t - TRACKER_W // 2)
                y = max(0, cy_t - TRACKER_H // 2)
                w, h = TRACKER_W, TRACKER_H

                # Validar que el bbox esté dentro del ROI
                rx0, ry0, rx1, ry1 = ROI
                if x < rx0 or y < ry0 or x+w > rx1 or y+h > ry1:
                    ok = False
                # Validar tamaño razonable
                elif w < 20 or h < 20 or w > 250 or h > 250:
                    ok = False

            if ok:
                rect_px = (x, y, w, h)
                cx = (x + w / 2) / frame.shape[1]
                cy = (y + h / 2) / frame.shape[0]
                self.posicion_anterior = (cx, cy)
                self._rect_anterior = rect_px
                self._frames_sin_deteccion = 0

                # ── Verificación anti-drift ───────────────────────────
                self._frames_desde_verificacion += 1
                if self._frames_desde_verificacion >= FRAMES_VERIFICAR:
                    self._frames_desde_verificacion = 0
                    # Primero: ¿el tracker todavía ve el color de Kong?
                    tracker_tiene_kong = self._tiene_color_kong(frame, rect_px)
                    if not tracker_tiene_kong:
                        # El tracker perdió a Kong — intentar re-init con template
                        resultado = self._detectar_con_hsv_template(frame)
                        if resultado is not None and resultado[2] >= 0.80:
                            rx, ry, rw, rh = resultado[0]
                            dist = np.hypot((x+w/2)-(rx+rw/2), (y+h/2)-(ry+rh/2))
                            self._inicializar_tracker(frame, resultado[0])
                            rect_px = resultado[0]
                            x, y, w, h = rect_px
                            cx = (x + w/2) / frame.shape[1]
                            cy = (y + h/2) / frame.shape[0]
                            self._pose_anterior  = resultado[1]
                            self.posicion_anterior = (cx, cy)
                            self._rect_anterior  = rect_px
                            cv2.rectangle(frame_resultado, (x,y), (x+w,y+h), (0,255,255), 2)
                            cv2.putText(frame_resultado, f"REDET d={dist:.0f}px", (x, y-5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
                            return (cx,cy), frame_resultado, self._pose_anterior, rect_px, resultado[2]
                # ─────────────────────────────────────────────────────

                cv2.rectangle(frame_resultado, (x, y), (x+w, y+h), (0, 165, 255), 2)
                cv2.circle(frame_resultado, (x + w//2, y + h//2), 5, (0, 165, 255), -1)
                cv2.putText(frame_resultado, "CSRT", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
                return (cx, cy), frame_resultado, self._pose_anterior, rect_px, 1.0
            else:
                self._tracker_activo = False
                self._frames_sin_deteccion = 0

        # ── Reinicializar con HSV + Template ─────────────────────────
        self._frames_sin_deteccion += 1
        if self._frames_sin_deteccion <= FRAMES_REINTENTAR or not self._tracker_activo:
            resultado = self._detectar_con_hsv_template(frame)
            if resultado is not None:
                rect_px, pose, conf = resultado
                x, y, w, h = rect_px
                self._inicializar_tracker(frame, rect_px)
                self._pose_anterior = pose
                self._rect_anterior = rect_px
                cx = (x + w / 2) / frame.shape[1]
                cy = (y + h / 2) / frame.shape[0]
                self.posicion_anterior = (cx, cy)
                self._frames_sin_deteccion = 0

                cv2.rectangle(frame_resultado, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.circle(frame_resultado, (x + w//2, y + h//2), 5, (0, 255, 0), -1)
                cv2.putText(frame_resultado, f"INIT conf={conf:.2f}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                return (cx, cy), frame_resultado, pose, rect_px, conf

        # ── Fallback: última posición conocida (máx 10 frames) ───────
        if self.posicion_anterior is not None and self._frames_sin_deteccion <= 10:
            return self.posicion_anterior, frame_resultado, self._pose_anterior, self._rect_anterior, 0.0

        return None, frame_resultado, None, None, 0.0

    def probar(self):
        print("=== DETECTOR DE KONG (CSRT + HSV + Template) ===")
        print("Presiona 'q' para salir | 's' para guardar | 'r' para reiniciar tracker")
        time.sleep(2)

        cv2.namedWindow("Kong Detector")
        fps_tiempo = time.time()
        fps_contador = 0

        while True:
            frame = self.capturar_pantalla()
            if frame is None:
                print("Esperando BlueStacks...")
                time.sleep(1)
                continue

            posicion, frame_resultado, pose, rect_px, conf = self.detectar_kong(frame)

            fps_contador += 1
            if time.time() - fps_tiempo >= 1.0:
                fps = fps_contador / (time.time() - fps_tiempo)
                if posicion and rect_px:
                    kx, ky, kw, kh = rect_px
                    area = kw * kh
                    region_hsv = cv2.cvtColor(frame[ky:ky+kh, kx:kx+kw], cv2.COLOR_BGR2HSV)
                    h_med = float(np.mean(region_hsv[:,:,0]))
                    s_med = float(np.mean(region_hsv[:,:,1]))
                    v_med = float(np.mean(region_hsv[:,:,2]))
                    print(f"FPS: {fps:.1f} | Kong: {posicion} | Pose: {pose} | conf={conf:.2f} | area={area} H={h_med:.0f} S={s_med:.0f} V={v_med:.0f}")
                else:
                    print(f"FPS: {fps:.1f} | Kong: None")
                fps_contador = 0
                fps_tiempo = time.time()

            cv2.imshow("Kong Detector", frame_resultado)
            cv2.moveWindow("Kong Detector", 100, 100)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                cv2.imwrite("kong_frame.png", frame_resultado)
                print("Frame guardado")
            elif key == ord("r"):
                self.tracker = None
                self._tracker_activo = False
                print("Tracker reiniciado")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = DetectorKong()
    detector.probar()