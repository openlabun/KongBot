"""
perceptor.py — Perceptor con hilo de detección en background.

El agente NUNCA espera a los detectores — lee el último estado disponible.
Los detectores corren continuamente en un hilo separado.
"""

import cv2
import numpy as np
import pygetwindow as gw
from mss import mss
import time
import threading

from deteccion.detector_kong     import DetectorKong
from deteccion.detector_bananas  import DetectorBananas
from deteccion.detector_gameover import DetectorGameOver
from deteccion.detector_agua     import DetectorAgua
from deteccion.detector_barriles import DetectorBarriles
from deteccion.detector_rocas    import DetectorRocas
from deteccion.detector_muros    import DetectorMuros
from deteccion.detector_mina     import DetectorMina
from deteccion.detector_tubo     import DetectorTubo

GAMEOVER_CADA = 10


class Perceptor:
    def __init__(self):
        self.titulo  = "BlueStacks"
        self.ventana = None
        self.actualizar_ventana()

        print("Cargando detectores...")
        self.det_kong     = DetectorKong()
        self.det_bananas  = DetectorBananas()
        self.det_gameover = DetectorGameOver()
        self.det_agua     = DetectorAgua()
        self.det_barriles = DetectorBarriles()
        self.det_rocas    = DetectorRocas()
        self.det_muros    = DetectorMuros()
        self.det_mina     = DetectorMina()
        self.det_tubo     = DetectorTubo()
        print("✅ Detectores listos")

        # Cadencia de cada detector
        self.AGUA_CADA     = 3
        self.BARRILES_CADA = 2
        self.ROCAS_CADA    = 3
        self.MUROS_CADA    = 3
        self.MINA_CADA     = 5
        self.TUBO_CADA     = 5

        # Últimos resultados
        self._ultimo_gameover    = False
        self._ultimo_agua        = False
        self._agua_cx            = None
        self._ultimo_agua_rects  = []
        self._ultimo_barriles       = []
        self._ultimo_barriles_rects = []
        self._ultimo_rocas          = []
        self._ultimo_rocas_rects    = []
        self._ultimo_muros_rects    = []
        self._ultimo_mina           = False
        self._ultimo_mina_pos       = None
        self._ultimo_mina_rect      = None
        self._ultimo_tubo           = False
        self._ultimo_tubo_pos       = None
        self._ultimo_tubo_rect      = None

        # Estado compartido
        self._estado      = self._estado_vacio()
        self._lock        = threading.Lock()
        self._frame_count = 0

        # Colisiones bananas
        self._bananas_recogidas = 0
        self._pico_colisiones   = 0
        self.MARGEN_KONG        = 5

        # Arrancar hilos
        self._activo   = True
        self._hilo_rapido = threading.Thread(target=self._loop_rapido, daemon=True)
        self._hilo_lento  = threading.Thread(target=self._loop_lento,  daemon=True)
        self._hilo_rapido.start()
        self._hilo_lento.start()

        print("Esperando primer frame...", end=" ")
        for _ in range(50):
            time.sleep(0.1)
            with self._lock:
                if self._estado["frame"] is not None:
                    break
        print("listo")

        self._display_activo = False
        self._total_bananas  = 0
        self._step_count     = 0

    # ── Ventana ──────────────────────────────────────────────────────
    def actualizar_ventana(self):
        ventanas = gw.getWindowsWithTitle("BlueStacks")
        if ventanas:
            self.ventana = ventanas[0]
            return True
        return False

    def _get_monitor(self):
        return {
            "top":    self.ventana.top,
            "left":   self.ventana.left,
            "width":  self.ventana.width,
            "height": self.ventana.height,
        }

    # ── Hilo rápido: Kong + Bananas + colisiones + GameOver ──────────
    def _loop_rapido(self):
        sct = mss()
        while self._activo:
            try:
                if self.ventana is None:
                    self.actualizar_ventana()
                    time.sleep(0.5)
                    continue

                screenshot = sct.grab(self._get_monitor())
                arr = np.array(screenshot)
                if arr is None or arr.size == 0 or arr.shape[0] == 0:
                    time.sleep(0.05)
                    continue
                frame = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

                kong_pos, _, kong_pose, kong_rect, _ = self.det_kong.detectar_kong(frame)
                cantidad, _, contornos, rects = self.det_bananas.detectar_bananas(frame)

                h_f, w_f = frame.shape[:2]
                bananas_pos = []
                for cnt in contornos:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    bananas_pos.append(((x + bw/2) / w_f, (y + bh/2) / h_f))

                self._detectar_colisiones(kong_rect, rects)

                with self._lock:
                    self._estado["kong"]      = kong_pos
                    self._estado["kong_rect"] = kong_rect
                    self._estado["kong_pose"] = kong_pose
                    self._estado["bananas"]   = {"cantidad": cantidad, "posiciones": bananas_pos, "rects": rects}
                    self._estado["frame"]     = frame

            except Exception as e:
                print(f"[perceptor-rapido] error: {e}")
                time.sleep(0.1)

    # ── Hilo lento: GameOver + Agua + Barriles + Rocas + Muros ───────
    def _loop_lento(self):
        sct = mss()
        self._frame_count = 0
        while self._activo:
            try:
                if self.ventana is None:
                    time.sleep(0.5)
                    continue

                screenshot = sct.grab(self._get_monitor())
                arr = np.array(screenshot)
                if arr is None or arr.size == 0 or arr.shape[0] == 0:
                    time.sleep(0.1)
                    continue
                frame = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

                self._frame_count += 1

                if self._frame_count % GAMEOVER_CADA == 0:
                    game_over, _, _ = self.det_gameover.detectar_gameover(frame)
                    self._ultimo_gameover = game_over

                if self._frame_count % self.AGUA_CADA == 0:
                    hay_agua, agua_cx, _, _, agua_rects = self.det_agua.detectar_agua(frame)
                    self._ultimo_agua       = hay_agua
                    self._agua_cx           = agua_cx
                    self._ultimo_agua_rects = agua_rects

                if self._frame_count % self.BARRILES_CADA == 0:
                    barriles, _, _ = self.det_barriles.detectar_barriles(frame)
                    self._ultimo_barriles       = [(cx, cy) for cx, cy, _ in barriles]
                    self._ultimo_barriles_rects = [r for _, _, r in barriles]

                if self._frame_count % self.ROCAS_CADA == 0:
                    rocas, _ = self.det_rocas.detectar_rocas(frame)
                    self._ultimo_rocas       = [(cx, cy) for cx, cy, _, _ in rocas]
                    self._ultimo_rocas_rects = [rect for _, _, _, rect in rocas]

                if self._frame_count % self.MUROS_CADA == 0:
                    muros, _, _ = self.det_muros.detectar_muros(frame)
                    self._ultimo_muros_rects = [m["rect"] for m in muros]

                if self._frame_count % self.MINA_CADA == 0:
                    hay_mina, mina_pos, mina_rect, _ = self.det_mina.detectar_mina(frame)
                    self._ultimo_mina      = hay_mina
                    self._ultimo_mina_pos  = mina_pos
                    self._ultimo_mina_rect = mina_rect

                if self._frame_count % self.TUBO_CADA == 0:
                    hay_tubo, tubo_pos, tubo_rect, _ = self.det_tubo.detectar_tubo(frame)
                    self._ultimo_tubo      = hay_tubo
                    self._ultimo_tubo_pos  = tubo_pos
                    self._ultimo_tubo_rect = tubo_rect

                with self._lock:
                    self._estado["barriles"]      = self._ultimo_barriles
                    self._estado["barriles_rects"] = self._ultimo_barriles_rects
                    self._estado["rocas"]          = self._ultimo_rocas
                    self._estado["rocas_rects"]    = self._ultimo_rocas_rects
                    self._estado["muros_rects"]    = self._ultimo_muros_rects
                    self._estado["agua"]        = self._ultimo_agua
                    self._estado["agua_cx"]     = self._agua_cx
                    self._estado["agua_rects"]  = self._ultimo_agua_rects
                    self._estado["game_over"]   = self._ultimo_gameover
                    self._estado["mina"]        = self._ultimo_mina
                    self._estado["mina_pos"]    = self._ultimo_mina_pos
                    self._estado["mina_rect"]   = self._ultimo_mina_rect
                    self._estado["tubo"]        = self._ultimo_tubo
                    self._estado["tubo_pos"]    = self._ultimo_tubo_pos
                    self._estado["tubo_rect"]   = self._ultimo_tubo_rect

            except Exception as e:
                print(f"[perceptor-lento] error: {e}")
                time.sleep(0.1)

    # ── Colisiones bananas ────────────────────────────────────────────
    def _detectar_colisiones(self, kong_rect, rects_bananas):
        if kong_rect is None:
            return
        kx, ky, kw, kh = kong_rect
        kx -= self.MARGEN_KONG;  ky -= self.MARGEN_KONG
        kw += self.MARGEN_KONG * 2; kh += self.MARGEN_KONG * 2

        colisiones_ahora = 0
        for (bx, by, bw, bh) in rects_bananas:
            if kx < bx+bw and kx+kw > bx and ky < by+bh and ky+kh > by:
                colisiones_ahora += 1

        if colisiones_ahora > self._pico_colisiones:
            self._pico_colisiones = colisiones_ahora
        if colisiones_ahora == 0 and self._pico_colisiones > 0:
            self._bananas_recogidas += self._pico_colisiones
            self._pico_colisiones    = 0

    def pop_bananas_recogidas(self):
        with self._lock:
            n = self._bananas_recogidas
            self._bananas_recogidas = 0
            return n

    def reset_colisiones(self):
        self._bananas_recogidas  = 0
        self._pico_colisiones    = 0
        self._ultimo_gameover    = False

    # ── API pública ───────────────────────────────────────────────────
    def get_estado(self):
        with self._lock:
            return dict(self._estado)

    def get_conteo_bananas(self):
        with self._lock:
            return self._estado["bananas"]["cantidad"]

    # ── Display ───────────────────────────────────────────────────────
    def start_display(self):
        self._display_activo = True
        self._hilo_display = threading.Thread(target=self._loop_display, daemon=True)
        self._hilo_display.start()

    def _loop_display(self):
        cv2.namedWindow("Debug")
        cv2.moveWindow("Debug", 1000, 100)
        ESCALA = 0.5

        while self._display_activo:
            estado = self.get_estado()
            frame  = estado["frame"]
            if frame is None:
                time.sleep(0.05)
                continue

            small = cv2.resize(frame, (0, 0), fx=ESCALA, fy=ESCALA)

            # Kong — naranja
            if estado["kong_rect"]:
                kx, ky, kw, kh = [int(v * ESCALA) for v in estado["kong_rect"]]
                cv2.rectangle(small, (kx, ky), (kx+kw, ky+kh), (0, 165, 255), 1)

            # Bananas — verde
            for bx, by, bw, bh in estado["bananas"]["rects"]:
                bx, by, bw, bh = [int(v * ESCALA) for v in (bx, by, bw, bh)]
                cv2.rectangle(small, (bx, by), (bx+bw, by+bh), (0, 255, 0), 1)

            # Barriles — verde
            for rx, ry, rw, rh in estado.get("barriles_rects", []):
                rx2 = int(rx * ESCALA)
                ry2 = int(ry * ESCALA)
                rw2 = int(rw * ESCALA)
                rh2 = int(rh * ESCALA)
                cv2.rectangle(small, (rx2, ry2), (rx2+rw2, ry2+rh2), (0, 255, 0), 1)

            # Rocas — verde
            for rx, ry, rw, rh in estado.get("rocas_rects", []):
                rx2 = int(rx * ESCALA)
                ry2 = int(ry * ESCALA)
                rw2 = int(rw * ESCALA)
                rh2 = int(rh * ESCALA)
                cv2.rectangle(small, (rx2, ry2), (rx2+rw2, ry2+rh2), (0, 255, 0), 1)

            # Muros — bounding boxes
            for mx, my, mw, mh in estado.get("muros_rects", []):
                mx2 = int(mx * ESCALA)
                my2 = int(my * ESCALA)
                mw2 = int(mw * ESCALA)
                mh2 = int(mh * ESCALA)
                cv2.rectangle(small, (mx2, my2), (mx2+mw2, my2+mh2), (0, 255, 0), 1)

            # Agua — verde
            for ax, ay, aw, ah in estado.get("agua_rects", []):
                ax2 = int(ax * ESCALA)
                ay2 = int(ay * ESCALA)
                aw2 = int(aw * ESCALA)
                ah2 = int(ah * ESCALA)
                cv2.rectangle(small, (ax2, ay2), (ax2+aw2, ay2+ah2), (0, 255, 0), 1)

            # Mina — naranja
            if estado.get("mina") and estado.get("mina_rect"):
                mx, my, mw, mh = [int(v * ESCALA) for v in estado["mina_rect"]]
                cv2.rectangle(small, (mx, my), (mx+mw, my+mh), (0, 255, 0), 1)
                #cv2.putText(small, "MINA", (mx, my - 4),
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

            # Tubo — amarillo
            if estado.get("tubo") and estado.get("tubo_rect"):
                tx, ty, tw, th = [int(v * ESCALA) for v in estado["tubo_rect"]]
                cv2.rectangle(small, (tx, ty), (tx+tw, ty+th), (0, 255, 0), 1)
                #cv2.putText(small, "TUBO", (tx, ty - 4),
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # Panel infos
            ##'''
            #cv2.rectangle(small, (3, 3), (200, 50), (0, 0, 0), -1)
            cv2.putText(small, f"Bananas: {self._total_bananas}  Step: {self._step_count}",
                        (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            ##'''
            if estado["game_over"]:
                cv2.putText(small, "GAME OVER", (6, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow("Debug", small)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyWindow("Debug")

    def parar(self):
        self._activo         = False
        self._display_activo = False

    # ── Estado vacío ──────────────────────────────────────────────────
    def _estado_vacio(self):
        return {
            "kong":        None,
            "kong_rect":   None,
            "kong_pose":   None,
            "bananas":     {"cantidad": 0, "posiciones": [], "rects": []},
            "barriles":      [],
            "barriles_rects": [],
            "rocas":         [],
            "rocas_rects":   [],
            "muros_rects":   [],
            "agua":        False,
            "agua_cx":     None,
            "agua_rects":  [],
            "game_over":   False,
            "mina":        False,
            "mina_pos":    None,
            "mina_rect":   None,
            "tubo":        False,
            "tubo_pos":    None,
            "tubo_rect":   None,
            "frame":       None,
        }

    # ── Debug ─────────────────────────────────────────────────────────
    def probar(self):
        print("=== PERCEPTOR ===  q=salir")
        time.sleep(2)
        self.start_display()

        fps_t = time.time()
        fps_c = 0

        while True:
            estado = self.get_estado()
            fps_c += 1
            if time.time() - fps_t >= 1.0:
                fps = fps_c / (time.time() - fps_t)
                print(f"FPS: {fps:.1f} | Kong: {estado['kong']} | "
                      f"Bananas: {estado['bananas']['cantidad']} | GO: {estado['game_over']}")
                fps_c = 0
                fps_t = time.time()
            time.sleep(0.05)


if __name__ == "__main__":
    p = Perceptor()
    p.probar()