"""
detector_muros.py — Detección de muros en Banana Kong
Resolución BlueStacks: 960x540

Tipos de muro:
  - Madera : bloques apilados naranja/dorado intenso, ratio h/w ~1.2-4
  - Piedra  : roca gris/rosada desaturada,             ratio h/w ~1.8-7

Estrategia híbrida:
  1. HSV + filtros de forma (área, h/w, solidez) → candidatos
  2. Template matching TM_CCOEFF_NORMED con alpha real → verificación

Templates requeridos en la carpeta templates/:
  muro_madera.png   — PNG con canal alpha real (fondo transparente)
  muro_piedra.png   — PNG con canal alpha real (fondo transparente)
"""

import cv2
import numpy as np
import pygetwindow as gw
from mss import mss
import time
import os


# ── ROI ───────────────────────────────────────────────────────────────
# Excluye HUD (top 60px) y zona de Kong (x < 250) para evitar confundir
# el pelaje marrón de Kong con muros de madera.
ROI = (180, 0, 960, 510)

# ── HSV madera ────────────────────────────────────────────────────────
# Naranja/dorado intenso. S_min=150 excluye troncos de árbol (S~100-120)
# y otros elementos marrones del fondo que tienen saturación más baja.
MADERA_HSV_BAJO = np.array([5,  150, 60])
MADERA_HSV_ALTO = np.array([22, 255, 255])

# ── HSV piedra ────────────────────────────────────────────────────────
# Gris rosado moderadamente saturado. Calibrado del template real:
# H p10=7, p90=35 | S p10=45, p90=146 | V p10=60, p90=247
PIEDRA_HSV_BAJO = np.array([0,  40, 110])
PIEDRA_HSV_ALTO = np.array([25, 130, 255])

# ── Filtros de forma ──────────────────────────────────────────────────
AREA_MIN            = 800
AREA_MAX            = 35000
MADERA_RATIO_HW_MIN = 1.2    # madera: más compacta
MADERA_RATIO_HW_MAX = 4.0
PIEDRA_RATIO_HW_MIN = 1.8    # piedra: más alta y delgada
PIEDRA_RATIO_HW_MAX = 7.0
SOLIDEZ_MIN         = 0.45

# ── Template matching ─────────────────────────────────────────────────
UMBRAL = 0.62
ESCALAS       = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
MARGEN_BLOB   = 15

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")


class DetectorMuros:
    def __init__(self):
        self.titulo = "BlueStacks"
        self.ventana = None
        self.sct = mss()
        self.actualizar_ventana()
        self._cargar_templates()

    # ── Templates ────────────────────────────────────────────────────
    def _cargar_templates(self):
        self.tpl_madera = self._leer_template("muro_madera-bg.png")
        self.tpl_piedra = self._leer_template("muro_piedra-bg.png")
        print(f"{'✅' if self.tpl_madera else '⚠️ '} Template muro_madera.png")
        print(f"{'✅' if self.tpl_piedra else '⚠️ '} Template muro_piedra.png")

    def _leer_template(self, filename):
        """Lee PNG con alpha real. Retorna (gris, alpha) o None."""
        ruta = os.path.join(TEMPLATES_DIR, filename)
        img  = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.shape[2] != 4 or img[:, :, 3].max() == 0:
            raise ValueError(
                f"{filename} no tiene canal alpha válido. "
                "Usa los PNGs con alpha entregados junto a este script."
            )
        alpha = img[:, :, 3]
        gris  = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        return (gris, alpha)

    # ── Ventana BlueStacks ───────────────────────────────────────────
    def actualizar_ventana(self):
        ventanas = gw.getWindowsWithTitle(self.titulo)
        if ventanas:
            self.ventana = ventanas[0]
            self.monitor = {
                "top":    self.ventana.top,
                "left":   self.ventana.left,
                "width":  self.ventana.width,
                "height": self.ventana.height,
            }
            return True
        return False

    def capturar_pantalla(self):
        if self.ventana is None:
            if not self.actualizar_ventana():
                return None
        try:
            self.monitor = {
                "top":    self.ventana.top,
                "left":   self.ventana.left,
                "width":  self.ventana.width,
                "height": self.ventana.height,
            }
        except Exception:
            self.ventana = None
            return None
        screenshot = self.sct.grab(self.monitor)
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)

    # ── Paso 1: candidatos HSV + forma ───────────────────────────────
    def _blobs_hsv(self, roi, hsv_bajo, hsv_alto, ratio_min, ratio_max):
        hsv     = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mascara = cv2.inRange(hsv, hsv_bajo, hsv_alto)
        mascara = cv2.erode(mascara,  None, iterations=1)
        mascara = cv2.dilate(mascara, None, iterations=2)

        cnts, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if not (AREA_MIN < area < AREA_MAX):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            ratio_hw = h / w if w > 0 else 0
            if not (ratio_min < ratio_hw < ratio_max):
                continue
            hull    = cv2.convexHull(cnt)
            a_hull  = cv2.contourArea(hull)
            solidez = area / a_hull if a_hull > 0 else 0
            if solidez < SOLIDEZ_MIN:
                continue
            blobs.append((x, y, w, h))
        return blobs, mascara

    # ── Paso 2: verificación con template matching ───────────────────
    def _match_blob(self, roi_gris, x, y, w, h, tpl):
        if tpl is None:
            return 0.0
        template_gris, alpha = tpl
        hr, wr = roi_gris.shape
        recorte = roi_gris[
            max(0, y - MARGEN_BLOB) : min(hr, y + h + MARGEN_BLOB),
            max(0, x - MARGEN_BLOB) : min(wr, x + w + MARGEN_BLOB),
        ]
        if recorte.size == 0:
            return 0.0

        mejor = 0.0
        ht, wt = template_gris.shape
        for esc in ESCALAS:
            nw, nh = int(wt * esc), int(ht * esc)
            if nw < 4 or nh < 4 or nw >= recorte.shape[1] or nh >= recorte.shape[0]:
                continue
            t_s = cv2.resize(template_gris, (nw, nh))
            a_s = cv2.resize(alpha,         (nw, nh))
            res = cv2.matchTemplate(recorte, t_s, cv2.TM_CCOEFF_NORMED, mask=a_s)
            _, val, _, _ = cv2.minMaxLoc(res)
            if np.isfinite(val) and val > mejor:
                mejor = val
        return mejor

    # ── Detección principal ──────────────────────────────────────────
    def detectar_muros(self, frame):
        """
        Retorna:
          muros   : lista de dicts {cx, cy, altura, ancho, confianza, rect}
          frame_r : frame anotado
          mascaras: {"madera": mask, "piedra": mask}
        """
        if frame is None:
            return [], frame, {}

        x0, y0, x1, y1 = ROI
        roi      = frame[y0:y1, x0:x1]
        roi_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        frame_r = frame.copy()
        cv2.rectangle(frame_r, (x0, y0), (x1, y1), (255, 255, 0), 1)

        blobs_m, mask_m = self._blobs_hsv(roi, MADERA_HSV_BAJO, MADERA_HSV_ALTO,
                                          MADERA_RATIO_HW_MIN, MADERA_RATIO_HW_MAX)
        blobs_p, mask_p = self._blobs_hsv(roi, PIEDRA_HSV_BAJO, PIEDRA_HSV_ALTO,
                                          PIEDRA_RATIO_HW_MIN, PIEDRA_RATIO_HW_MAX)

        # Candidatos en gris claro
        for bx, by, bw, bh in blobs_m:
            cv2.rectangle(frame_r, (x0+bx, y0+by), (x0+bx+bw, y0+by+bh), (180,120,60), 1)
        for bx, by, bw, bh in blobs_p:
            cv2.rectangle(frame_r, (x0+bx, y0+by), (x0+bx+bw, y0+by+bh), (160,160,160), 1)

        # Verificar con template
        candidatos = (
            [(b, self.tpl_madera) for b in blobs_m] +
            [(b, self.tpl_piedra) for b in blobs_p]
        )

        muros = []

        for (bx, by, bw, bh), tpl in candidatos:
            conf = self._match_blob(roi_gris, bx, by, bw, bh, tpl)
            if conf < UMBRAL:
                continue
            xr, yr = x0 + bx, y0 + by
            cx = (xr + bw / 2) / frame.shape[1]
            cy = (yr + bh / 2) / frame.shape[0]
            muros.append({
                "cx":        cx,
                "cy":        cy,
                "altura":    bh,
                "ancho":     bw,
                "confianza": conf,
                "rect":      (xr, yr, bw, bh),
            })
            cv2.rectangle(frame_r, (xr, yr), (xr+bw, yr+bh), (0, 140, 255), 2)
            cv2.putText(frame_r, f"muro {conf:.2f}",
                        (xr, yr - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1)

        cv2.putText(frame_r, f"Muros: {len(muros)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255) if muros else (0, 200, 0), 2)

        return muros, frame_r, {"madera": mask_m, "piedra": mask_p}

    # ── Modo prueba ──────────────────────────────────────────────────
    def probar(self):
        print("=== DETECTOR DE MUROS ===")
        print("q=salir | s=guardar | m=masks HSV")
        print("Naranja=madera  Gris=piedra")
        time.sleep(2)

        mostrar_masks = False
        cv2.namedWindow("Muros")

        while True:
            frame = self.capturar_pantalla()
            if frame is None:
                time.sleep(1)
                continue

            muros, frame_r, masks = self.detectar_muros(frame)
            '''
            if muros:
                for m in muros:
                    print(f"🧱 cx={m['cx']:.2f}  h={m['altura']}px  conf={m['confianza']:.2f}")
            '''
            cv2.imshow("Muros", frame_r)
            cv2.moveWindow("Muros", 100, 100)

            if mostrar_masks:
                cv2.imshow("Mask Madera", masks["madera"])
                cv2.moveWindow("Mask Madera", 800, 100)
                cv2.imshow("Mask Piedra",  masks["piedra"])
                cv2.moveWindow("Mask Piedra",  800, 350)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                ts = int(time.time())
                cv2.imwrite(f"muros_{ts}.png", frame_r)
                print(f"Guardado muros_{ts}.png")
            elif key == ord("m"):
                mostrar_masks = not mostrar_masks
                if not mostrar_masks:
                    cv2.destroyWindow("Mask Madera")
                    cv2.destroyWindow("Mask Piedra")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = DetectorMuros()
    detector.probar()