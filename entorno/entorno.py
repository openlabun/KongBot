"""
entorno.py — Entorno Gymnasium simplificado para Banana Kong RL
Resolución BlueStacks: 960x540

Observación (13 valores normalizados [0,1]):
    [0] kong_cx
    [1] kong_cy
    [2] banana1_dx  (distancia horizontal relativa a Kong, centrado en 0.5)
    [3] banana1_cy
    [4] banana2_dx
    [5] banana2_cy
    [6] hay_agua (0 o 1)
    [7] barril1_dx  (distancia horizontal relativa a Kong)
    [8] barril1_cy
    [9] roca1_dx    (distancia horizontal relativa a Kong)
    [10] roca1_cy
    [11] muro1_dx   (distancia horizontal relativa a Kong)
    [12] muro1_cy
    [13] mina_dx    (distancia horizontal relativa a Kong)
    [14] tubo_dx    (distancia horizontal relativa a Kong)

Acciones (Discrete 4):
    0 - NADA
    1 - PLANEAR (tap=saltar, mantener=planear)
    2 - DASH
    3 - BAJAR

Recompensas:
    +1.0  por banana recogida
    +0.01 por sobrevivir cada step
    -20.0 por game over
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import cv2
import pyautogui

from entorno.perceptor  import Perceptor
from controles.acciones import ModuloAcciones, NADA, PLANEAR, DASH, BAJAR

MAX_STEPS      = 2000
DELAY_ACCION   = 0.05
DELAY_REINICIO = 3.0
OBS_SIZE       = 15
MARGEN_KONG    = 2


class BananaKongEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        self.perceptor = Perceptor()
        self.acciones  = ModuloAcciones()
        self.perceptor.start_display()  # ventana de visualización independiente

        self._step_count      = 0
        self._total_bananas   = 0
        self._reward_episodio = 0.0
        self.perceptor.reset_colisiones()
        self._primer_episodio = True
        self._ultimo_frame    = None
        self._ultimo_estado   = None

    # ── Reset ─────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        pyautogui.keyUp('w')

        if self._primer_episodio:
            self._primer_episodio = False
        else:
            self._reiniciar_juego()

        self._step_count      = 0
        self._total_bananas   = 0
        self._reward_episodio = 0.0
        self.perceptor.reset_colisiones()

        estado = self.perceptor.get_estado()
        obs    = self._estado_a_obs(estado)
        return obs, {}

    # ── Step ──────────────────────────────────────────────────────────
    def step(self, accion):
        self.acciones.ejecutar(accion)
        time.sleep(DELAY_ACCION)

        estado = self.perceptor.get_estado()
        obs    = self._estado_a_obs(estado)

        # Bananas recogidas detectadas en el hilo del perceptor
        # que corre continuamente — no pierde picos entre steps
        bananas_recogidas = self.perceptor.pop_bananas_recogidas()

        self._ultimo_frame  = estado.get("frame")
        self._ultimo_estado = estado


        '''
        # Penalizar DASH cerca de mina o tubo
        if accion == DASH and (estado.get("mina") or estado.get("tubo")):
            print("⚠️  Penalizando DASH cerca de mina/tubo")
            reward_dash_penalty = -5.0
        else:
            reward_dash_penalty = 0.0
        '''

        # Reward
        reward  = 0.02 # + reward_dash_penalty
        reward += bananas_recogidas * 1.0

        terminated = False
        if estado["game_over"] and self._step_count > 10:
            reward    -= 10.0
            terminated = True

        self._reward_episodio += reward
        self._total_bananas   += bananas_recogidas
        self._step_count      += 1
        # Actualizar contadores del display
        self.perceptor._total_bananas = self._total_bananas
        self.perceptor._step_count    = self._step_count
        truncated = self._step_count >= MAX_STEPS

        if terminated or truncated:
            print(f"  [ep end] steps={self._step_count} bananas={self._total_bananas} reward_total={self._reward_episodio:+.2f}")

        return obs, reward, terminated, truncated, {
            "kong":    estado["kong"],
            "bananas": bananas_recogidas,
        }

    # ── Observación ───────────────────────────────────────────────────
    def _estado_a_obs(self, estado):
        obs = np.zeros(OBS_SIZE, dtype=np.float32)

        # Kong posición
        if estado["kong"]:
            kong_cx, kong_cy = estado["kong"]
        else:
            kong_cx, kong_cy = 0.3, 0.5
        obs[0] = kong_cx
        obs[1] = kong_cy

        # Bananas — distancia horizontal relativa a Kong (positivo = a la derecha)
        bananas = sorted(
            estado["bananas"]["posiciones"],
            key=lambda p: abs(p[0] - kong_cx)
        )
        for i, (cx, cy) in enumerate(bananas[:2]):
            obs[2 + i*2] = np.clip(cx - kong_cx + 0.5, 0.0, 1.0)  # centrado en 0.5
            obs[3 + i*2] = cy

        obs[6] = 1.0 if estado.get("agua") else 0.0

        # Barril más cercano — distancia horizontal relativa
        barriles = sorted(
            estado.get("barriles", []),
            key=lambda p: abs(p[0] - kong_cx)
        )
        if barriles:
            obs[7] = np.clip(barriles[0][0] - kong_cx + 0.5, 0.0, 1.0)
            obs[8] = barriles[0][1]

        # Roca más cercana — distancia horizontal relativa
        rocas = sorted(
            estado.get("rocas", []),
            key=lambda p: abs(p[0] - kong_cx)
        )
        if rocas:
            obs[9]  = np.clip(rocas[0][0] - kong_cx + 0.5, 0.0, 1.0)
            obs[10] = rocas[0][1]

        # Muro más cercano — distancia horizontal relativa
        muros_pos = []
        for mx, my, mw, mh in estado.get("muros_rects", []):
            cx_m = (mx + mw/2) / 960
            cy_m = (my + mh/2) / 540
            muros_pos.append((cx_m, cy_m))
        muros_pos = sorted(muros_pos, key=lambda p: abs(p[0] - kong_cx))
        if muros_pos:
            obs[11] = np.clip(muros_pos[0][0] - kong_cx + 0.5, 0.0, 1.0)
            obs[12] = muros_pos[0][1]

        # Mina — distancia horizontal relativa
        if estado.get("mina_pos"):
            obs[13] = np.clip(estado["mina_pos"][0] - kong_cx + 0.5, 0.0, 1.0)
        else:
            obs[13] = 0.5  # sin mina = neutro

        # Tubo — distancia horizontal relativa
        if estado.get("tubo_pos"):
            obs[14] = np.clip(estado["tubo_pos"][0] - kong_cx + 0.5, 0.0, 1.0)
        else:
            obs[14] = 0.5  # sin tubo = neutro

        return obs

    # ── Render ────────────────────────────────────────────────────────
    def render(self):
        pass  # display corre en hilo propio del perceptor

    # ── Reinicio ──────────────────────────────────────────────────────
    def _reiniciar_juego(self):
        import os
        tpl_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               "deteccion", "templates")
        tpl_flecha     = self._cargar_template(os.path.join(tpl_dir, "flecha.png"))
        tpl_play_again = self._cargar_template(os.path.join(tpl_dir, "play_again.png"))

        print("Esperando pantalla Revive...")
        time.sleep(5.5)  # +1s extra

        if not self._esperar_y_clicar(tpl_flecha, timeout=9.0, etiqueta="flecha"):
            print("Flecha no encontrada — presionando W como fallback...")
            pyautogui.press('w')
        time.sleep(2.0)  # +1s extra

        if not self._esperar_y_clicar(tpl_play_again, timeout=9.0, etiqueta="Play Again"):
            print("Play Again no encontrado — presionando W como fallback...")
            pyautogui.press('w')
        time.sleep(1.0)  # +1s extra

        # Resetear tracker DESPUÉS de Play Again — el juego ya está cargando
        self.perceptor.det_kong.reset()
        time.sleep(DELAY_REINICIO)
        print("Juego reiniciado")

    def _cargar_template(self, ruta):
        import cv2
        img = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.shape[2] == 4:
            return cv2.cvtColor(img[:,:,:3], cv2.COLOR_BGR2GRAY), img[:,:,3]
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None

    def _esperar_y_clicar(self, template, timeout=8.0, etiqueta=""):
        import cv2
        if template is None:
            return False
        t_gris, t_alpha = template
        inicio = time.time()
        while time.time() - inicio < timeout:
            estado = self.perceptor.get_estado()
            frame  = estado["frame"]
            if frame is None:
                time.sleep(0.2)
                continue
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for escala in [0.9, 1.0, 1.1]:
                h_t, w_t = t_gris.shape
                nw, nh = int(w_t*escala), int(h_t*escala)
                if nw >= gris.shape[1] or nh >= gris.shape[0]:
                    continue
                t_s = cv2.resize(t_gris, (nw, nh))
                if t_alpha is not None:
                    a_s = cv2.resize(t_alpha, (nw, nh))
                    res = cv2.matchTemplate(gris, t_s, cv2.TM_CCOEFF_NORMED, mask=a_s)
                else:
                    res = cv2.matchTemplate(gris, t_s, cv2.TM_CCOEFF_NORMED)
                _, val, _, loc = cv2.minMaxLoc(res)
                if val >= 0.65:
                    cx = loc[0] + nw//2 + self.perceptor.ventana.left
                    cy = loc[1] + nh//2 + self.perceptor.ventana.top
                    print(f"  {etiqueta} encontrado (conf={val:.2f})")
                    pyautogui.click(cx, cy)
                    return True
            time.sleep(0.3)
        return False

    def close(self):
        pyautogui.keyUp('w')
        cv2.destroyAllWindows()
        print("Entorno cerrado")