"""
evaluar.py — Evaluación del agente entrenado vs baseline aleatorio
Uso:
    python -m entrenamiento.evaluar                    # evaluar último modelo
    python -m entrenamiento.evaluar --baseline         # solo baseline aleatorio
    python -m entrenamiento.evaluar --ambos            # agente + baseline + comparar
    python -m entrenamiento.evaluar --modelo <ruta>    # modelo específico
"""

import argparse
import os
import re
import time
import numpy as np
from collections import deque
from scipy import stats

from stable_baselines3 import PPO
from entorno.entorno import BananaKongEnv

RUTA_MODELO       = "modelos/banana_kong_ppo"
RUTA_LOGS         = "logs"
RUTA_CHECKPOINTS  = "modelos/checkpoints"
N_EPISODIOS       = 30
N_STACK           = 4
OBS_SIZE          = 23


# ── Utilidades ──────────────────────────────────────────────────────

def extraer_run_id(ruta):
    nombre = os.path.basename(ruta).replace(".zip", "")
    m = re.match(r"banana_kong_ppo_(.+)_final$", nombre)
    return m.group(1) if m else None


def encontrar_ultimo_checkpoint(run_id):
    d = os.path.join(RUTA_CHECKPOINTS, f"run_{run_id}")
    if not os.path.exists(d):
        return None
    zips = [f for f in os.listdir(d) if f.endswith("_steps.zip")]
    if not zips:
        return None
    zips.sort(key=lambda x: int(re.search(r'(\d+)_steps', x).group(1)))
    return os.path.join(d, zips[-1][:-4])


def cargar_vecnormalize(run_id):
    path = os.path.join(RUTA_LOGS, f"run_{run_id}", "vecnormalize.pkl")
    if not os.path.exists(path):
        return None
    try:
        import cloudpickle
        with open(path, "rb") as f:
            vn = cloudpickle.load(f)
        vn.training = False
        vn.norm_reward = False
        return vn
    except Exception as e:
        print(f"  ⚠️  Error cargando VecNormalize: {e}")
        return None


# ── Frame stacking wrapper ──────────────────────────────────────────

class FrameStackWrapper:
    """Stackea frames para modelos entrenados con VecFrameStack.
    Normaliza antes de stackear (mismo orden que entrenamiento).
    Mantiene API Gymnasium.
    """
    def __init__(self, env, n_stack=4, vecnormalize=None):
        self.env = env
        self.n_stack = n_stack
        self.buffer = None
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._vn = vecnormalize

    def _norm(self, obs):
        if self._vn is None:
            return obs
        return self._vn.normalize_obs(obs)

    def reset(self):
        obs, info = self.env.reset()
        obs = self._norm(obs)
        self.buffer = deque([obs.copy()] * self.n_stack, maxlen=self.n_stack)
        return self._get_stacked(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._norm(obs)
        self.buffer.append(obs.copy())
        return self._get_stacked(), reward, terminated, truncated, info

    def _get_stacked(self):
        return np.concatenate(self.buffer).astype(np.float32)

    def close(self):
        self.env.close()


# ── Evaluación ──────────────────────────────────────────────────────

def evaluar_agente(env, modelo, n_episodios, etiqueta="Agente"):
    rewards, duraciones, bananas = [], [], []
    episodios_completados = 0

    print(f"\n=== {etiqueta} ({n_episodios} episodios) ===")
    
    while episodios_completados < n_episodios:
        obs, _ = env.reset()
        reward_total = 0
        steps = 0

        while True:
            if modelo is not None:
                accion, _ = modelo.predict(obs, deterministic=True)
            else:
                accion = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(int(accion))
            reward_total += reward
            steps += 1

            if terminated or truncated:
                break

        rewards.append(reward_total)
        duraciones.append(steps)
        episodios_completados += 1
        # Las bananas se capturan del print "[ep end]" del entorno

    print(f"\n  Reward  — media={np.mean(rewards):+.2f}  std={np.std(rewards):.2f}  min={np.min(rewards):+.2f}  max={np.max(rewards):+.2f}")
    print(f"  Steps   — media={np.mean(duraciones):.1f}  std={np.std(duraciones):.1f}")
    return rewards, duraciones, []


def comparar(rewards_a, rewards_b):
    media_a, media_b = np.mean(rewards_a), np.mean(rewards_b)
    mejora = media_a - media_b
    t_stat, p_valor = stats.ttest_ind(rewards_a, rewards_b)

    print("\n=== COMPARACIÓN ESTADÍSTICA ===")
    print(f"  Agente:   {media_a:+.2f}")
    print(f"  Baseline: {media_b:+.2f}")
    print(f"  Mejora:   {mejora:+.2f}")
    print(f"  t-stat:   {t_stat:.3f}")
    print(f"  p-valor:  {p_valor:.4f}")

    if p_valor < 0.05 and media_a > media_b:
        print("\n  ✅ El agente supera el baseline (p < 0.05) — objetivo CUMPLIDO")
    elif media_a > media_b:
        print(f"\n  ⚠️  Mejor reward pero no significativo (p={p_valor:.3f})")
        print("      Necesitás más episodios o más entrenamiento.")
    else:
        print("\n  ❌ El agente NO supera el baseline.")


# ── Creación de entorno según el modelo ─────────────────────────────

def crear_env(obs_dim, run_id=None):
    env = BananaKongEnv()

    if obs_dim == OBS_SIZE:
        print(f"  Obs = {obs_dim} → entorno directo")
        return env

    if obs_dim == N_STACK * OBS_SIZE:
        print(f"  Obs = {obs_dim} → FrameStack({N_STACK})")
        vn = cargar_vecnormalize(run_id) if run_id else None
        if vn:
            print(f"  VecNormalize aplicado (run={run_id})")
        return FrameStackWrapper(env, N_STACK, vn)

    print(f"  ❌ Modelo incompatible: espera obs_dim={obs_dim}, "
          f"código actual={OBS_SIZE} (stackeado={N_STACK*OBS_SIZE})")
    if obs_dim == 60:
        print("     Este modelo es de una versión anterior del entorno (60 observaciones).")
        print("     No se puede evaluar con el código actual — necesitás el entorno viejo.")
    return None


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="Solo baseline aleatorio")
    parser.add_argument("--ambos",    action="store_true", help="Agente + baseline + comparar")
    parser.add_argument("--n",        type=int, default=N_EPISODIOS, help="Número de episodios")
    parser.add_argument("--modelo",   type=str, default=RUTA_MODELO, help="Ruta al modelo (sin .zip)")
    args = parser.parse_args()

    print("=== EVALUACIÓN BANANA KONG ===")
    print("Asegurate de que BlueStacks esté abierto y el juego corriendo")
    time.sleep(3)

    modelo = None
    run_id = extraer_run_id(args.modelo)

    if not args.baseline:
        ruta = args.modelo + ".zip" if not args.modelo.endswith(".zip") else args.modelo

        if not os.path.exists(ruta):
            print(f"❌ No se encontró modelo en {ruta}")
            checkpoints = encontrar_ultimo_checkpoint(run_id) if run_id else None
            if checkpoints:
                print(f"   Usando último checkpoint: {checkpoints}.zip")
                ruta = checkpoints + ".zip"
            else:
                print("   No hay checkpoints disponibles.")
                return

        modelo = PPO.load(ruta.replace(".zip", ""))
        obs_dim = modelo.observation_space.shape[0]
        print(f"  Modelo cargado: obs_dim={obs_dim}")

    else:
        obs_dim = OBS_SIZE

    env = crear_env(obs_dim, run_id)
    if env is None:
        return

    if args.baseline:
        evaluar_agente(env, None, args.n, etiqueta="Baseline aleatorio")

    elif args.ambos:
        rewards_a, _, _ = evaluar_agente(env, modelo, args.n, etiqueta="Agente PPO")
        env_base = BananaKongEnv()
        rewards_b, _, _ = evaluar_agente(env_base, None, args.n, etiqueta="Baseline aleatorio")
        env_base.close()
        comparar(rewards_a, rewards_b)

    else:
        evaluar_agente(env, modelo, args.n, etiqueta="Agente PPO")

    env.close()


if __name__ == "__main__":
    main()
