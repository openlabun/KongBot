"""
entrenar.py — Script de entrenamiento PPO para Banana Kong
Uso:
    python entrenar.py              # entrenamiento desde cero
    python entrenar.py --continuar  # continuar entrenamiento previo
"""

import argparse
import os
import time
import signal
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from entorno.entorno import BananaKongEnv

N_STACK = 4  # número de estados apilados

# Configuracion
RUTA_MODELO      = "modelos/banana_kong_ppo"
RUTA_LOGS        = "logs/"
RUTA_CHECKPOINTS = "modelos/checkpoints/"
TOTAL_TIMESTEPS  = 50_000
GUARDAR_CADA     = 10_000

PPO_CONFIG = {
    "learning_rate":   3e-4,
    "n_steps":         2048,
    "batch_size":      128,
    "n_epochs":        10,
    "gamma":           0.99,
    "gae_lambda":      0.95,
    "clip_range":      0.2,
    #"ent_coef":       0.01,
    "verbose":         1,
    "tensorboard_log": RUTA_LOGS,
}

# Parada limpia con Ctrl+C
modelo_global = None

def parada_emergencia(sig, frame):
    print("\nCtrl+C detectado — guardando modelo...")
    if modelo_global is not None:
        modelo_global.save(RUTA_MODELO + "_interrumpido")
        print(f"Modelo guardado en {RUTA_MODELO}_interrumpido.zip")
    sys.exit(0)

signal.signal(signal.SIGINT, parada_emergencia)


def main():
    global modelo_global

    parser = argparse.ArgumentParser()
    parser.add_argument("--continuar", action="store_true",
                        help="Continuar entrenamiento desde ultimo checkpoint")
    args = parser.parse_args()

    os.makedirs("modelos", exist_ok=True)
    os.makedirs(RUTA_CHECKPOINTS, exist_ok=True)
    os.makedirs(RUTA_LOGS, exist_ok=True)

    print("=== ENTRENAMIENTO BANANA KONG PPO ===")
    print("Asegurate de que BlueStacks este abierto y el juego corriendo")
    print("Ctrl+C para pausar y guardar\n")
    time.sleep(3)

    print("Inicializando entorno...")
    env = DummyVecEnv([lambda: Monitor(BananaKongEnv(), RUTA_LOGS)])
    env = VecFrameStack(env, n_stack=N_STACK)

    checkpoint_callback = CheckpointCallback(
        save_freq=GUARDAR_CADA,
        save_path=RUTA_CHECKPOINTS,
        name_prefix="banana_kong",
        verbose=1,
    )

    if args.continuar and os.path.exists(RUTA_MODELO + ".zip"):
        print(f"Cargando modelo desde {RUTA_MODELO}.zip...")
        modelo = PPO.load(RUTA_MODELO, env=env)
    else:
        print("Creando modelo nuevo...")
        modelo = PPO("MlpPolicy", env=env, **PPO_CONFIG)

    modelo_global = modelo

    print(f"Entrenando por {TOTAL_TIMESTEPS:,} steps...")
    print(f"Checkpoints cada {GUARDAR_CADA:,} steps en {RUTA_CHECKPOINTS}")
    print(f"Para ver progreso: tensorboard --logdir {RUTA_LOGS}\n")

    modelo.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        reset_num_timesteps=not args.continuar,
        progress_bar=True,
    )

    modelo.save(RUTA_MODELO)
    print(f"\nEntrenamiento completado — modelo guardado en {RUTA_MODELO}.zip")
    env.close()


if __name__ == "__main__":
    main()