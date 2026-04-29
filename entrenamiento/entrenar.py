"""
entrenar.py — Script de entrenamiento PPO para Banana Kong

USO SIMPLE:
    python entrenar.py                                    # Entrenamiento nuevo
    python entrenar.py --perfil explorador                # Perfil con más exploración
    python entrenar.py --continuar                        # Continuar último entrenamiento
    python entrenar.py --continuar --add-steps 50000      # +50k más
    python entrenar.py --total-steps 500000               # Entrenar hasta 500k
    python entrenar.py --run-id exp_v1 --perfil agresivo
    python entrenar.py --listar-perfiles                  # Ver perfiles disponibles
"""

import argparse
import os
import time
import signal
import sys
import glob
import json
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from entorno.entorno import BananaKongEnv

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

N_STACK = 4
RUTA_MODELO      = "modelos/banana_kong_ppo"
RUTA_LOGS        = "logs/"
RUTA_CHECKPOINTS = "modelos/checkpoints/"
GUARDAR_CADA     = 10_000
ARCHIVO_RUN_ID   = ".run_id_actual"

# ============================================================================
# PERFILES DE CONFIGURACIÓN PPO
# ============================================================================

PPO_PERFILES = {
    "default": {
        "learning_rate": 1e-4,
        "n_steps": 2048,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.008,
        "verbose": 1,
    },
    "conservador": {
        "learning_rate": 7e-5,
        "n_steps": 2048,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.15,
        "ent_coef": 0.005,
        "verbose": 1,
    },
    "explorador": {
        "learning_rate": 2e-4,
        "n_steps": 4096,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.015,
        "verbose": 1,
    },
    "agresivo": {
        "learning_rate": 1e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 8,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.15,
        "ent_coef": 0.02,
        "verbose": 1,
    },
    "rapido": {
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "batch_size": 64,
        "n_epochs": 8,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "verbose": 1,
    },
}

# ============================================================================
# VARIABLES GLOBALES
# ============================================================================

modelo_global = None
env_global = None
run_id = None
log_dir = None
vecnormalize_path = None


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def mostrar_perfiles():
    """Muestra los perfiles disponibles"""
    print("\n📋 Perfiles disponibles:")
    print("-" * 60)
    for nombre, config in PPO_PERFILES.items():
        print(f"  {nombre:12} → lr={config['learning_rate']:.0e}, "
              f"batch={config['batch_size']}, "
              f"ent_coef={config['ent_coef']}")
    print("-" * 60)


def obtener_run_id(args):
    """Obtiene el run_id de forma inteligente"""
    if args.run_id:
        with open(ARCHIVO_RUN_ID, "w") as f:
            f.write(args.run_id)
        return args.run_id

    if args.continuar:
        if os.path.exists(ARCHIVO_RUN_ID):
            with open(ARCHIVO_RUN_ID, "r") as f:
                return f.read().strip()
        else:
            print("\n❌ ERROR: No hay entrenamiento previo para continuar")
            print("   Ejecuta sin --continuar para empezar uno nuevo")
            sys.exit(1)

    rid = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(ARCHIVO_RUN_ID, "w") as f:
        f.write(rid)
    return rid


def parada_emergencia(sig, frame):
    """Maneja Ctrl+C guardando modelo y VecNormalize"""
    print("\n" + "="*60)
    print("⚠️  Ctrl+C detectado — guardando estado...")
    print("="*60)

    if modelo_global:
        ruta_backup = f"{RUTA_MODELO}_{run_id}_backup"
        modelo_global.save(ruta_backup)
        print(f"💾 Modelo guardado en {ruta_backup}.zip")

    if env_global and vecnormalize_path:
        env_global.save(vecnormalize_path)
        print(f"💾 VecNormalize guardado en {vecnormalize_path}")

    print(f"📁 Run ID: {run_id}")
    print(f"📁 Logs en: {log_dir}")
    sys.exit(0)


def extraer_steps_de_checkpoint(path_checkpoint):
    """Extrae el número de steps del nombre de un checkpoint"""
    try:
        nombre_base = os.path.basename(path_checkpoint)
        partes = nombre_base.replace(".zip", "").split("_")
        for i, parte in enumerate(partes):
            if parte.isdigit() and i + 1 < len(partes) and partes[i + 1] == "steps":
                return int(parte)
        return 0
    except:
        return 0


def encontrar_ultimo_checkpoint(ruta_checkpoints_run, prefijo):
    """Encuentra el checkpoint más reciente por número de steps"""
    patron = os.path.join(ruta_checkpoints_run, f"{prefijo}_*_steps.zip")
    archivos = glob.glob(patron)
    if not archivos:
        return None
    archivos_con_steps = [(extraer_steps_de_checkpoint(f), f) for f in archivos]
    archivos_con_steps.sort(reverse=True)
    return archivos_con_steps[0][1]


def guardar_metadata(log_dir, args, ppo_config, steps_previos, modo, timesteps_a_entrenar):
    """Guarda información del run"""
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "modo": modo,
        "perfil": args.perfil if hasattr(args, 'perfil') else "default",
        "steps_previos": steps_previos,
        "steps_a_entrenar": timesteps_a_entrenar,
        "vecnormalize": True,
        "n_stack": N_STACK,
        "config": ppo_config.copy(),
    }
    ruta_metadata = os.path.join(log_dir, "run_metadata.json")
    with open(ruta_metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def formatear_tiempo(segundos):
    """Formatea segundos en formato legible"""
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    segs = int(segundos % 60)
    if horas > 0:
        return f"{horas}h {minutos}m {segs}s"
    elif minutos > 0:
        return f"{minutos}m {segs}s"
    else:
        return f"{segs}s"


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    global modelo_global, env_global, run_id, log_dir, vecnormalize_path

    # ------------------------------------------------------------------------
    # Argumentos
    # ------------------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Entrenamiento PPO para Banana Kong",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EJEMPLOS:
  # Perfiles predefinidos
  python entrenar.py --perfil explorador
  python entrenar.py --perfil agresivo --run-id exp_agresivo
  
  # Continuar entrenamiento
  python entrenar.py --continuar
  python entrenar.py --continuar --add-steps 50000
  
  # Objetivo fijo
  python entrenar.py --total-steps 500000 --perfil conservador
        """
    )

    parser.add_argument("--continuar", action="store_true",
                        help="Continuar último entrenamiento")
    parser.add_argument("--run-id", type=str, default=None,
                        help="ID personalizado (opcional)")
    parser.add_argument("--perfil", type=str, default="default",
                        choices=list(PPO_PERFILES.keys()),
                        help="Perfil de configuración PPO")
    parser.add_argument("--listar-perfiles", action="store_true",
                        help="Mostrar perfiles disponibles y salir")

    modo_group = parser.add_mutually_exclusive_group()
    modo_group.add_argument("--total-steps", type=int, default=None,
                           help="Entrenar hasta X steps totales")
    modo_group.add_argument("--add-steps", type=int, default=100000,
                           help="Añadir X steps (default: 100k)")

    args = parser.parse_args()

    if args.listar_perfiles:
        mostrar_perfiles()
        return

    # ------------------------------------------------------------------------
    # Inicialización
    # ------------------------------------------------------------------------

    run_id = obtener_run_id(args)
    ppo_config = PPO_PERFILES[args.perfil].copy()

    if args.total_steps is not None:
        modo = "objetivo_fijo"
        steps_param = args.total_steps
    else:
        modo = "incremental"
        steps_param = args.add_steps

    log_dir = os.path.join(RUTA_LOGS, f"run_{run_id}")
    monitor_dir = os.path.join(log_dir, "monitor")
    tensorboard_dir = os.path.join(log_dir, "tensorboard")
    checkpoints_dir = os.path.join(RUTA_CHECKPOINTS, f"run_{run_id}")
    vecnormalize_path = os.path.join(log_dir, "vecnormalize.pkl")

    for d in ["modelos", monitor_dir, tensorboard_dir, checkpoints_dir]:
        os.makedirs(d, exist_ok=True)

    ppo_config["tensorboard_log"] = tensorboard_dir
    signal.signal(signal.SIGINT, parada_emergencia)

    # ------------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------------

    print("\n" + "="*60)
    print("🎮 ENTRENAMIENTO BANANA KONG - PPO")
    print("="*60)
    print(f"📁 Run ID: {run_id}")
    print(f"🎯 Modo: {modo.upper()}")
    print(f"⚙️  Perfil: {args.perfil.upper()}")
    print(f"{'🔄 Continuando' if args.continuar else '🆕 Nuevo entrenamiento'}")
    print("="*60)

    print("\n📊 Configuración PPO:")
    print(f"   learning_rate: {ppo_config['learning_rate']}")
    print(f"   batch_size:    {ppo_config['batch_size']}")
    print(f"   n_epochs:      {ppo_config['n_epochs']}")
    print(f"   ent_coef:      {ppo_config['ent_coef']}")
    print(f"   clip_range:    {ppo_config['clip_range']}")

    if not args.continuar:
        print("\n⚠️  Asegúrate de que BlueStacks y Banana Kong están abiertos")
        time.sleep(3)

    # ------------------------------------------------------------------------
    # Entorno + VecNormalize
    # ------------------------------------------------------------------------

    print("\n🔄 Inicializando entorno...")

    # 1. Entorno base
    env = DummyVecEnv([lambda: Monitor(BananaKongEnv(), monitor_dir)])

    # 2. VecNormalize (ANTES de FrameStack)
    if args.continuar and os.path.exists(vecnormalize_path):
        print("📦 Cargando VecNormalize guardado...")
        env = VecNormalize.load(vecnormalize_path, env)
        env.training = True
        env.norm_reward = True
    else:
        print("📦 Creando nuevo VecNormalize...")
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=ppo_config["gamma"]
        )

    # 3. FrameStack (AL FINAL)
    env = VecFrameStack(env, n_stack=N_STACK)

    env_global = env
    print("✅ Entorno listo (con VecNormalize)")

    # ------------------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------------------

    checkpoint_callback = CheckpointCallback(
        save_freq=GUARDAR_CADA,
        save_path=checkpoints_dir,
        name_prefix=f"banana_kong_{run_id}",
        verbose=1,
    )

    # ------------------------------------------------------------------------
    # Carga del modelo
    # ------------------------------------------------------------------------

    steps_previos = 0
    prefijo = f"banana_kong_{run_id}"

    if args.continuar:
        print("\n🔍 Buscando checkpoint...")
        ultimo_checkpoint = encontrar_ultimo_checkpoint(checkpoints_dir, prefijo)

        if ultimo_checkpoint:
            steps_previos = extraer_steps_de_checkpoint(ultimo_checkpoint)
            print(f"✅ Cargado: {os.path.basename(ultimo_checkpoint)}")
            print(f"   Steps acumulados: {steps_previos:,}")

            if modo == "objetivo_fijo":
                timesteps_a_entrenar = max(0, steps_param - steps_previos)
                if timesteps_a_entrenar <= 0:
                    print(f"\n🎉 ¡Objetivo ya alcanzado! ({steps_previos:,} steps)")
                    env.close()
                    return
                print(f"   Steps restantes: {timesteps_a_entrenar:,}")
            else:
                timesteps_a_entrenar = steps_param
                print(f"   Añadiendo: {steps_param:,} steps")
                print(f"   Total después: {steps_previos + steps_param:,}")

            modelo = PPO.load(ultimo_checkpoint, env=env)
        else:
            print(f"\n❌ No se encontraron checkpoints para run_id='{run_id}'")
            sys.exit(1)
    else:
        print("\n🆕 Creando modelo nuevo...")
        modelo = PPO("MlpPolicy", env=env, **ppo_config)
        timesteps_a_entrenar = steps_param
        print(f"   Steps a entrenar: {timesteps_a_entrenar:,}")

    modelo_global = modelo
    guardar_metadata(log_dir, args, ppo_config, steps_previos, modo, timesteps_a_entrenar)

    # ------------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------------

    print("\n" + "="*60)
    print("⚙️  CONFIGURACIÓN DE ENTRENAMIENTO")
    print("="*60)
    print(f"Steps a entrenar: {timesteps_a_entrenar:,}")
    print(f"Steps acumulados: {steps_previos:,}")
    print(f"Total después:    {steps_previos + timesteps_a_entrenar:,}")
    print(f"Checkpoints cada: {GUARDAR_CADA:,}")
    print(f"VecNormalize:     {'✅ Activado' if env.norm_obs else '❌ Desactivado'}")
    print("="*60)

    print("\n🚀 ENTRENANDO...")
    print(f"📊 tensorboard --logdir {RUTA_LOGS}")
    print("-"*60 + "\n")

    tiempo_inicio = time.time()

    try:
        modelo.learn(
            total_timesteps=timesteps_a_entrenar,
            callback=checkpoint_callback,
            reset_num_timesteps=False if args.continuar else True,
            progress_bar=True,
        )

        tiempo_total = time.time() - tiempo_inicio

        print("\n" + "="*60)
        print("✅ ¡ENTRENAMIENTO COMPLETADO!")
        print("="*60)
        print(f"⏱️  Tiempo: {formatear_tiempo(tiempo_total)}")
        print(f"📊 Steps totales: {steps_previos + timesteps_a_entrenar:,}")

        modelo.save(f"{RUTA_MODELO}_{run_id}_final")
        modelo.save(RUTA_MODELO)
        env.save(vecnormalize_path)

        print(f"💾 Modelo guardado en {RUTA_MODELO}_{run_id}_final.zip")
        print(f"💾 VecNormalize guardado en {vecnormalize_path}")
        print("="*60)

    except KeyboardInterrupt:
        parada_emergencia(None, None)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        if modelo_global:
            modelo_global.save(f"{RUTA_MODELO}_{run_id}_error")
        if env_global:
            env_global.save(vecnormalize_path)
        import traceback
        traceback.print_exc()

    finally:
        env.close()
        print("\n👋 Listo!")


if __name__ == "__main__":
    main()