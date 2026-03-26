# Bot Autónomo para Banana Kong - Aprendizaje por Refuerzo

**Universidad del Norte - Facultad de Ingeniería de Sistemas**  
**Proyecto Final - Grupo 2 - Aprendizaje por Refuerzo**  
**Kidman Cabana, Santiago Romero - Barranquilla, Colombia - 2026**

---

## Tabla de Contenidos

1. [Introducción](#1-introducción)
2. [Planteamiento del Problema](#2-planteamiento-del-problema)
3. [Restricciones y Supuestos](#3-restricciones-y-supuestos)
4. [Alcance del Proyecto](#4-alcance-del-proyecto)
5. [Objetivos](#5-objetivos)
6. [Estado del Arte](#6-estado-del-arte)
7. [Propuesta de Solución](#7-propuesta-de-solución)
8. [Implementación Técnica](#8-implementación-técnica)
9. [Decisiones de Diseño y Alternativas Evaluadas](#9-decisiones-de-diseño-y-alternativas-evaluadas)
10. [Requerimientos](#10-requerimientos)
11. [Criterios de Aceptación](#11-criterios-de-aceptación)
12. [Cronograma del Proyecto](#12-cronograma-del-proyecto)
13. [Diagramas](#13-diagramas)
14. [Instalación y Uso](#14-instalación-y-uso)
15. [Referencias](#15-referencias)

---

## 1. Introducción

Banana Kong es un videojuego de plataformas y carrera continua (*endless runner*) desarrollado por FDG Entertainment, disponible para plataformas móviles Android e iOS. El juego presenta a un gorila que debe desplazarse por una selva tropical recolectando plátanos, esquivando obstáculos y utilizando animales de apoyo para avanzar. Su espacio de acciones reducido (salto/planeo, dash y agacharse) lo convierte en un candidato adecuado para el entrenamiento de un agente basado en aprendizaje por refuerzo.

Este proyecto propone construir un agente que perciba el juego exclusivamente a través de la pantalla y ejecute acciones simulando entradas de teclado, sin acceso a la memoria del juego ni modificación del APK. El módulo de percepción usa visión por computador (OpenCV) con detectores especializados por tipo de objeto; el módulo de decisión usa PPO (Proximal Policy Optimization) implementado con Stable-Baselines3 con frame stacking de 4 estados consecutivos.

---

## 2. Planteamiento del Problema

Los videojuegos comerciales son sistemas de caja negra: no exponen su estado interno mediante APIs públicas. La única información disponible para un agente externo es la imagen renderizada en pantalla. Esto genera una brecha técnica concreta: integrar captura visual, percepción computacional y ejecución de acciones en un pipeline coherente que opere en tiempo real es un problema de ingeniería no trivial, especialmente con hardware académico limitado.

**Pregunta central:** ¿Es posible diseñar e implementar, bajo restricciones académicas de tiempo y hardware, un agente autónomo basado en aprendizaje por refuerzo que aprenda a jugar Banana Kong en un emulador Android para PC, utilizando únicamente información visual y simulación de entradas de teclado, alcanzando un puntaje promedio de 5.000–6.000 puntos por partida?

---

## 3. Restricciones y Supuestos

### 3.1 Restricciones Técnicas

- **Sin acceso interno al juego:** El sistema trata Banana Kong como caja negra. No se lee ni modifica la memoria del proceso, ni se inyecta código en el emulador.
- **Captura exclusivamente visual:** Toda la información del estado proviene de capturas de pantalla con `mss`. No se usa audio, tráfico de red ni otras fuentes.
- **Acciones mediante teclado simulado:** Las interacciones se ejecutan a través de `pyautogui` simulando las teclas configuradas en BlueStacks Game Controls. No se usa ADB por problemas de latencia y conflicto con eventos táctiles.
- **Resolución fija 960×540:** Todos los detectores están calibrados para esta resolución. Cambiarla requiere recalibrar ROIs y umbrales.
- **Latencia objetivo:** El ciclo completo captura → percepción → decisión → acción debe completarse en menos de 100 ms.
- **Hardware de consumo:** Desarrollo en equipos con GPU AMD RX 9070 XT. El entrenamiento corre en CPU dado que ROCm (el equivalente de CUDA para AMD) no tiene soporte oficial en Windows.

### 3.2 Restricciones del Entorno de Juego

- **Meta de puntaje:** El agente debe alcanzar un puntaje promedio de **5.000–6.000 puntos** por episodio como criterio de éxito.
- **Restricción de mundos alternativos:** El agente **no debe entrar a mundos alternativos** accesibles mediante cuevas, zonas de agua, cielo u otros portales. Estos mundos cambian radicalmente la paleta de colores, la geometría de obstáculos y la estructura del HUD, invalidando todos los detectores calibrados para el mundo principal (selva). Esta restricción se implementa penalizando fuertemente la detección de agua (que precede a las transiciones de mundo) y limitando el ROI de percepción.
- **Mundo único:** El agente opera exclusivamente en el mundo de la selva (mundo inicial). No se contemplan otros biomas.
- **Configuración gráfica fija:** La ventana del emulador permanece en primer plano y visible durante toda la ejecución.

### 3.3 Restricciones Normativas

- El proyecto es estrictamente académico y no comercial.
- No se redistribuye el APK del juego.
- El bot opera exclusivamente en modalidad de un jugador (offline).

### 3.4 Supuestos

- Los elementos clave del juego son visualmente distinguibles con las técnicas implementadas en condiciones normales del mundo selva.
- Los colores, formas y posiciones de los elementos del HUD y del entorno son consistentes entre partidas dentro del mismo mundo.
- El juego no recibirá actualizaciones que cambien significativamente su interfaz visual durante el semestre.

---

## 4. Alcance del Proyecto

### Incluido

- Pipeline completo: captura → percepción → decisión → acción
- Detectores especializados para: Kong, barriles, bananas, agua, rocas, muros (madera y piedra), game over
- Entorno compatible con la interfaz OpenAI Gymnasium
- Entrenamiento con PPO usando Stable-Baselines3 con frame stacking
- Reinicio automático de episodios
- Evaluación frente a política aleatoria de referencia (*baseline*)
- Documentación técnica completa

### Excluido

- Soporte para múltiples juegos o biomas distintos al mundo selva
- Detección de objetos interactivos opcionales (lianas, trampolines, guacamaya, plataformas flotantes) — se dejan para iteraciones futuras una vez consolidada la política básica de supervivencia
- Interfaz gráfica de usuario (GUI): la ejecución es por línea de comandos
- Modificación del APK, archivos del emulador o código del juego
- Generalización a múltiples resoluciones o versiones del juego

---

## 5. Objetivos

### General

Diseñar e implementar un agente autónomo basado en aprendizaje por refuerzo profundo que aprenda a jugar Banana Kong en un emulador Android para PC, utilizando exclusivamente información visual de la pantalla y simulación de teclado, alcanzando al final del semestre un puntaje promedio de 5.000–6.000 puntos por episodio, superior al de una política aleatoria de referencia.

### Específicos

1. Implementar un módulo de captura capaz de obtener fotogramas del emulador a mínimo 15 FPS con latencia individual menor a 50 ms.
2. Desarrollar detectores de visión por computador para cada tipo de objeto relevante del juego, con precisión superior al 85% en condiciones normales del mundo selva.
3. Diseñar y formalizar el entorno Gymnasium con espacio de estados, acciones y función de recompensa.
4. Entrenar al menos un agente PPO durante un mínimo de 500.000 pasos, documentando curvas de aprendizaje.
5. Evaluar el agente frente a una política aleatoria, demostrando mejora estadísticamente significativa en puntaje promedio por episodio en al menos 30 episodios.
6. Documentar el sistema completo en este repositorio con READMEs, diagramas y resultados de experimentos.

---

## 6. Estado del Arte

### 6.1 Aprendizaje por Refuerzo en Videojuegos

El trabajo de Mnih et al. (2015) con DQN demostró que una red neuronal puede aprender políticas de juego competitivas directamente desde píxeles en juegos de Atari. Schulman et al. (2017) propusieron PPO, algoritmo de gradiente de política con mayor estabilidad de entrenamiento, que es el que utilizamos en este proyecto por su buen desempeño con espacios de acción discretos pequeños y su disponibilidad en Stable-Baselines3.

### 6.2 Bots para Endless Runners

Proyectos como el bot para Subway Surfers de Yeh et al. (2021) usaron visión por computador con OpenCV para detectar obstáculos mediante segmentación por color, sin aprendizaje automático. Lograron tiempos de supervivencia superiores al jugador promedio pero con robustez limitada a condiciones de color constante. Nuestro enfoque híbrido (HSV + template matching + RL) busca mayor generalización.

### 6.3 Vacíos que Abordamos

- Escasez de implementaciones académicas reproducibles de agentes RL visuales para juegos móviles en emulador
- Ausencia de pipelines completos documentados que integren percepción clásica con RL para endless runners en plataformas de consumo
- Falta de comparación directa entre detección puramente por color vs. detección híbrida para objetos con alta variación visual

---

## 7. Propuesta de Solución

### 7.1 Arquitectura General

```
BlueStacks (960x540)
        │
        ▼
┌─────────────────────────────────────────┐
│              Perceptor                  │
│  ┌──────────────────┐  ┌─────────────┐ │
│  │   Hilo Rápido    │  │ Hilo Lento  │ │
│  │ Kong + Bananas   │  │ Barriles    │ │
│  │ + Colisiones     │  │ Rocas       │ │
│  │ (~20 FPS)        │  │ Muros       │ │
│  └────────┬─────────┘  │ Agua        │ │
│           │             │ Game Over   │ │
│           └──────┬──────┘             │ │
│                  ▼                    │ │
│            Estado compartido          │ │
│            (threading.Lock)           │ │
└──────────────────┬──────────────────-─┘
                   │ estado (dict)
                   ▼
        ┌──────────────────┐
        │  BananaKongEnv   │  Entorno Gymnasium:
        │  (entorno.py)    │  obs vector 13 floats × 4 frames
        │                  │  Calcula reward, detecta terminación
        └────────┬─────────┘
                 │ obs (52 floats con frame stacking)
                 ▼
        ┌──────────────────┐
        │   PPO Agent      │  Stable-Baselines3 MlpPolicy
        │ + VecFrameStack  │  selecciona acción 0-3
        └────────┬─────────┘
                 │ acción
                 ▼
        ┌──────────────────┐
        │  ModuloAcciones  │  pyautogui: ejecuta tecla en BlueStacks
        └──────────────────┘
```

La arquitectura de dos hilos en el Perceptor es clave: el hilo rápido corre Kong y Bananas en cada frame (crítico para la detección de colisiones y el reward), mientras el hilo lento corre el resto de detectores con cadencias variables, sin bloquear al agente.

### 7.2 Espacio de Acciones

| ID | Acción | Tecla BlueStacks | Descripción |
|----|--------|-----------------|-------------|
| 0 | NADA | — | El juego avanza automáticamente |
| 1 | PLANEAR | W | Tap = saltar; mantener = planear |
| 2 | DASH | D | Impulso hacia adelante |
| 3 | BAJAR | S | Deslizarse hacia abajo |

**Nota sobre la implementación de controles:** Inicialmente se intentó simular el dash mediante `pyautogui.drag()`, lo que provocaba que BlueStacks registrara el inicio del drag como un tap, haciendo saltar a Kong antes del dash. La solución adoptada fue configurar el dash directamente en el **Game Controls de BlueStacks** como una tecla (`D`), eliminando por completo la necesidad de simular gestos táctiles.

**Planeo implícito:** Mientras el agente seleccione PLANEAR en steps consecutivos, el módulo mantiene W presionado (`keyDown`). Al seleccionar cualquier otra acción, suelta W (`keyUp`). Esto permite que la duración del planeo emerja del comportamiento aprendido sin necesidad de una acción separada.

### 7.3 Función de Recompensa

| Evento | Recompensa |
|--------|-----------|
| Sobrevivir cada step | +0.02 |
| Banana recogida | +1.0 por banana |
| Game over | −10.0 |

**Detección de bananas recogidas:** Se utiliza detección de colisión por bounding box entre el rect de Kong (expandido 10px) y los rects de bananas visibles. La lógica del pico de colisiones (`_pico_colisiones`) acumula el máximo de colisiones simultáneas y contabiliza cuando baja a cero, evitando doble conteo y falsos positivos por frames perdidos. Esta lógica corre en el hilo rápido del Perceptor a máxima velocidad, sin perder colisiones entre steps del agente.

**Balance del reward:** El reward de supervivencia (+0.02/step) se calibró para que en un episodio de ~100 steps genere +2.0, inferior al costo del game over (−10.0). Esto incentiva sobrevivir sin que el agente aprenda que morir es gratuito.

**Período de gracia:** El game over se ignora durante los primeros 10 steps de cada episodio para evitar falsos positivos durante la transición de pantalla del reinicio.

### 7.4 Vector de Observación

El estado se convierte a un vector de **13 floats normalizados [0, 1]**:

```
[0]   kong_cx
[1]   kong_cy
[2]   banana1_dx  (distancia horizontal relativa a Kong, centrado en 0.5)
[3]   banana1_cy
[4]   banana2_dx
[5]   banana2_cy
[6]   hay_agua    (0 o 1)
[7]   barril1_dx  (distancia horizontal relativa a Kong)
[8]   barril1_cy
[9]   roca1_dx    (distancia horizontal relativa a Kong)
[10]  roca1_cy
[11]  muro1_dx    (distancia horizontal relativa a Kong)
[12]  muro1_cy
```

Los obstáculos y bananas se expresan como **distancia horizontal relativa a Kong** (centrada en 0.5) en lugar de posición absoluta. Esto le da al agente información invariante: un valor de 0.7 siempre significa "el obstáculo está 0.2 a la derecha de Kong", independientemente de dónde esté Kong en pantalla.

Con **frame stacking de 4 estados**, el observation space efectivo es de **52 floats**, permitiendo al agente inferir velocidad y dirección de los obstáculos a partir del cambio entre estados consecutivos.

---

## 8. Implementación Técnica

### 8.1 Módulo de Percepción — Arquitectura de Hilos

El `Perceptor` opera con dos hilos de detección independientes más un hilo de display:

**Hilo rápido** (corre en cada frame, ~20 FPS):
- Detector de Kong (CSRT + HSV + Template Matching)
- Detector de Bananas (HSV)
- Lógica de colisión banana-Kong

**Hilo lento** (cadencia por detector):
- Game Over: cada 10 frames
- Agua: cada 3 frames
- Barriles: cada 2 frames
- Rocas: cada 3 frames
- Muros: cada 3 frames

El estado compartido se protege con `threading.Lock()`. El agente nunca espera a los detectores — siempre lee el último estado disponible. Esta separación garantiza que la detección de colisiones (crítica para el reward de bananas) corre a máxima velocidad sin ser bloqueada por los detectores más lentos.

### 8.2 Estrategia de Detección por Tipo de Objeto

La mayoría de los detectores siguen un enfoque **híbrido HSV + Template Matching**:

- **Solo HSV:** Insuficiente para objetos que comparten colores con el fondo. Da demasiados falsos positivos.
- **Solo Template Matching:** Lento sobre el frame completo y sensible a variaciones de escala.
- **Híbrido:** HSV reduce el frame a un conjunto pequeño de blobs candidatos. Template matching corre solo sobre esos recortes, siendo 10–50x más rápido y más preciso.

| Detector | Estrategia | Razón |
|----------|-----------|-------|
| Kong | CSRT Tracker + HSV + Template multi-pose | CSRT para seguimiento fluido; HSV+Template para inicialización y recuperación |
| Bananas | HSV (S≥180, amarillo) | Color muy distintivo; S alto excluye follaje y estatuas |
| Agua | HSV (azul/celeste) | Color único en la escena |
| Barriles | HSV (marrón V alto) + Template | V alto distingue interior brillante del suelo oscuro |
| Rocas | Template matching (TM_CCOEFF_NORMED) a media resolución | Colores no separables del fondo con HSV |
| Muros madera | HSV (naranja S>150) + Template | S alto distingue de troncos (S~100-120) |
| Muros piedra | HSV (gris rosado) + Template | Rango estrecho de saturación |
| Game Over | Template matching sobre ROI central | Pantalla estática, muy confiable |

### 8.3 Detector de Kong — CSRT Tracker

El detector de Kong usa una arquitectura de tres capas:

1. **HSV + Template Matching:** Inicializa la detección. Busca blobs con color de piel de Kong en el ROI `(80, 60, 420, 510)` y verifica con template matching (9 poses: inicio, corriendo×2, saltando×2, paracaídas, dash, liana, guacamaya). Umbral de confianza: 0.65.

2. **CSRT Tracker:** Una vez inicializado con el bounding box de Kong, sigue a Kong frame a frame sin depender del color. Es robusto ante cambios de apariencia y oclusiones parciales. Se valida en cada frame que el bbox esté dentro del ROI, tenga tamaño razonable (20–250px) y posición horizontal válida (cx entre 0.05 y 0.45).

3. **Fallback:** Si CSRT falla, retorna la última posición conocida por máximo 10 frames mientras intenta reinicializar con HSV+Template.

### 8.4 Configuración de ROIs

```python
ROI_KONG      = (80, 60, 420, 510)    # franja izquierda, excluye HUD
ROI_BARRILES  = (160, 80, 900, 480)   # excluye zona de Kong
ROI_BANANAS   = (160, 60, 960, 510)   # ROI amplio para colisión
ROI_AGUA      = (0, 300, 960, 510)    # franja inferior
ROI_GAMEOVER  = (200, 100, 760, 400)  # zona central de pantalla
ROI_MUROS     = (200, 60, 960, 510)   # excluye zona de Kong
ROI_ROCAS     = (200, 60, 960, 510)   # excluye zona de Kong
```

### 8.5 Entrenamiento

```python
PPO_CONFIG = {
    "learning_rate": 2e-4,
    "n_steps":       2048,
    "batch_size":    128,
    "n_epochs":      10,
    "gamma":         0.99,
    "gae_lambda":    0.95,
    "clip_range":    0.2,
    "ent_coef":      0.01,
}

N_STACK = 4  # frame stacking
```

```python
env = DummyVecEnv([lambda: Monitor(BananaKongEnv(), RUTA_LOGS)])
env = VecFrameStack(env, n_stack=N_STACK)
modelo = PPO("MlpPolicy", env, **PPO_CONFIG)
```

```bash
python -m entrenamiento.entrenar              # desde cero
python -m entrenamiento.entrenar --continuar  # continuar desde checkpoint
```

---

## 9. Decisiones de Diseño y Alternativas Evaluadas

Esta sección documenta las principales decisiones de diseño tomadas durante el desarrollo, junto con las alternativas evaluadas y los criterios de selección.

### 9.1 Detección de Kong: Evolución del Enfoque

**Problema:** Kong comparte colores muy similares con muros de madera (H=13-42, S=100-170) y otros elementos del fondo, haciendo que HSV solo genere demasiados falsos positivos.

**Alternativas evaluadas:**

| Alternativa | Resultado | Razón de descarte |
|-------------|-----------|-------------------|
| HSV puro | ❌ Inestable | Falsos positivos con muros y barriles |
| HSV + Template Matching | ⚠️ Funcional pero titila | Sin memoria temporal entre frames |
| KCF Tracker | ❌ Inestable | Titila en cambios de pose (dash, paracaídas) |
| CSRT Tracker solo | ⚠️ Bueno pero se desvía | Sin mecanismo de corrección cuando pierde a Kong |
| MOG2 (Background Subtraction) | ❌ Descartado | El fondo animado (árboles, nubes) genera demasiado ruido |
| Franja HSV fija | ⚠️ Rápido pero impreciso | Kong en cx≈0.25 pero titila sin suavizado |
| YOLO / Deep Learning | ❌ No viable en CPU | AMD RX 9070 XT sin soporte ROCm en Windows |

**Decisión final:** CSRT + HSV + Template Matching en cascada. HSV+Template inicializa y recupera; CSRT sigue frame a frame con validación de bbox (ROI, tamaño, posición cx 0.05–0.45). Fallback de 10 frames si se pierde la detección.

### 9.2 Detección de Bananas Recogidas: Evolución del Reward

**Problema:** Contar bananas recogidas con precisión es crítico para el reward, pero varios enfoques fallaron.

| Alternativa | Resultado | Problema |
|-------------|-----------|---------|
| `bananas_ahora < bananas_prev` | ❌ | Falsos positivos cuando bananas salen del ROI |
| OCR del HUD (Tesseract) | ❌ | Demasiado lento para tiempo real |
| TTL/cooldown por posición | ❌ | El escenario se mueve, posición cambia |
| Delta de colisiones activas | ⚠️ | Pierde bananas en steps rápidos |
| **Pico de colisiones en hilo rápido** | ✅ | Robusto, corre a máxima velocidad |

**Decisión final:** Lógica del pico de colisiones (`_pico_colisiones`) corriendo en el hilo rápido del Perceptor. El pico acumula el máximo de colisiones simultáneas y contabiliza cuando baja a cero, independientemente de la velocidad del agente.

### 9.3 Arquitectura del Perceptor: Un Hilo vs. Dos Hilos

**Problema inicial:** Un solo hilo corría todos los detectores secuencialmente. El detector de Kong (CSRT) y los detectores lentos (muros con 9 escalas × 2 tipos) se bloqueaban mutuamente, causando que las colisiones de bananas se perdieran entre steps.

**Decisión:** Separar en hilo rápido (Kong + Bananas) y hilo lento (resto). El GIL de Python impide paralelismo real en CPU, pero la separación garantiza que la lógica crítica de colisiones no se bloquea por los detectores costosos.

**Limitación del GIL:** Se exploró `multiprocessing` para paralelismo real, pero el costo de serialización de frames entre procesos (cada frame BGR de 960×540 pesa ~1.5MB) resultó mayor que el beneficio. La arquitectura de dos hilos con estado compartido y `threading.Lock()` es el compromiso óptimo para este caso.

### 9.4 Vector de Observación: Posición Absoluta vs. Relativa

**Versión inicial (24 floats, posición absoluta):**
```
[0-1] kong_cx, kong_cy
[2-7] barril1_cx, barril1_cy, barril2_cx, barril2_cy, ...
```

**Problema:** La posición absoluta de un barril no le dice al agente si está cerca o lejos de Kong. cx=0.5 puede ser peligroso o seguro dependiendo de dónde esté Kong.

**Versión actual (13 floats, distancia relativa):**
```
[7] barril1_dx = clip(barril_cx - kong_cx + 0.5, 0, 1)
```
Un valor de 0.5 significa "en la misma posición que Kong"; >0.5 significa a la derecha; <0.5 a la izquierda. Esta representación es invariante a la posición de Kong en pantalla.

### 9.5 Frame Stacking

**Problema:** Con un solo estado, el agente no puede inferir velocidad ni dirección de los obstáculos. Un barril a cx=0.7 puede estar acercándose o alejándose — con un solo frame es imposible saberlo.

**Alternativas consideradas:**

| Alternativa | Evaluación |
|-------------|-----------|
| Estado único (sin stacking) | El agente ve posición pero no movimiento |
| Agregar velocidad al vector | Requiere calcular deltas manualmente y añade ruido |
| Frame stacking con imágenes (CnnPolicy) | Requiere cambiar toda la arquitectura a CNN; impracticable con detectores actuales |
| **VecFrameStack con vectores (MlpPolicy)** | ✅ Trivial de implementar; observation space pasa de 13 a 52 floats |

**Decisión final:** `VecFrameStack(env, n_stack=4)` de Stable-Baselines3. El agente recibe los últimos 4 estados concatenados (52 floats), permitiendo inferir velocidad sin cambios en los detectores ni en el entorno.

### 9.6 Hiperparámetros PPO: Evolución

| Experimento | Config | Resultado | Problema |
|-------------|--------|-----------|---------|
| PPO_6 | lr=1e-4, n_steps=1024, batch=64 | reward máx=12.33 | Alta varianza, oscilación |
| PPO_7 | lr=3e-4, n_steps=512, batch=64, game_over=-20 | reward todo negativo | -20 dominó sobre reward positivo |
| PPO_10 | lr=2e-4, n_steps=2048, batch=128, game_over=-10 | reward máx=5.08 ✅ | Tendencia positiva clara |

**Config final:** `lr=2e-4` (compromiso entre velocidad y estabilidad), `n_steps=2048` (más experiencia por update, menor oscilación), `batch_size=128` (proporcional a n_steps), `ent_coef=0.01` (incentivo de exploración), `game_over=-10` (penalización proporcional al reward acumulado típico).

### 9.7 Algoritmo de RL: PPO vs. Alternativas

| Algoritmo | Evaluación para Banana Kong |
|-----------|----------------------------|
| **PPO** ✅ | Robusto con acciones discretas, estable con observaciones ruidosas, soporte nativo en SB3 |
| DQN | Funciona con acciones discretas pero requiere replay buffer grande; menos estable con observaciones ruidosas |
| SAC | Diseñado para acciones continuas; no aplica directamente |
| QR-DQN | Variante distribucional de DQN; potencialmente mejor con rewards variables pero más complejo |
| A3C/A2C | Requiere múltiples entornos paralelos; impracticable con un solo BlueStacks |

---

## 10. Requerimientos

### Funcionales

| ID | Requerimiento |
|----|--------------|
| RF-01 | Capturar fotogramas del emulador a mínimo 15 FPS |
| RF-02 | Detectar posición de Kong, barriles, bananas, rocas, muros y agua en cada frame |
| RF-03 | Detectar fin de episodio (game over) con máximo 1s de latencia |
| RF-04 | Reiniciar el juego automáticamente al final de cada episodio |
| RF-05 | Exponer entorno compatible con OpenAI Gymnasium (step, reset, render) |
| RF-06 | Entrenar agente PPO y guardar checkpoints periódicos |
| RF-07 | Evaluar agente entrenado y comparar con política aleatoria |
| RF-08 | Ejecutar acciones mediante teclas configuradas en BlueStacks |
| RF-09 | Registrar métricas de entrenamiento en logs para TensorBoard |

### No Funcionales

| ID | Requerimiento |
|----|--------------|
| RNF-01 | Ciclo completo captura → decisión → acción < 100 ms (90% de los casos) |
| RNF-02 | Captura sostenida ≥ 15 FPS durante sesiones de > 30 minutos |
| RNF-03 | ROIs calibrados para resolución fija 960×540 de BlueStacks 5 |
| RNF-04 | Sistema ejecutable con un solo comando desde terminal |
| RNF-05 | Código organizado en módulos independientes con docstrings |
| RNF-06 | Compatible con Windows 10/11, Python 3.9+, BlueStacks 5 |

---

## 11. Criterios de Aceptación

| ID | Criterio | Métrica |
|----|----------|---------|
| CA-01 | Captura funcional | ≥ 15 FPS durante 30 min continuas |
| CA-02 | Latencia del pipeline | < 100 ms en el 90% de los ciclos |
| CA-03 | Reinicio automático | Exitoso en ≥ 95% de los episodios |
| CA-04 | Compatibilidad Gym | Entorno ejecuta episodios completos sin errores |
| CA-05 | Entrenamiento completado | ≥ 500.000 pasos sin fallos críticos |
| CA-06 | Superación de baseline | Puntaje promedio agente > baseline en 30 episodios (t-test p < 0.05) |
| CA-07 | Meta de puntaje | Promedio ≥ 5.000 puntos por episodio tras entrenamiento completo |
| CA-08 | Restricción de mundos | El agente no entra a mundos alternativos en ≥ 90% de episodios evaluados |
| CA-09 | Repositorio documentado | README completo, instrucciones reproducibles, código comentado |

---

## 12. Cronograma del Proyecto

> **Versión del cronograma:** v2 (entrega `crono1`) — 2026-03-11

| Semanas | Fase | Actividades principales | Prototipo |
|---------|------|------------------------|-----------|
| 1–2 | Selección del videojuego | Evaluación de candidatos, criterios de selección, decisión documentada, boceto del pipeline | P1: Documento de selección y justificación de Banana Kong |
| 3–4 | Configuración del entorno | Instalación BlueStacks 960×540, configuración Game Controls (W/D/S), instalación de dependencias Python, benchmark FPS | P2: Captura funcional a ≥ 15 FPS con reporte de latencia en consola |
| 4–7 | Módulo de percepción | Preparación de templates PNG con alpha; detectores de Kong (9 poses), bananas, agua, barriles, rocas (2 tipos), muros (madera y piedra), game over; integración en clase `Perceptor`; calibración de ROIs | P3: `perceptor.py` en modo demo con bounding boxes en tiempo real sobre todos los objetos |
| 5–7 | Entorno Gymnasium | Diseño del espacio de observación (13 floats × 4 frames) y acciones (Discrete(4)); implementación de `BananaKongEnv`; función de recompensa con detección de colisión; reinicio automático en 3 pasos; período de gracia de 10 steps | P4: Entorno ejecutando episodios completos sin errores + `evaluar.py` con baseline aleatorio |
| 7–8 | Integración del pipeline | Integración captura → percepción → decisión → acción; arquitectura de dos hilos en Perceptor; medición de latencia end-to-end; validación de ROIs en partidas reales | P5: Pipeline completo corriendo 5 episodios sin intervención manual, latencia p90 < 100 ms |
| 8–13 | Entrenamiento RL con PPO | Configuración de hiperparámetros; corridas de 50k pasos para validación; ajuste de hiperparámetros y rewards; corrida completa ≥ 500k pasos con frame stacking y checkpoints cada 10k | P6: Checkpoint a 100k pasos con curva de reward creciente<br>P7: Modelo final `banana_kong_ppo.zip` a ≥ 500k pasos |
| 13–14 | Evaluación | Evaluación formal agente PPO (30 episodios); evaluación baseline aleatorio (30 episodios); t-test de medias; verificación restricción de mundos alternativos | P8: Informe de evaluación con tabla de puntajes, t-test y distribución (boxplot) |
| 14–15 | Optimización y robustez | Ajuste fino según fallos identificados; re-entrenamiento parcial si hay regresión; pruebas de robustez; corrección de detectores con mayor tasa de error | P9: Agente refinado con puntaje promedio ≥ 5.000 puntos |
| 15–16 | Documentación y entrega | Limpieza del repositorio; actualización del README con resultados; reporte final; grabación de demo (≥ 3 episodios); presentación | P10: Entrega final con repositorio documentado, demo en video y todos los CA verificados |

### Diagrama de Gantt

```
Sem →   1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
─────────────────────────────────────────────────────────
Selección [P1]
        ████
Configuración [P2]
              ████
Percepción [P3]
                 ████████████
Entorno Gym [P4]
                    ████████
Integración [P5]
                          ████
Entrenamiento [P6→P7]
                              ████████████████
Evaluación [P8]
                                          ████
Optimización [P9]
                                              ████
Documentación [P10]
                                                  ████
─────────────────────────────────────────────────────────
Entrega crono1: semana 4 (branch crono1)
```

---

## 13. Diagramas

### Arquitectura del Sistema
![Arquitectura](diagramas/arquitectura_sistemas.png)

### Interacción entre Módulos
![Módulos](diagramas/iteraccion_modulos.png)

### Secuencia
![Secuencia](diagramas/secuencia.png)

---

## 14. Instalación y Uso

### Requisitos
 
- Python 3.9+
- BlueStacks 5
- GPU AMD o NVIDIA (opcional; el entrenamiento corre en CPU por limitaciones de soporte ROCm en Windows)
 
---
 
### 1. Clonar el repositorio
 
```bash
git clone https://github.com/KidmanC/KongBot-Agente-Aut-nomo-para-Banana-Kong-mediante-Aprendizaje-por-Refuerzo
```
 
---
 
### 2. Crear entorno virtual e instalar dependencias
 
```bash
python -m venv .venv
```
 
Activar el entorno virtual:
 
- **Windows:**
  ```bash
  .venv\Scripts\activate
  ```
- **Mac/Linux:**
  ```bash
  source .venv/bin/activate
  ```
 
Instalar dependencias:
 
```bash
pip install -r requirements.txt
```
 
---
 
### 3. Configurar BlueStacks
 
#### Resolución
 
La resolución debe ser exactamente **960×540**. Para configurarla:
 
1. Abre BlueStacks y anda a **Configuración → Display**
2. Resolución: `960 × 540`
3. DPI: `240`
4. Guarda y reiniciá BlueStacks
 
#### Desactivar anuncios
 
> **Importante:** Los anuncios de BlueStacks modifican el tamaño de la ventana de juego, lo que desplaza los ROIs de todos los detectores y causa fallos en la detección.
 
Para desactivarlos:
 
1. Abre BlueStacks → **Configuración → Preferencias**
2. Busca la opción **"Permitir que BlueStacks muestre anuncios"** (o similar)
3. **Desactivala**
4. Reiniciá BlueStacks
 
#### Controles
 
Dentro del juego, abre el **Game Controls** (ícono de teclado en la barra lateral de BlueStacks) y configura las siguientes teclas:
 
| Tecla | Acción en el juego |
|-------|-------------------|
| `W` | Saltar / Planear |
| `D` | Dash (impulso hacia adelante) |
| `S` | Bajar / Deslizarse |
 
---
 
### 4. Estructura del proyecto
 
```
Aprendizaje-por-refuerzo/
│
├── deteccion/
│   ├── templates/
│   │   ├── barril-bg.png
│   │   ├── kong_corriendo1-bg.png
│   │   ├── kong_corriendo3-bg.png
│   │   ├── kong_dash-bg.png
│   │   ├── kong_guacamaya-bg.png
│   │   ├── kong_inicio-bg.png
│   │   ├── kong_liana-bg.png
│   │   ├── kong_paracaidas-bg.png
│   │   ├── kong_saltando-bg.png
│   │   ├── kong_saltando2-bg.png
│   │   ├── muro_madera-bg.png
│   │   ├── muro_piedra-bg.png
│   │   ├── roca1-bg.png
│   │   ├── roca2-bg.png
│   │   ├── flecha.png
│   │   ├── play_again.png
│   │   └── revive_texto.png
│   ├── __init__.py
│   ├── detector_agua.py
│   ├── detector_bananas.py
│   ├── detector_barriles.py
│   ├── detector_gameover.py
│   ├── detector_kong.py
│   ├── detector_muros.py
│   └── detector_rocas.py
│
├── entorno/
│   ├── __init__.py
│   ├── entorno.py
│   └── perceptor.py
│
├── entrenamiento/
│   └── entrenar.py
│
├── controles/
│   ├── __init__.py
│   └── acciones.py
│
├── modelos/            ← generado automáticamente, en .gitignore
├── logs/               ← generado automáticamente, en .gitignore
├── requirements.txt
├── .gitignore
└── README.md
```
 
---
 
### 5. Probar detectores individualmente
 
```bash
python -m deteccion.detector_kong
python -m deteccion.detector_bananas
python -m deteccion.detector_barriles
python -m deteccion.detector_rocas
python -m deteccion.detector_muros
python -m deteccion.detector_agua
python -m deteccion.detector_gameover
python -m entorno.perceptor
```
 
---
 
### 6. Entrenar el agente
 
```bash
# Entrenamiento desde cero
python -m entrenamiento.entrenar
 
# Continuar entrenamiento previo
python -m entrenamiento.entrenar --continuar
```
 
---
 
### 7. Evaluar el agente
 
```bash
# Evaluar modelo guardado (30 episodios)
python -m entrenamiento.evaluar

# Evaluar agente + baseline y comparar estadísticamente
python -m entrenamiento.evaluar --ambos
```

---

### 8. Monitorear el entrenamiento
 
```bash
tensorboard --logdir logs/
```
 
Abrí `http://localhost:6006` en el navegador para ver las curvas de recompensa en tiempo real.

---

## 15. Referencias

1. V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, vol. 518, pp. 529–533, 2015.
2. OpenAI, "OpenAI Five," 2019. https://openai.com/five
3. O. Vinyals et al., "Grandmaster level in StarCraft II using multi-agent reinforcement learning," *Nature*, vol. 575, pp. 350–354, 2019.
4. J. Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.
5. V. Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning," *ICML*, 2016.
6. Y.-H. Yeh et al., "Automated Game Bot for Subway Surfers Using Computer Vision," *IEEE ICCE*, 2021.
7. G. Brockman et al., "OpenAI Gym," arXiv:1606.01540, 2016.
8. A. Raffin et al., "Stable-Baselines3: Reliable Reinforcement Learning Implementations," *JMLR*, vol. 22, 2021.
9. G. Bradski, "The OpenCV Library," *Dr. Dobb's Journal*, 2000.
10. ScreenInfo, "mss: An ultra-fast cross-platform multiple screenshots module," https://python-mss.readthedocs.io