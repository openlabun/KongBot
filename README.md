# Bot Autónomo para Banana Kong - Aprendizaje por Refuerzo
 
**Universidad del Norte - Facultad de Ingeniería de Sistemas**  
**Proyecto Final - Grupo 2 - Aprendizaje por Refuerzo**  
**Kidman Cabana, Santiago Romero - Barranquilla, Colombia - 2026**
 
---
 
## Tabla de Contenidos
 
1. [Introducción](#1-introducción)
2. [Planteamiento del Problema](#2-planteamiento-del-problema)
3. [Objetivos](#3-objetivos)
4. [Estado del Arte](#4-estado-del-arte)
5. [Diseño y Arquitectura](#5-diseño-y-arquitectura)
6. [Implementación](#6-implementación)
7. [Plan de Pruebas](#7-plan-de-pruebas)
8. [Referencias](#8-referencias)
 
---
 
## 1. Introducción
 
Banana Kong es un videojuego de plataformas y carrera continua (*endless runner*) desarrollado por FDG Entertainment, disponible para plataformas móviles Android e iOS. El juego presenta a un gorila que debe desplazarse por una selva tropical recolectando plátanos, esquivando obstáculos y utilizando animales de apoyo para avanzar. Su espacio de acciones reducido (salto y planeo, dash y agacharse) lo convierte en un candidato adecuado para el entrenamiento de un agente basado en aprendizaje por refuerzo, dado que las decisiones son discretas y el entorno es visualmente consistente dentro de un mismo mundo.
 
Este proyecto propone construir un agente autónomo que perciba el juego exclusivamente a través de la pantalla y ejecute acciones simulando entradas de teclado, sin acceso a la memoria del juego ni modificación del APK. El módulo de percepción utiliza visión por computador con OpenCV, implementando detectores especializados por tipo de objeto mediante estrategias híbridas de segmentación HSV y template matching. El módulo de decisión utiliza PPO (Proximal Policy Optimization) implementado con Stable-Baselines3, un algoritmo de gradiente de política con buena estabilidad de entrenamiento para espacios de acción discretos pequeños.
 
Durante el semestre, el proyecto ha avanzado desde la conceptualización inicial hasta contar con un pipeline funcional completo: captura de pantalla, detección de objetos, entorno Gymnasium, entrenamiento con PPO y reinicio automático de episodios. Los primeros experimentos de entrenamiento con 250.000 pasos muestran una curva de aprendizaje positiva con recompensa promedio por episodio creciendo de 4 a 9, lo que valida la viabilidad del enfoque aunque evidencia que alcanzar el rendimiento objetivo requerirá sesiones de entrenamiento considerablemente más largas.
 
---
 
## 2. Planteamiento del Problema
 
### 2.1 Descripción del Problema
 
Los videojuegos comerciales son sistemas de caja negra: no exponen su estado interno mediante APIs públicas. La única información disponible para un agente externo es la imagen renderizada en pantalla. Esto genera una brecha técnica concreta: integrar captura visual, percepción computacional y ejecución de acciones en un pipeline coherente que opere en tiempo real es un problema de ingeniería no trivial, especialmente con hardware académico limitado. En el contexto específico de Banana Kong, el desafío se amplifica porque los objetos relevantes (Kong, barriles, bananas, muros, agua) tienen variaciones visuales significativas según el fondo, la iluminación del nivel y las animaciones en curso.
 
**Pregunta central:** ¿Es posible diseñar e implementar, bajo restricciones académicas de tiempo y hardware, un agente autónomo basado en aprendizaje por refuerzo que aprenda a jugar Banana Kong en un emulador Android para PC, utilizando únicamente información visual y simulación de entradas de teclado, alcanzando una recompensa promedio por episodio (`ep_rew_mean`) entre 10 y 15?
 
Un desafío adicional identificado durante el desarrollo es que alcanzar recompensas más ambiciosas (ep_rew_mean > 20) de forma consistente probablemente requeriría entrenamientos superiores al millón de pasos, un volumen que excede las restricciones de tiempo y hardware del proyecto académico. Los primeros experimentos con 250.000 pasos muestran que el modelo alcanzó ep_rew_mean = 10.46 en su mejor checkpoint (160k steps), con un máximo de 72.15 en episodios individuales, lo que valida que la meta de 10–15 es alcanzable y que el potencial del agente supera ampliamente su rendimiento promedio actual. Este hallazgo motiva investigación futura sobre técnicas que aceleren la convergencia, como curriculum learning, recompensas más densas o representaciones de estado más ricas.
 
### 2.2 Restricciones y Supuestos
 
**Restricciones técnicas:**
 
- **Sin acceso interno al juego:** El sistema trata Banana Kong como caja negra. No se lee ni modifica la memoria del proceso, ni se inyecta código en el emulador.
- **Captura exclusivamente visual:** Toda la información del estado proviene de capturas de pantalla con `mss`. No se usa audio, tráfico de red ni otras fuentes.
- **Acciones mediante teclado simulado:** Las interacciones se ejecutan a través de `pyautogui` simulando las teclas configuradas en BlueStacks Game Controls. No se usa ADB por problemas de latencia y conflicto con eventos táctiles.
- **Resolución fija 960×540:** Todos los detectores están calibrados para esta resolución. Cambiarla requiere recalibrar ROIs y umbrales.
- **Latencia objetivo:** El ciclo completo captura → percepción → decisión → acción debe completarse en menos de 100 ms.
- **Hardware de consumo:** Desarrollo en equipos con GPU NVIDIA de gama media. Sin clústeres ni instancias cloud.
 
**Restricciones del entorno de juego:**
 
- **Meta de rendimiento:** El agente debe alcanzar una recompensa promedio por episodio (`ep_rew_mean`) **entre 10 y 15** como criterio de éxito del proyecto académico. Se adopta esta métrica en lugar del puntaje del juego porque el puntaje está influenciado por multiplicadores y power-ups que el agente no controla directamente, mientras que la recompensa es completamente observable, reproducible y alineada con lo que el agente realmente optimiza.
- **Restricción de mundos alternativos:** El agente no debe entrar a mundos alternativos accesibles mediante cuevas (mina), zonas de agua (mundo submarino) o cohetes (zona aérea). Estos mundos cambian radicalmente la paleta de colores, la geometría de obstáculos y la estructura del HUD, invalidando todos los detectores calibrados para el mundo principal. La restricción se implementa mediante detección de las entradas a mundos alternativos y penalización de las acciones que llevan a ellas.
- **Mundo único:** El agente opera exclusivamente en el mundo de la selva (mundo inicial). No se contemplan otros biomas.
- **Configuración gráfica fija:** La ventana del emulador permanece en primer plano y visible durante toda la ejecución.
 
**Restricciones normativas:** El proyecto es estrictamente académico y no comercial. No se redistribuye el APK del juego. El bot opera exclusivamente en modalidad de un jugador (offline).
 
**Supuestos:** Los elementos clave del juego son visualmente distinguibles con las técnicas implementadas en condiciones normales del mundo selva. Los colores, formas y posiciones de los elementos del HUD y del entorno son consistentes entre partidas dentro del mismo mundo. El juego no recibirá actualizaciones que cambien significativamente su interfaz visual durante el semestre.
 
### 2.3 Alcance
 
**Incluido:**
 
- Pipeline completo: captura → percepción → decisión → acción
- Detectores especializados para: Kong, barriles, bananas, agua, muros (madera y piedra), rocas, game over, mina y tubo amarillo
- Entorno compatible con la interfaz OpenAI Gymnasium
- Entrenamiento con PPO usando Stable-Baselines3
- Reinicio automático de episodios
- Evaluación frente a política aleatoria de referencia (*baseline*)
- Documentación técnica completa
 
**Excluido:**
 
- Soporte para múltiples juegos o biomas distintos al mundo selva
- Detección de objetos interactivos opcionales (lianas, trampolines, guacamaya)
- Interfaz gráfica de usuario (GUI): la ejecución es por línea de comandos
- Modificación del APK, archivos del emulador o código del juego
- Generalización a múltiples resoluciones o versiones del juego
 
---
 
## 3. Objetivos
 
### General
 
Diseñar e implementar un agente autónomo basado en aprendizaje por refuerzo profundo que aprenda a jugar Banana Kong en un emulador Android para PC, utilizando exclusivamente información visual de la pantalla y simulación de teclado, alcanzando al final del semestre una recompensa promedio por episodio (`ep_rew_mean`) entre 10 y 15, medida sobre los últimos 100 episodios del entrenamiento, superior al de una política aleatoria de referencia.
 
### Específicos
 
1. Implementar un módulo de captura capaz de obtener fotogramas del emulador a mínimo 15 FPS con latencia individual menor a 50 ms.
2. Desarrollar detectores de visión por computador para cada tipo de objeto relevante del juego, con precisión superior al 85% en condiciones normales del mundo selva.
3. Diseñar y formalizar el entorno Gymnasium con espacio de estados, acciones y función de recompensa.
4. Entrenar al menos un agente PPO durante un mínimo de 500.000 pasos, documentando curvas de aprendizaje.
5. Evaluar el agente frente a una política aleatoria, demostrando mejora estadísticamente significativa en puntaje promedio por episodio en al menos 30 episodios.
6. Documentar el sistema completo en el repositorio con READMEs, diagramas y resultados de experimentos.
 
Los objetivos están ordenados por dependencia técnica: sin captura funcional no hay percepción, sin percepción no hay entorno, y sin entorno no hay entrenamiento. Esta cadena implica que los primeros objetivos son bloqueantes para los siguientes, lo que condicionó el plan de trabajo durante el semestre. La experiencia del proyecto también revela un desafío no anticipado: 500.000 pasos son suficientes para demostrar aprendizaje y acercarse a la meta de ep_rew_mean ∈ [10, 15], pero probablemente insuficientes para superarla de forma consistente con el hardware disponible. Esto abre una línea de trabajo futuro centrada en acelerar la convergencia mediante mejoras en el espacio de observación, la función de recompensa o el uso de técnicas como curriculum learning.
 
---
 
## 4. Estado del Arte
 
### 4.1 Aprendizaje por Refuerzo en Videojuegos
 
El trabajo de Mnih et al. (2015) con DQN demostró que una red neuronal puede aprender políticas de juego competitivas directamente desde píxeles en juegos de Atari, abriendo el campo del aprendizaje por refuerzo profundo aplicado a videojuegos. Esta aproximación requiere grandes volúmenes de experiencia (en el orden de millones de pasos) pero no necesita ningún conocimiento previo sobre la estructura del juego. En nuestro proyecto, procesar píxeles directamente mediante una política CNN (`CnnPolicy` en Stable-Baselines3) es técnicamente posible pero se descartó por las restricciones de hardware: un paso de entrenamiento con entrada visual es aproximadamente 10 veces más costoso que con un vector de características.
 
Schulman et al. (2017) propusieron PPO, algoritmo de gradiente de política con mayor estabilidad de entrenamiento que sus predecesores (TRPO, A3C). PPO limita el cambio en la política en cada actualización mediante una función de pérdida con clip, lo que reduce el riesgo de colapso de política que se observa con learning rates altos. Esta propiedad es especialmente relevante en nuestro caso, donde los experimentos iniciales con `learning_rate=3e-4` mostraron exactamente ese problema: la recompensa alcanzó un pico en ~30.000 pasos y luego colapsó sostenidamente hasta los 300.000 pasos. Reducir la tasa a `1e-4` estabilizó el aprendizaje en sesiones posteriores.
 
OpenAI Five (2019) y AlphaStar de DeepMind (2019) demostraron que los agentes de RL pueden alcanzar nivel de experto humano en juegos complejos, pero con recursos computacionales que están completamente fuera del alcance académico: miles de CPUs y GPUs operando durante semanas. Estos trabajos son relevantes no por ser replicables en nuestro contexto, sino porque establecen el límite superior de lo alcanzable con RL en videojuegos y muestran que la brecha entre un agente funcional y uno de nivel experto es fundamentalmente una cuestión de escala computacional.
 
### 4.2 Bots para Endless Runners
 
Proyectos como el bot para Subway Surfers de Yeh et al. (2021) usaron visión por computador con OpenCV para detectar obstáculos mediante segmentación por color, sin aprendizaje automático. Lograron tiempos de supervivencia superiores al jugador promedio pero con robustez limitada a condiciones de color constante. La principal debilidad de este enfoque es que las reglas de evasión deben codificarse manualmente: el sistema no aprende a esquivar, simplemente ejecuta heurísticas predefinidas. Nuestro enfoque híbrido, percepción clásica combinada con RL, busca mantener la confiabilidad de la detección visual mientras permite que el comportamiento de evasión emerja del aprendizaje.
 
### 4.3 Vacíos que Abordamos
 
La literatura académica muestra escasez de implementaciones reproducibles de agentes RL visuales para juegos móviles en emulador. La mayoría de los trabajos publicados operan sobre juegos de Atari o entornos sintéticos, donde existe acceso directo al estado interno del juego. Nuestro trabajo aborda el caso más restrictivo y realista de un juego comercial de caja negra, donde toda la información debe extraerse visualmente. Adicionalmente, la combinación de detección HSV y template matching como preprocesamiento para reducir el espacio de búsqueda antes del template matching es una contribución práctica: en lugar de buscar templates sobre el frame completo (costoso), se usan filtros de color para identificar candidatos y luego se aplica template matching solo sobre esos recortes, logrando una aceleración de 10-50x en la detección.
 
---
 
## 5. Diseño y Arquitectura
 
### 5.1 Evaluación de Alternativas
 
Para el módulo de percepción se evaluaron tres enfoques principales. El primero fue procesar píxeles directamente con una red CNN (`CnnPolicy`), que tiene la ventaja de no requerir ingeniería manual de características pero demanda 10x más pasos de entrenamiento y es más sensible al sobreajuste con hardware limitado. El segundo fue usar solo segmentación por color HSV, que es rápido y simple pero genera demasiados falsos positivos para objetos que comparten colores con el fondo (muros de madera vs. troncos de árbol, ambos marrones). El tercero, adoptado en este proyecto, fue el enfoque híbrido HSV + template matching: HSV reduce el frame a un conjunto pequeño de blobs candidatos, y template matching corre solo sobre esos recortes, siendo 10-50x más rápido que buscar sobre el frame completo y más preciso que HSV solo.
 
Para el algoritmo de RL se consideraron PPO, DQN y A3C. DQN es adecuado para espacios de acción discretos pequeños pero menos estable en entornos con recompensas dispersas. A3C requiere múltiples trabajadores paralelos, lo que complica la implementación cuando hay dependencia de una sola instancia del emulador. PPO fue seleccionado por su estabilidad documentada, disponibilidad en Stable-Baselines3 y buen desempeño reportado en la literatura para tareas similares con pocas acciones discretas.
 
Para la simulación de controles se evaluaron ADB (Android Debug Bridge), `pyautogui` con gestos táctiles y `pyautogui` con teclas configuradas en BlueStacks Game Controls. ADB presentó problemas de latencia y conflicto con eventos táctiles nativos. Los gestos táctiles mediante drag causaban que BlueStacks interpretara el inicio del gesto como un tap, haciendo saltar a Kong antes del dash. La solución definitiva fue configurar el dash directamente como una tecla (`D`) en Game Controls, eliminando por completo la necesidad de simular gestos.
 
### 5.2 Arquitectura
 
La arquitectura del sistema sigue un pipeline secuencial con dos hilos de detección paralelos para maximizar el FPS:
 
```
BlueStacks (960x540)
        │
        ▼
┌───────────────┐
│    mss        │  Captura de pantalla
└───────┬───────┘
        │ frame BGR
        ▼
┌─────────────────────────────────┐
│  Perceptor (dos hilos)          │
│  ├── Hilo rápido: Kong, Bananas │
│  └── Hilo lento: Obstáculos,    │
│       GameOver, Agua, Mina, Tubo│
└───────┬─────────────────────────┘
        │ estado (dict con 15+ claves)
        ▼
┌───────────────┐
│ BananaKongEnv │  Entorno Gymnasium
│               │  obs vector (15 floats)
│               │  reward, terminated
└───────┬───────┘
        │ obs
        ▼
┌───────────────┐
│  PPO Agent    │  Stable-Baselines3 MlpPolicy
└───────┬───────┘
        │ acción (0-3)
        ▼
┌───────────────┐
│ ModuloAcciones│  pyautogui: W / D / S
└───────────────┘
```
 
La separación en hilo rápido (Kong y bananas, cada frame) y hilo lento (obstáculos y detectores costosos, cada 2-5 frames) permite mantener la detección de Kong fluida sin sacrificar la detección de obstáculos. El agente siempre lee el último estado disponible sin bloquearse, garantizando que el DELAY_ACCION de 50ms no se acumule con el tiempo de detección.
 
---
 
## 6. Implementación
 
### 6.1 Stack Tecnológico
 
| Tecnología | Rol | Justificación |
|-----------|-----|--------------|
| Python 3.9+ | Lenguaje principal | Ecosistema RL maduro |
| OpenCV | Visión por computador | Template matching, HSV, contornos |
| mss | Captura de pantalla | ~60 FPS, menor latencia que PIL |
| Stable-Baselines3 | Algoritmo PPO | Implementación robusta y documentada |
| Gymnasium | Interfaz entorno | Estándar de la industria para RL |
| pyautogui | Simulación de teclado | Compatible con BlueStacks en Windows |
| BlueStacks 5 | Emulador Android | Game Controls con teclas personalizables |
| TensorBoard | Monitoreo | Curvas de aprendizaje en tiempo real |
 
El stack fue seleccionado priorizando madurez y disponibilidad de documentación. Todas las dependencias son de código abierto y están disponibles en PyPI, lo que facilita la reproducibilidad del proyecto.
 
### 6.2 Componentes
 
**Detectores de visión por computador:** Cada tipo de objeto tiene su propio módulo detector en `deteccion/`. Todos siguen la estrategia híbrida HSV + template matching, excepto el game over (template matching puro sobre ROI pequeño) y el agua (HSV puro, color muy distintivo). Los templates se entregan como PNG con canal alpha real para usar la máscara en `cv2.matchTemplate(..., mask=alpha)`, lo que hace el matching más robusto ante variaciones de fondo. Se usa `TM_CCOEFF_NORMED` por ser más intuitivo (máximo = mejor coincidencia) y más robusto ante variaciones de brillo que `TM_SQDIFF_NORMED`.
 
**Perceptor:** Módulo central que coordina todos los detectores en dos hilos daemon. El hilo rápido captura y procesa Kong y bananas en cada iteración. El hilo lento procesa obstáculos pesados (barriles, rocas, muros, agua, mina, tubo) con cadencia configurable (cada 2-5 frames). El estado se protege con `threading.Lock()` y el agente siempre lee el último estado disponible sin bloquearse. Adicionalmente corre un tercer hilo de display que muestra bounding boxes en tiempo real a escala 50% para monitoreo visual durante el entrenamiento.
 
**Entorno Gymnasium (`BananaKongEnv`):** Implementa la interfaz estándar `step()` / `reset()`. El vector de observación contiene 15 floats normalizados en [0,1]: posición de Kong, distancias relativas a las 2 bananas más cercanas, flag de agua, distancia relativa al barril, roca y muro más cercanos, y distancias al tubo y mina si están visibles. Las recompensas son +0.02 por step de supervivencia, +1.0 por banana recogida y -10.0 por game over. El reinicio automático usa template matching para detectar la pantalla de "Revive" y "Play Again", con fallback a presionar W si no se encuentran.
 
**Módulo de acciones:** Implementa las 4 acciones discretas mediante `pyautogui`. La acción PLANEAR usa `keyDown`/`keyUp` para que el agente controle la duración del planeo implícitamente eligiendo la acción en steps consecutivos. Las teclas W, D, S están configuradas en BlueStacks Game Controls, eliminando la necesidad de simular gestos táctiles.
 
### 6.3 Integraciones
 
La única integración externa es BlueStacks como emulador Android. El sistema se comunica con él exclusivamente a través de dos canales: captura de pantalla con `mss` (lectura) y simulación de teclado con `pyautogui` (escritura). No hay llamadas a APIs externas ni bases de datos. Los modelos entrenados se guardan como archivos `.zip` de Stable-Baselines3 y los logs de entrenamiento en formato TensorBoard en la carpeta `logs/`. Los checkpoints periódicos permiten retomar el entrenamiento desde cualquier punto.
 
---
 
## 7. Plan de Pruebas
 
### 7.1 Pruebas por Componentes
 
Cada detector se puede ejecutar de forma independiente mediante `python -m deteccion.detector_<nombre>`, que abre una ventana con el frame anotado en tiempo real. Los criterios de éxito por detector son: precisión superior al 85% (menos del 15% de falsos positivos o falsos negativos en condiciones normales del mundo selva), estabilidad en sesiones continuas de 5 minutos sin errores no controlados, y latencia de detección inferior a 20ms por frame para los detectores del hilo rápido. Durante el desarrollo se usaron sesiones de calibración visual para ajustar umbrales HSV, área mínima de blobs y umbral de confianza de template matching, guardando prints de debug con los valores reales de cada detección para análisis sistemático.
 
El perceptor completo se prueba con `python -m entorno.perceptor`, que activa todos los detectores simultáneamente y muestra la ventana Debug con todas las bounding boxes superpuestas. Esta prueba verifica la integración del hilo rápido y lento, la ausencia de race conditions y el FPS resultante del sistema completo. Un FPS sostenido por encima de 10 en la ventana Debug indica que el sistema puede operar en tiempo real durante el entrenamiento.
 
El entorno Gymnasium se valida con `gymnasium.utils.env_checker.check_env(env)`, que verifica que la implementación cumple la interfaz estándar: dimensiones correctas del espacio de observación y acción, tipos de dato correctos, ausencia de valores NaN o infinitos en las observaciones, y comportamiento correcto de `reset()` y `step()`. Este criterio de aceptación (CA-04) es necesario para garantizar compatibilidad con Stable-Baselines3.
 
### 7.2 Pruebas de Integración
 
La prueba de integración principal es el entrenamiento completo: `python -m entrenamiento.entrenar`. Esta prueba verifica que todos los componentes interactúan correctamente en el flujo completo: el Perceptor alimenta el entorno, el entorno alimenta el agente PPO, el agente ejecuta acciones en BlueStacks y el entorno detecta correctamente el game over y reinicia el episodio. Los criterios de éxito son: el entrenamiento corre sin errores críticos durante al menos 1 hora, los checkpoints se guardan correctamente cada 10.000 pasos, y la curva de recompensa muestra tendencia positiva en las primeras 50.000 iteraciones.
 
El reinicio automático de episodios se evalúa midiendo la tasa de éxito del flujo completo revive → flecha → play again. Se considera exitoso si al menos el 95% de los episodios reinician correctamente sin intervención manual. Los casos de fallo, principalmente cuando las pantallas de revive no aparecen por entrar a mundos alternativos, se manejan con fallback de tecla W y timeout configurable.
 
### 7.3 Pruebas de Evaluación del Agente
 
La evaluación formal del agente se realiza comparándolo contra una política aleatoria de referencia (*baseline*) en al menos 30 episodios cada uno. Las métricas registradas son: recompensa total por episodio, número de bananas recogidas, duración del episodio en steps y puntaje del juego. La hipótesis nula es que el agente no supera al baseline, y se rechaza si el t-test de una cola da p < 0.05. Este criterio estadístico (CA-06) garantiza que la mejora observada no se debe al azar.
 
La evaluación de la restricción de mundos alternativos (CA-08) se mide como la proporción de episodios en los que el agente no entra a la mina ni al tubo amarillo en un conjunto de 30 episodios evaluados. El criterio de aceptación es que esto ocurra en al menos el 90% de los episodios. Esta métrica es difícil de medir automáticamente dado que requeriría detectar visualmente el cambio de bioma; en su lugar, se monitorea manualmente durante las sesiones de evaluación y se registra el número de veces que aparece el mensaje de mundo alternativo en los logs de consola.
 
---
 
## 8. Referencias
 
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
 