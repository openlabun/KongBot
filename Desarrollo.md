# Manual de desarrollo

## 1. Propósito del documento

Este documento tiene como objetivo servir de guía técnica para comprender, mantener, extender y dar continuidad al desarrollo del proyecto. Está dirigido a futuros equipos de trabajo que necesiten familiarizarse rápidamente con la estructura del repositorio, la organización de la solución, los scripts, las variables de configuración y el flujo de trabajo del sistema.

## 2. Descripción general del proyecto desde la perspectiva de desarrollo

El proyecto es un pipeline de aprendizaje por refuerzo visual que opera sobre un juego comercial de caja negra (Banana Kong) ejecutándose en un emulador Android (BlueStacks). La solución se implementa completamente en Python como un proceso monolítico local, sin arquitectura cliente-servidor ni contenedores.

El pipeline sigue un flujo secuencial: captura de pantalla → percepción (detección de objetos) → decisión (agente PPO) → acción (simulación de teclado). La percepción es el componente más complejo, con 9 detectores especializados coordinados en 2 hilos daemon con cadencias configurables.

### 2.1 Tecnologías principales

- **Lenguaje:** Python 3.9+
- **Visión por computador:** OpenCV (HSV, template matching, contornos, CSRT tracker, morfología)
- **Captura:** mss (Windows GDI, ~60 FPS)
- **RL:** Stable-Baselines3 (PPO, MlpPolicy, VecFrameStack)
- **Entorno:** Gymnasium (interfaz step/reset)
- **Acciones:** pyautogui (keyDown/Up)
- **Utilidades:** numpy, pygetwindow, threading
- **Monitoreo:** TensorBoard

### 2.2 Componentes principales

- **9 Detectores:** Kong, Bananas, Barriles, Agua, Rocas, Muros (madera+piedra), Mina, Tubo, Game Over. Cada uno en su propio módulo con método `probar()` para testing individual.
- **Perceptor:** Coordina detectores en 2 hilos daemon (rápido: Kong+Bananas, lento: obstáculos). Publica estado compartido protegido por `threading.Lock()`.
- **BananaKongEnv:** Entorno Gymnasium que convierte el estado del Perceptor en vector de 23 floats, calcula recompensas y detecta terminación.
- **Agente PPO:** Modelo MlpPolicy con VecFrameStack(4), entrenado con Stable-Baselines3.
- **ModuloAcciones:** Traduce acciones discretas (0-3) a teclas W/D/S.

## 3. Estructura del repositorio

### 3.1 Árbol general del repositorio

```
/
├── deteccion/
│   ├── templates/
│   ├── __init__.py
│   ├── detector_agua.py
│   ├── detector_bananas.py
│   ├── detector_barriles.py
│   ├── detector_gameover.py
│   ├── detector_kong.py
│   ├── detector_mina.py
│   ├── detector_muros.py
│   ├── detector_rocas.py
│   └── detector_tubo.py
├── entorno/
│   ├── __init__.py
│   ├── entorno.py
│   ├── perceptor.py
│   └── reward_bananas.py
├── entrenamiento/
│   ├── entrenar.py
│   └── evaluar.py
├── controles/
│   ├── __init__.py
│   └── acciones.py
├── diagramas/
│   ├── arquitectura_sistemas.txt
│   ├── arquitectura_sistemas.png
│   ├── iteraccion_modulos.txt
│   ├── iteraccion_modulos.png
│   └── secuencia.txt
│   └── secuencia.png
├── modelos/                    ← generado automáticamente (.gitignore)
├── logs/                       ← generado automáticamente (.gitignore)
├── .venv/                      ← generado automáticamente (.gitignore)
├── .run_id_actual              ← ID del entrenamiento activo
├── .gitignore
├── requirements.txt
├── com.fdgentertainment.bananakong.cfg    ← configuración de controles BlueStacks
├── com.fdgentertainment.bananakong_*.apk  ← APK del juego
├── README.md
├── Informe.md
├── Instalacion.md
└── Desarrollo.md
```

### 3.2 Descripción de directorios y archivos relevantes

- **deteccion/**: 9 módulos de detección de objetos, cada uno con su clase `DetectorX` y método `probar()` para testing. La carpeta `templates/` contiene los PNG con canal alpha usados por template matching.
- **entorno/**: `perceptor.py` coordina los detectores en hilos daemon. `entorno.py` implementa `BananaKongEnv` (Gymnasium). `reward_bananas.py` contiene lógica alternativa de recompensa.
- **entrenamiento/**: `entrenar.py` es el script principal de entrenamiento con soporte para `--continuar` y `--add-steps`. `evaluar.py` para evaluación del modelo entrenado.
- **controles/**: `acciones.py` traduce acciones (0-3) a pyautogui.keyDown/Up.
- **diagramas/**: Archivos PlantUML (.txt) y sus renders PNG de arquitectura, interacción y secuencia.
- **modelos/**: Checkpoints de PPO (.zip), generado durante entrenamiento.
- **logs/**: Logs de TensorBoard, generados durante entrenamiento.

## 4. Organización de la solución a nivel de código

### 4.1 Organización por módulos o capas

El proyecto está organizado por capas de pipeline:

1. **Capa de captura** (`mss`): única responsabilidad, obtener frames de BlueStacks
2. **Capa de percepción** (`deteccion/` + `perceptor.py`):
   - Hilo rápido: Kong (HSV + Template + CSRT Tracker + anti-drift) + Bananas (HSV + solidez + Kong masking)
   - Hilo lento: Barriles, Rocas, Muros, Agua, Mina, Tubo, Game Over (cadencias configurables)
3. **Capa de entorno** (`entorno.py`): estado → observación (23 floats), recompensa, terminación
4. **Capa de decisión** (Stable-Baselines3 PPO): inferencia de acción desde observación
5. **Capa de acción** (`acciones.py`): acción → teclas W/D/S

### 4.2 Relación entre componentes del sistema y código fuente

| Componente | Archivo | Clase/Función |
|---|---|---|
| Detector Kong | `deteccion/detector_kong.py` | `DetectorKong.detectar_kong()` |
| Detector Bananas | `deteccion/detector_bananas.py` | `DetectorBananas.detectar_bananas()` |
| Detector Barriles | `deteccion/detector_barriles.py` | `DetectorBarriles.detectar_barriles()` |
| Detector Agua | `deteccion/detector_agua.py` | `DetectorAgua.detectar_agua()` |
| Detector Rocas | `deteccion/detector_rocas.py` | `DetectorRocas.detectar_rocas()` |
| Detector Muros | `deteccion/detector_muros.py` | `DetectorMuros.detectar_muros()` |
| Detector Mina | `deteccion/detector_mina.py` | `DetectorMina.detectar_mina()` |
| Detector Tubo | `deteccion/detector_tubo.py` | `DetectorTubo.detectar_tubo()` |
| Detector Game Over | `deteccion/detector_gameover.py` | `DetectorGameOver.detectar_gameover()` |
| Perceptor | `entorno/perceptor.py` | `Perceptor` (hilo rápido + hilo lento) |
| Entorno | `entorno/entorno.py` | `BananaKongEnv` (gym.Env) |
| Acciones | `controles/acciones.py` | `ModuloAcciones.ejecutar()` |
| Entrenamiento | `entrenamiento/entrenar.py` | `main()` con PPO |

## 5. Contenedores

No se utilizan contenedores en este proyecto. El sistema requiere acceso directo a la ventana de BlueStacks en Windows y simulación de teclado, lo que no es compatible con Docker.

## 6. Scripts y automatizaciones

### 6.1 Scripts principales

| Comando | Descripción |
|---|---|
| `python -m deteccion.detector_kong` | Probar detector de Kong en tiempo real |
| `python -m deteccion.detector_bananas` | Probar detector de Bananas |
| `python -m deteccion.detector_barriles` | Probar detector de Barriles |
| `python -m deteccion.detector_rocas` | Probar detector de Rocas |
| `python -m deteccion.detector_muros` | Probar detector de Muros |
| `python -m deteccion.detector_agua` | Probar detector de Agua |
| `python -m deteccion.detector_mina` | Probar detector de Mina |
| `python -m deteccion.detector_tubo` | Probar detector de Tubo |
| `python -m deteccion.detector_gameover` | Probar detector de Game Over |
| `python -m entorno.perceptor` | Probar todos los detectores juntos con ventana Debug |
| `python -m entrenamiento.entrenar` | Entrenamiento nuevo desde cero |
| `python -m entrenamiento.entrenar --continuar` | Continuar entrenamiento previo |
| `python -m entrenamiento.entrenar --continuar --add-steps 100000` | Añadir 100K steps al entrenamiento actual |
| `tensorboard --logdir logs/` | Abrir TensorBoard para monitorear métricas |

### 6.2 Ubicación de scripts auxiliares

No hay carpeta `scripts/` dedicada. Los scripts de prueba son los métodos `probar()` dentro de cada módulo de detector.

### 6.3 Consideraciones para su uso

- Todos los scripts de detección requieren BlueStacks abierto con el juego corriendo
- Los detectores abren ventanas OpenCV; presionar `q` para cerrar
- El entrenamiento requiere que BlueStacks esté en primer plano y visible
- Los logs de entrenamiento se guardan en `logs/run_<id>/tensorboard/`

## 7. Variables de entorno

### 7.1 Variables requeridas

No se requieren variables de entorno.

### 7.2 Variables por ambiente

No aplica.

### 7.3 Archivos de configuración

| Archivo | Propósito |
|---|---|
| `.run_id_actual` | Guarda el ID del entrenamiento activo para continuar con `--continuar` |
| `com.fdgentertainment.bananakong.cfg` | Configuración exportada de controles de BlueStacks |
| `requirements.txt` | Dependencias de Python |

### 7.4 Manejo de secretos

No hay secretos ni credenciales en el proyecto.

## 8. Flujo de trabajo de desarrollo

### 8.1 Preparación del entorno

1. Clonar el repositorio
2. Crear y activar entorno virtual: `python -m venv .venv` + `.venv\Scripts\activate`
3. Instalar dependencias: `pip install -r requirements.txt`
4. Configurar BlueStacks (resolución 960×540, DPI 240, desactivar anuncios, importar .cfg)
5. Verificar detectores: `python -m entorno.perceptor`

### 8.2 Desarrollo de nuevas funcionalidades

Para agregar un nuevo detector:

1. Crear `deteccion/detector_<nombre>.py` con clase `Detector<Nombre>` y método `detectar_<nombre>(frame)`
2. El método debe retornar posición, bounding box y frame anotado
3. Agregar el detector al `Perceptor` en `entorno/perceptor.py` (importar + instanciar + asignar cadencia)
4. Agregar los campos correspondientes al estado en `_estado_vacio()` y al vector de observación en `_estado_a_obs()`
5. Probar individualmente: `python -m deteccion.detector_<nombre>`
6. Probar integrado: `python -m entorno.perceptor`

Para modificar hiperparámetros de entrenamiento:

1. Editar el diccionario de hiperparámetros en `entrenamiento/entrenar.py`
2. Ejecutar entrenamiento nuevo o continuar con `--continuar`

### 8.3 Ejecución de pruebas y validaciones

- **Pruebas unitarias de detectores:** Cada detector tiene método `probar()` que abre ventana en tiempo real
- **Pruebas de integración:** `python -m entorno.perceptor` verifica todos los detectores simultáneamente
- **Pruebas de pipeline completo:** Ejecutar entrenamiento corto (10K steps) y verificar que el agente recibe observaciones, ejecuta acciones y los episodios terminan/reinician correctamente
- **Métricas de rendimiento:** Verificar en consola que el hilo rápido opera a ≥15 FPS y el hilo lento a ≥15 FPS

### 8.4 Integración de cambios

- Usar ramas de feature para nuevas funcionalidades
- Verificar que los detectores no introducen falsos positivos antes de integrar
- Documentar cambios de umbrales o ROIs en el commit message

## 9. Dependencias y servicios externos

### 9.1 Servicios externos integrados

| Servicio | Propósito |
|---|---|
| BlueStacks 5 | Emulador Android para ejecutar Banana Kong |
| Banana Kong APK | Juego objetivo (incluido en el repositorio) |

### 9.2 Requisitos de acceso

- BlueStacks 5 instalado y configurado
- Banana Kong instalado en BlueStacks
- Ventana de BlueStacks en primer plano y visible durante la ejecución

### 9.3 Consideraciones de desarrollo y pruebas

- No hay entornos sandbox ni mocks; las pruebas requieren el juego real
- Los templates PNG deben coincidir visualmente con los elementos del juego
- Si la versión del juego cambia, los templates y umbrales deben recalibrarse

## 10. Convenciones del proyecto

### 10.1 Convenciones de código

- **Nomenclatura de archivos:** `detector_<objeto>.py` para detectores, nombres descriptivos para módulos
- **Nombres de clases:** `Detector<Objeto>` (PascalCase)
- **Nombres de métodos:** `detectar_<objeto>(frame)` (snake_case)
- **Constantes:** MAYUSCULAS con guión bajo (ej: `BARRIL_AREA_MIN`, `UMBRAL`)
- **Docstrings:** Cada módulo tiene docstring descriptivo al inicio
- **Umbrales y configuraciones:** Definidos como constantes al inicio de cada módulo para fácil calibración

### 10.2 Convenciones de repositorio

- **Ramas:** `main` para código estable, ramas de feature para desarrollo
- **Commits:** Mensajes descriptivos en español, indicando el módulo afectado
- **Modelos y logs:** Excluidos de git vía `.gitignore` (archivos grandes)

### 10.3 Convenciones de documentación

- El `Informe.md` es el documento técnico principal
- `Instalacion.md` contiene la guía de instalación y configuración
- `Desarrollo.md` (este documento) contiene la guía técnica de desarrollo
- Los diagramas de arquitectura se mantienen en `diagramas/` como PlantUML (.txt) y PNG

## 11. Problemas frecuentes y recomendaciones

### 11.1 Problemas frecuentes

| Problema | Causa | Solución |
|---|---|---|
| Falsos positivos en barriles | Área máxima muy alta | Reducir `BARRIL_AREA_MAX` (calibrado a 1400) |
| Kong no detectado | Tracker CSRT perdió al personaje | El anti-drift re-inicializa automáticamente cada 25 frames |
| Game over no detectado | Template desactualizado | Capturar nuevo template del texto "Revive?" |
| Entrenamiento colapsa | Learning rate muy alto | Usar ≤ 2e-4; el colapso ocurre con 3e-4 |
| FPS muy bajos | Todos los detectores en cada frame | Ajustar cadencias (`*_CADA`) según criticidad |

### 11.2 Deuda técnica conocida

- **Evaluación formal pendiente:** No se realizó evaluación contra baseline aleatoria con análisis estadístico (t-test)
- **Un solo bioma:** El agente opera exclusivamente en el mundo selva; otros mundos invalidan los detectores
- **Sin interfaz gráfica:** La ejecución es por línea de comandos
- **Resolución fija:** Todos los detectores están calibrados para 960×540; no hay soporte para múltiples resoluciones
- **Objetos no detectados:** Lianas, trampolines y guacamaya no tienen detectores implementados

### 11.3 Recomendaciones para continuidad

1. **Si se quiere soportar más mundos:** Implementar un detector de transición de mundo que pause la detección y recalibre ROIs/umbrales
2. **Si se quiere mejorar rendimiento:** Considerar reducir la resolución de captura o usar detección por GPU (CUDA OpenCV)
3. **Si se quiere más precisión:** Agregar detección de objetos interactivos (lianas, trampolines) para que el agente pueda usarlos estratégicamente
4. **Si se quiere generalizar:** Implementar un pipeline de calibración automática que ajuste umbrales HSV a partir de screenshots etiquetados

## 12. Historial de decisiones técnicas relevantes

| Decisión | Alternativas consideradas | Razón |
|---|---|---|
| **PPO sobre DQN/A3C** | DQN (más sensible a hiperparámetros), A3C (requiere múltiples workers) | PPO es más estable, disponible en SB3, buen desempeño con pocas acciones discretas |
| **HSV + Template sobre CNN pura** | CNN end-to-end (10x más costoso en entrenamiento) | Hardware académico limitado; extracción manual de features es 10x más eficiente en muestras |
| **Teclas sobre gestos táctiles** | ADB (latencia alta), gestos drag (BlueStacks interpreta como tap) | Game Controls con teclas elimina el problema de tap accidental |
| **Vector de features sobre píxeles** | Política CNN directa desde píxeles | 10x menos pasos de entrenamiento necesarios con features extraídas manualmente |
| **VecFrameStack(4) sobre observación simple** | Observación de un solo frame | Permite al agente inferir velocidades y trayectorias sin modelo recurrente |
| **Flags de presencia sobre 0.0 por defecto** | Usar 0.0 como valor cuando no hay obstáculo | El modelo interpretaba 0.0 como "obstáculo en la esquina superior izquierda" |
| **2 hilos daemon sobre hilo único** | Un solo hilo para todos los detectores | Kong y bananas necesitan actualización cada frame; obstáculos pueden ser menos frecuentes |
| **CSRT Tracker sobre detección por frame** | Re-detectar Kong en cada frame con HSV+Template | El tracker es más rápido y evita perder a Kong durante animaciones complejas |
| **Entrenamiento en CPU sobre GPU** | GPU con CUDA | PPO con MlpPolicy es liviano; el cuello de botella es la percepción, no la inferencia |

## 13. Referencias relacionadas

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [mss Documentation](https://python-mss.readthedocs.io/)
- [pyautogui Documentation](https://pyautogui.readthedocs.io/)
- [Informe principal del proyecto](./Informe.md)
- [Guía de instalación](./Instalacion.md)
