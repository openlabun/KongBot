# Instalación y despliegue

## 1. Descripción general de la solución

### 1.1 Lenguajes y tecnologías utilizadas

| Tecnología | Rol |
|---|---|
| Python 3.9+ | Lenguaje principal |
| OpenCV | Visión por computador (HSV, template matching, contornos, CSRT tracker) |
| mss | Captura de pantalla (~60 FPS, Windows GDI) |
| Stable-Baselines3 | Algoritmo PPO (MlpPolicy, VecFrameStack) |
| Gymnasium | Interfaz estándar de entorno RL (step, reset) |
| pyautogui | Simulación de teclado (keyDown/Up) |
| pygetwindow | Detección de ventana de BlueStacks |
| numpy | Operaciones numéricas y procesamiento de arrays |
| TensorBoard | Monitoreo de métricas de entrenamiento |
| BlueStacks 5 | Emulador Android para ejecutar el juego |

### 1.2 Componentes de la solución

El sistema está organizado en 4 capas:

1. **Captura** (`mss`): obtiene fotogramas de la ventana de BlueStacks a ~60 FPS
2. **Percepción** (`deteccion/` + `entorno/perceptor.py`): 9 detectores especializados coordinados en 2 hilos daemon (rápido y lento)
3. **Decisión** (`entorno/entorno.py` + `entrenamiento/`): entorno Gymnasium + agente PPO con VecFrameStack(4)
4. **Acción** (`controles/acciones.py`): traduce acciones discretas a teclas W/D/S

## 2. Requisitos previos

### 2.1 Software requerido

| Software | Versión | Notas |
|---|---|---|
| Windows | 10/11 | Requerido (usa API GDI para captura) |
| Python | 3.9+ | 3.11 recomendado |
| BlueStacks 5 | Última estable | Con Banana Kong instalado |
| Banana Kong APK | 1.9.16+ | Incluido en el repositorio |
| Procesador | Moderno | Entrenamiento en CPU, percepción OpenCV en CPU |

### 2.2 Variables de entorno

No se requieren variables de entorno. Toda la configuración se realiza por código (umbrales de detectores, cadencias, hiperparámetros de entrenamiento).

## 3. Instalación para ambiente de desarrollo

### 3.1 Desarrollo sin contenedores

#### 3.1.1 Clonar el repositorio

```bash
git clone https://github.com/KidmanC/KongBot-Agente-Aut-nomo-para-Banana-Kong-mediante-Aprendizaje-por-Refuerzo
cd KongBot-Agente-Aut-nomo-para-Banana-Kong-mediante-Aprendizaje-por-Refuerzo
```

#### 3.1.2 Instalar dependencias

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

#### 3.1.3 Configurar BlueStacks

**Instalar el juego**

El APK de Banana Kong se incluye en la raíz del repositorio. Instálalo arrastrando el archivo `.apk` sobre la ventana de BlueStacks. Esto evita tener que iniciar sesión en Google Play Store.

**Importar configuración de controles**

El archivo `com.fdgentertainment.bananakong.cfg` en la raíz del repositorio contiene la configuración de controles exportada de BlueStacks. Para importarla:

1. Abre BlueStacks → **Configuración → Importar/Exportar configuración de controles**
2. Selecciona el archivo `.cfg`
3. Reinicia BlueStacks

Si prefieres configurar los controles manualmente, sigue las instrucciones de la sección **Controles** más abajo.

**Resolución**

La resolución debe ser exactamente **960×540**:

1. Abre BlueStacks → **Configuración → Display**
2. Resolución: `960 × 540`
3. DPI: `240`
4. Guarda y reinicia BlueStacks

**Desactivar anuncios**

> **Importante:** Los anuncios de BlueStacks modifican el tamaño de la ventana de juego, lo que desplaza los ROIs de todos los detectores y causa fallos en la detección.

1. Abre BlueStacks → **Configuración → Preferencias**
2. Busca la opción **"Permitir que BlueStacks muestre anuncios"**
3. **Desactívala**
4. Reinicia BlueStacks

**Modo avión (alternativa):**

Si los anuncios persisten, activa el modo avión del emulador:

1. En la **barra lateral derecha** de BlueStacks, busca el ícono de **Modo avión**
2. Actívalo para desconectar BlueStacks de internet
3. Esto elimina cualquier anuncio que dependa de conexión de red

> El modo avión es la opción más efectiva, ya que bloquea anuncios que no se desactivan desde Preferencias.

**Controles**

Dentro del juego, abre el **Game Controls** (ícono de teclado en la barra lateral de BlueStacks). La configuración **no se hace tecla por tecla**, se usa un control de tipo **SWIPE**:

1. En el editor de controles, busca el item **SWIPE** y arrástralo a la pantalla
2. Configura los siguientes gestos:

| Gesto | Tecla | Acción en el juego |
|-------|-------|-------------------|
| Deslizar derecha | `D` | Dash (impulso hacia adelante) |
| Deslizar abajo | `S` | Bajar / Deslizarse |

3. Para la tecla `W` (Saltar / Planear), agrega un control de tipo **Tap** y mapealo en la **parte inferior derecha de la pantalla**, donde Kong salta al tocar.

> **¿Por qué SWIPE y no teclas individuales?** BlueStacks interpreta el inicio de cualquier drag como un tap, lo que hacía que Kong saltara antes de ejecutar el dash. Usar un control SWIPE nativo separa correctamente el gesto de salto del gesto de dash.

#### 3.1.4 Ejecutar servicios requeridos

No se requieren servicios externos. El sistema opera completamente en el proceso Python local con BlueStacks como único componente externo.

#### 3.1.5 Iniciar la aplicación

**Verificar detectores individualmente:**

```bash
python -m deteccion.detector_kong
python -m deteccion.detector_bananas
python -m deteccion.detector_barriles
python -m deteccion.detector_rocas
python -m deteccion.detector_muros
python -m deteccion.detector_agua
python -m deteccion.detector_mina
python -m deteccion.detector_tubo
python -m deteccion.detector_gameover
python -m entorno.perceptor        # todos los detectores juntos
```

Cada detector abre una ventana con las detecciones en tiempo real. Presiona `q` para cerrar.

**Entrenar el agente:**

```bash
# Entrenamiento desde cero
python -m entrenamiento.entrenar

# Continuar entrenamiento previo
python -m entrenamiento.entrenar --continuar

# Añadir steps específicos
python -m entrenamiento.entrenar --continuar --add-steps 100000
```

**Monitorear en TensorBoard:**

```bash
tensorboard --logdir logs/
```

Abre `http://localhost:6006` en el navegador.

### 3.2 Desarrollo con contenedores

No aplica. El proyecto requiere acceso directo a la ventana de BlueStacks en Windows y simulación de teclado, lo que no es compatible con contenedores Docker.

## 4. Despliegue

### 4.1 Arquitectura de despliegue

No aplica. El proyecto es académico y opera en una máquina local con BlueStacks. No hay servidor, API ni despliegue en la nube.

### 4.2 Proceso de actualización

Para actualizar el proyecto:

1. Hacer `git pull` para obtener los últimos cambios
2. Reinstalar dependencias si `requirements.txt` cambió: `pip install -r requirements.txt`
3. Verificar que los templates en `deteccion/templates/` estén actualizados
4. Si la versión del juego cambió, recalibrar umbrales de detectores

### 4.3 Despliegue sin contenedores

#### 4.3.1 Preparación del servidor

No aplica. Se ejecuta en máquina local con Windows 10/11.

#### 4.3.2 Instalación de dependencias

Ver sección 3.1.2.

#### 4.3.3 Configuración de la aplicación

Ver sección 3.1.3 (configuración de BlueStacks).

#### 4.3.4 Ejecución de la aplicación

Ver sección 3.1.5.

#### 4.3.5 Actualización de versiones

Si se actualiza la versión de Banana Kong:

1. Capturar nuevos screenshots del juego
2. Extraer nuevos templates PNG con canal alpha
3. Recalibrar umbrales HSV ejecutando cada detector en modo prueba
4. Ajustar ROIs si la disposición visual cambió

### 4.4 Despliegue con contenedores

No aplica.

## 5. Verificación de funcionamiento

Para verificar que todo funciona correctamente:

1. **Captura:** Ejecutar `python -m entorno.perceptor` y verificar que la ventana Debug muestra el frame del juego
2. **Percepción:** Verificar que cada detector marca correctamente los objetos en pantalla (bounding boxes de colores)
3. **FPS:** Verificar en consola que el hilo rápido opera a ≥15 FPS y el hilo lento a ≥15 FPS
4. **Entorno:** Ejecutar `python -m entrenamiento.entrenar` y verificar que el agente comienza a recibir observaciones y ejecutar acciones
5. **Reinicio:** Morir intencionalmente en el juego y verificar que el entorno detecta game over y reinicia automáticamente

## 6. Solución de problemas frecuentes

| Problema | Causa | Solución |
|---|---|---|
| Detectores no encuentran objetos | Resolución incorrecta | Configurar BlueStacks a 960×540 exactamente |
| Falsos positivos constantes | Anuncios de BlueStacks activos | Desactivar anuncios en Preferencias o activar Modo avión |
| Kong salta antes del dash | Controles mal configurados | Usar SWIPE nativo, no gestos manuales |
| Template no encontrado | Archivo PNG sin canal alpha | Regenerar templates con fondo transparente |
| Ventana de BlueStacks no detectada | Título de ventana cambiado | Verificar que el título contenga "BlueStacks" |
| Entrenamiento muy lento | CPU limitado | Reducir n_steps o batch_size en hiperparámetros |
| Colapso de política | Learning rate muy alto | Usar learning_rate ≤ 2e-4 |
| Game over no detectado | Template de "Revive" desactualizado | Capturar nuevo template del texto |

## 7. Mantenimiento y actualización

- **Templates:** Si el juego se actualiza y cambian los gráficos visuales, los templates PNG deben regenerarse
- **Umbrales HSV:** Si los colores del juego cambian, recalibrar rangos HSV ejecutando cada detector y observando los valores H/S/V reales
- **ROIs:** Si la disposición de la pantalla cambia, ajustar las coordenadas de ROI en cada detector
- **Hiperparámetros:** Si se busca mejor rendimiento, experimentar con learning_rate, ent_coef y n_steps
- **Cadencias:** Ajustar `AGUA_CADA`, `BARRILES_CADA`, etc. según la carga computacional del sistema

## 8. Referencias relacionadas

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [mss Documentation](https://python-mss.readthedocs.io/)
- [pyautogui Documentation](https://pyautogui.readthedocs.io/)
- [BlueStacks Support](https://support.bluestacks.com/)
