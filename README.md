# Bot Autónomo para Banana Kong - Aprendizaje por Refuerzo

## Resumen ejecutivo

Este proyecto desarrolla un agente autónomo basado en aprendizaje por refuerzo (PPO) que aprende a jugar Banana Kong, un videojuego endless runner para Android, ejecutándose en un emulador BlueStacks para PC. El agente percibe el juego exclusivamente a través de capturas de pantalla y ejecuta acciones mediante simulación de teclado, sin acceso a la memoria del juego ni modificación del APK.

La problemática abordada es la integración de captura visual, percepción computacional y ejecución de acciones en un pipeline coherente que opere en tiempo real sobre un juego comercial de caja negra. La solución implementa 9 detectores especializados mediante estrategias híbridas de segmentación HSV y template matching, un entorno Gymnasium con vector de observación de 23 valores, y un agente PPO entrenado durante 500.000 pasos.

Los resultados obtenidos (ep_rew_mean de 19.06 y 17.98) superaron la meta establecida de [15, 20], con un pipeline que opera a 33-38 FPS en percepción y ~50-60 ms de latencia por ciclo, cumpliendo todos los requerimientos funcionales y no funcionales definidos.

## Documentación del repositorio

| Documento | Descripción |
|---|---|
| [Informe.md](./Informe.md) | Documento principal del proyecto: introducción, marco conceptual, arquitectura, implementación, resultados |
| [Instalacion.md](./Instalacion.md) | Guía de instalación, configuración de BlueStacks y despliegue |
| [Desarrollo.md](./Desarrollo.md) | Manual de desarrollo: estructura del código, detectores, flujo de trabajo, decisiones técnicas |

## Estudiantes

| Nombre | GitHub |
|---|---|
| Kidman Cabana | [@KidmanC](https://github.com/KidmanC) |
| Santiago Romero | [@SantiagoR0214](https://github.com/SantiagoR0214) |

## Tutores

- Margarita Gamarra (Docente proponente)
- Augusto Salazar (Co-asesor)