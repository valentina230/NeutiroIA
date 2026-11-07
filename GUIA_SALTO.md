# Neurito 2.0: aprende a saltar obstáculos

Neurito aprende **cuándo saltar** ante obstáculos que se aproximan con diferentes velocidades. El cerebro se implementa manualmente con NumPy y el aprendizaje puede observarse en Pygame.

## ¿Qué aprende?

La red recibe cuatro características normalizadas: distancia al obstáculo, velocidad, tiempo estimado hasta alcanzarlo (distancia / velocidad) y la **altura del obstáculo**. El tiempo estimado es una característica derivada, no una observación independiente.

Los obstáculos aparecen al azar en **filas de 1 a 3 seguidos**, y cada uno puede ser **azul** (36 px), **rojo** (47 px) o **morado** (58 px, el más alto y con menos margen para saltar). Entre un obstáculo y el siguiente hay tiempo para aterrizar y volver a saltar. La red mira siempre el obstáculo más cercano que aún no ha pasado.

La arquitectura es **4 → 12 → 1**: doce neuronas ocultas con ReLU y una salida con sigmoide. La última devuelve una puntuación entre cero y uno que, con el umbral 0.5, determina si Neurito debe saltar cuando está en el suelo.

## ¿Cómo aprende?

Aplicamos **aprendizaje supervisado**, no aprendizaje por refuerzo. Creamos datos sintéticos que incluyen ejemplos donde es conveniente saltar y otros donde debe esperar. Para generar las etiquetas, definimos una ventana de referencia calculada con la física (`ventana_ttc()` en `salto_juego.py`): saltar cuando el tiempo estimado cae en el rango en que Neurito va a estar por encima del obstáculo mientras lo cruza, con un colchón de 2 fotogramas. Esa ventana depende de la altura del obstáculo y de la velocidad (por ejemplo, a 4.5 px/frame: rojo de 6 a 12.9 fotogramas; morado de 7 a 10.9). No es una regla aprendida por recompensa; si se cambia, no garantiza evitar choques.

La red ajusta sus pesos a partir de la **entropía cruzada binaria**, la **retropropagación** y el **descenso de gradiente**:

`W_nuevo = W_anterior - tasa_de_aprendizaje * gradiente`

El conjunto de entrenamiento equilibra aproximadamente las dos decisiones. De otra manera, una red podría aparentar buena precisión diciendo casi siempre «esperar».

La precisión se mide sobre ejemplos distintos de los utilizados en el entrenamiento. **Una precisión alta no garantiza un buen resultado en el juego**, que también depende de la física del salto.

### Tope de precisión (94 %)

En `saltar.py`, `PRECISION_MAXIMA = 0.94` detiene el entrenamiento al llegar a esa precisión de prueba. Con el tope, Neurito salta casi siempre pero falla algunas veces (en nuestras pruebas, alrededor de 4 choques por cada 100 obstáculos): justo la situación de la pregunta central. Con `PRECISION_MAXIMA = None` la precisión sube a ~99 % y no falla. Además, el mismo 94 % puede dar resultados distintos según la semilla del cerebro, porque la precisión no mide cuándo salta cada vez.

## Archivos

```text
Neurito/
├── main.py                 # Mascota original: comer, dormir, jugar
├── red-sencilla.py         # Ejemplo de una neurona
├── salto_modelo.py         # Red neuronal manual y datos de entrenamiento
├── salto_juego.py          # Física, obstáculos (azul/rojo/morado) y colisiones
├── experimentos.py         # Experimentos con la red de main.py
├── INFORME.md              # Resultados y explicaciones
├── saltar.py               # Juego y visualización del entrenamiento
├── GUIA_SALTO.md
└── assets/
    └── *.png              # Imágenes de la mascota
```

## Ejecución

```bash
uv sync
uv run saltar.py
```

No se necesitan dependencias nuevas: se reutilizan NumPy y Pygame. Al abrir, el cerebro se entrena unos segundos hasta el tope de precisión y Neurito salta solo desde el primer obstáculo.

## Controles

| Tecla | Función |
|---|---|
| `1` | Entrenar una época |
| `2` | Entrenar cien épocas |
| `3` | Activar o pausar entrenamiento continuo |
| `A` | Alternar control automático y manual |
| `Espacio` | Saltar manualmente |
| `C` | Reiniciar pista y contadores |
| `R` | Reiniciar pesos (cerebro sin entrenar) y contadores |
| `T` | Entrenar de golpe (hasta 3000 épocas o hasta el tope de precisión) |
| `Esc` | Salir |

## Experimentos

1. Pulsa `R` para dejar el cerebro sin entrenar y prueba el juego. Anota colisiones y obstáculos esquivados.
2. Entrena de cien en cien épocas; compara la pérdida, la precisión y el comportamiento.
3. Cambia `ocultas=12` por `ocultas=3` o `ocultas=24`; compara generalización.
4. Modifica la tasa de aprendizaje `0.8` a `0.01` o `1.2` y estudia la convergencia.
5. Cambia el `margen` de `ventana_ttc()` (de 2.0 a 0 o a 5) o la altura del morado en `TIPOS_OBSTACULO`. Comprueba si la política resultante sigue siendo compatible con la física.

**Pregunta central:** ¿por qué una red puede clasificar ejemplos con buena precisión y aun así chocar?
