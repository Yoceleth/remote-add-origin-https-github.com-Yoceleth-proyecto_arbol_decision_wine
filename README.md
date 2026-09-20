# Clasificación con Árbol de Decisión — Wine Dataset

## Descripción

Este proyecto implementa un clasificador mediante un **Árbol de Decisión** utilizando el dataset `Wine` incluido en `scikit-learn`.

El material de la actividad indica que el dataset contiene **178 muestras**, **13 características químicas** y **3 clases de vino**. El objetivo es entrenar un árbol, medir su precisión y observar cómo cambian las reglas al modificar `max_depth`.

## Requisitos

- Python 3
- scikit-learn

Instalación:

```bash
pip install scikit-learn
```

## Ejecución

Desde esta carpeta:

```bash
python main.py
```

El programa:

1. Carga el dataset Wine.
2. Divide los datos en 80% para entrenamiento y 20% para prueba.
3. Entrena un árbol con `max_depth=2`.
4. Calcula la precisión sobre los datos de prueba.
5. Muestra las reglas aprendidas.
6. Compara `max_depth` de 1, 2, 3, 4, 5 y `None`.
7. Muestra la profundidad real y número de hojas de cada árbol.

## Resultados y observaciones

La precisión depende de la partición de entrenamiento/prueba. En este proyecto se utiliza `random_state=42` y `stratify=y` para que la ejecución sea reproducible y las tres clases estén representadas proporcionalmente.

### ¿Qué sucede al cambiar `max_depth`?

`max_depth` limita cuántos niveles puede tener el árbol. Con una profundidad pequeña, el árbol genera pocas reglas y es relativamente fácil de interpretar, aunque puede tener menor capacidad para separar correctamente las clases.

Al aumentar la profundidad, el árbol puede crear reglas más específicas y separar mejor los datos de entrenamiento. Sin embargo, también aumenta la complejidad y puede producir reglas más numerosas y difíciles de interpretar. Por ello, una mayor profundidad no debe considerarse automáticamente mejor.

### ¿Cuál fue el máximo posible?

En este proyecto se compara explícitamente `max_depth=None`, que no establece un límite artificial de profundidad. La profundidad real alcanzada por el árbol se muestra en la salida del programa mediante `tree.get_depth()`.

### ¿Qué ocurre sin limitar la profundidad?

Con `max_depth=None`, el algoritmo puede continuar dividiendo los datos mientras encuentre divisiones que cumplan sus criterios de entrenamiento. Por ello, el árbol puede tener muchas más reglas y hojas que el árbol con `max_depth=2`.

La diferencia principal es la complejidad: el árbol sin límite puede representar relaciones más detalladas, pero sus reglas son menos sencillas de interpretar y existe mayor riesgo de ajustarse demasiado a los datos de entrenamiento.

## Evaluación de precisión

El programa calcula la precisión mediante:

```python
accuracy_score(y_test, y_pred)
```

La precisión reportada corresponde al conjunto de prueba, que representa el 20% de los datos.

Para conocer los valores exactos obtenidos en la ejecución, se debe ejecutar:

```bash
python main.py
```

## Opinión sobre los resultados

Considero que el dataset es apropiado para una práctica de clasificación con árboles de decisión porque tiene una variable objetivo categórica con tres clases y varias características numéricas que pueden utilizarse para construir reglas de separación.

El árbol resulta especialmente útil en este contexto porque sus decisiones pueden expresarse como reglas del tipo "si-entonces". Esto permite observar de manera relativamente sencilla qué características participan en las decisiones del modelo.

## ¿El dataset cumple con los requerimientos para utilizarse en un árbol de decisiones?

Sí. El dataset cumple con los requerimientos básicos de esta práctica: contiene observaciones con características de entrada y una variable objetivo con clases definidas. Además, las características son numéricas, por lo que pueden utilizarse directamente en las divisiones del árbol.

## Características fundamentales

Entre las características disponibles se encuentran:

- Alcohol
- Ácido málico
- Cenizas
- Magnesio
- Flavonoides
- Proline
- Y otras características químicas incluidas en el dataset

El árbol determina cuáles son útiles para separar las clases durante el entrenamiento. No es necesario asumir de antemano que una característica es la más importante: el modelo genera sus divisiones a partir de los datos.

Como posible información adicional, se podrían incorporar otras mediciones relacionadas con las propiedades químicas del vino, siempre que fueran consistentes, medibles y estuvieran disponibles para todas las muestras.

