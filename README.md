# ¿Pueden consensuar los árboles de decisión?: Experimentos con inteligencia colectiva y Random Forest

#### María Giménez Costa, Federico Giorgi y Gastón Loza Montaña

Este repositorio cuenta con el desarrollo para el Proyecto Final de la Licenciatura en Tecnología Digital de la Universidad Torcuato Di Tella.

## Resumen

Este trabajo explora alternativas al algoritmo **Random Forest** (RF) para mejorar su rendimiento en problemas de regresión, implementando estrategias de agregación inspiradas en la deliberación grupal humana. RF utiliza múltiples árboles de decisión y combina sus predicciones mediante la media, pero estudios recientes de comportamiento humano demuestran que la combinación de decisiones consensuadas es un mecanismo superador.

Con esta base, se extiende la implementación de RF de la librería *Scikit-learn*, creando nuevos modelos que agrupen a los árboles de decisión independientes y simulen un “debate” entre los mismos para alcanzar predicciones consensuadas con la hipótesis de que agregar esta etapa intermedia de deliberación provea predicciones más precisas que el algoritmo tradicional. Se implementaron distintas variantes:

- Exclusión de predicciones extremas, eliminando valores atípicos dentro de cada grupo.
- Promedios ponderados según confianza, asignando más peso a los árboles más confiables e incluyendo una variante combinada con la exclusión de extremos.
- Construcción de nuevos árboles, donde se genera un árbol nuevo por grupo combinando la información de los árboles iniciales.
- Intercambio de conocimiento, donde las predicciones de los árboles en el grupo influyen en los demás durante su construcción.

Los experimentos, evaluados con diferentes conjuntos de datos y mediante el error cuadrático medio (MSE), mostraron que las variantes propuestas lograron un rendimiento similar al RF estándar, sin mejoras significativas. En particular, la variante de combinación de árboles mostró un rendimiento inferior, mientras que las demás variantes tuvieron resultados variables según los datos analizados, con algunas mejoras leves en ciertos casos.

A pesar de que la hipótesis inicial no fue confirmada, este estudio abre nuevas líneas de investigación. Futuras exploraciones podrían incluir la simulación de factores como la presión social entre árboles o aplicar estas técnicas en otros modelos de machine learning. Optimizar hiperparámetros no explorados en esta investigación también podría mejorar los resultados, y estas variantes propuestas podrían integrarse en futuras versiones de *Scikit-learn* para su aplicación en diversas áreas.

## Implementación

La configuración del entorno con la librería *Scikit-learn* en modo editable para correr las implementaciones al algoritmo original se llevó a cabo con el prodecimiendo descripto en `setup.pdf`.

Las variantes del algortimo de RF se implementaron dentro del código fuente de la librería en la carpeta `scikit-learn-main/sklearn`.

A continuación los archivos adicionales o con código agregado:

- `ensamble/`:
    - `__init__.py`
    - `_extremos_forest.py`
    - `_first_split_combiner.py`
    - `_group_debate.py`
    - `_oob_forest.py`
    - `_shared_knowledge.py`
- `tree/`:
    - `__init__.py`
    - `_combined_tree.py`
    - `_ext_tree.py`
    - `_extended_tree.pxd`
    - `_extended_tree.pyx`
    - `_tree_comb.pxd`
    - `_tree_comb.pyx`
    - `_splitter.pxd`
    - `_splitter.pyx`

