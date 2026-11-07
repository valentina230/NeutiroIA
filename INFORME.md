# Informe de experimentos: el cerebro de Neurito

Red de `main.py`: 3 entradas (hambre, energía, aburrimiento) → 6 neuronas ocultas (ReLU) → 3 salidas (Softmax: comer, dormir, jugar). Se entrena con 3000 ejemplos sintéticos, 1200 épocas y tasa 0.1. Los resultados salen de `experimentos.py` (`uv run experimentos.py`); la precisión se mide con 1000 ejemplos nuevos.

La etiqueta sale de una regla inventada por nosotros: gana la necesidad más alta entre `hambre`, `1 - energía` y `aburrimiento`. Lo que mide la «precisión» es cuánto copia la red esa regla.

## Experimentos

**1. Dos necesidades fijas, variar la tercera.** Con energía 0.5 (cansancio 0.5) y aburrimiento 0.3, la acción cambia de *dormir* a *comer* cuando el hambre pasa de ~0.5. Es justo donde la regla cambia: el hambre empieza a superar al cansancio.

**2. Prioridades casi iguales.**

| Caso | comer | dormir | jugar |
|---|---|---|---|
| hambre 0.50 vs cansancio 0.50 | 43.5 % | 56.5 % | 0.0 % |
| hambre 0.52 vs cansancio 0.50 | 52.6 % | 47.4 % | 0.0 % |
| hambre 0.80 vs cansancio 0.50 | 99.4 % | 0.6 % | 0.0 % |

Cerca del empate las probabilidades quedan repartidas (la red «duda»). Una diferencia de 0.02 ya voltea la decisión, pero con poca confianza. Con diferencia grande Softmax se acerca al 100 %.

**3. Neuronas ocultas** (media de 3 semillas):

| Ocultas | Pérdida | Precisión prueba |
|---|---|---|
| 3 | 0.1747 | 96.8 % |
| 6 | 0.1437 | 98.8 % |
| 10 | 0.1264 | 98.8 % |
| 20 | 0.1201 | 99.2 % |

Más neuronas bajan la pérdida, pero con rendimientos decrecientes: de 6 a 20 la mejora es pequeña. La regla es sencilla, así que 3 neuronas ya bastan para ~97 %.

**4. Épocas y tasa de aprendizaje.**

| Épocas (tasa 0.1) | Pérdida | Precisión prueba |
|---|---|---|
| 50 | 1.014 | 62.2 % |
| 200 | 0.598 | 87.5 % |
| 600 | 0.217 | 97.2 % |
| 1200 | 0.137 | 98.3 % |
| 5000 | 0.062 | 99.3 % |

Mejora rápido y luego se frena (curva que se aplana).

| Tasa (1200 épocas) | Pérdida | Precisión prueba | Comportamiento |
|---|---|---|---|
| 0.001 | 1.083 | 31.4 % | demasiado lenta, casi no aprende |
| 0.01 | 0.829 | 81.3 % | lenta |
| 0.1 | 0.137 | 98.3 % | estable |
| 1.0 | 0.075 | 95.9 % | la pérdida sube y baja (580 subidas): ruidosa |
| 5.0 | 1.098 | 33.6 % | falla |
| 20.0 | 3.858 | 33.8 % | inestable |

**5. Una cuarta acción.** Haría falta: (a) una cuarta neurona de salida (`W2` pasa de 6×3 a 6×4 y `b2` de 1×3 a 1×4); (b) una regla de etiquetas que la incluya, por ejemplo una cuarta necesidad «higiene» y la acción «bañarse» (esto añade una entrada: `W1` pasa de 3×6 a 4×6); (c) ejemplos suficientes de la nueva clase, también cerca de las fronteras con las otras, y balanceados; (d) volver a entrenar y volver a medir la precisión.

## Explica tus resultados

**1. ¿En qué se diferencian pesos y sesgos?** Los pesos (`W1`, `W2`) multiplican cada entrada y fijan cuánta importancia tiene (p. ej. cuánto pesa el hambre en la neurona de «comer»). Los sesgos (`b1`, `b2`) se suman al resultado sin depender de ninguna entrada y desplazan el umbral a partir del cual una neurona se activa. Esta red tiene 3·6 + 6 + 6·3 + 3 = 45 parámetros.

**2. ¿Por qué el cansancio es `1 - energía`?** La regla compara «necesidades» y en las tres debe valer que más alto = más urgente. Hambre y aburrimiento ya funcionan así, pero la energía va al revés: energía alta significa poca necesidad de dormir. Con `1 - energía` el cansancio crece cuando la energía baja, y `argmax` puede comparar las tres en la misma dirección.

**3. ¿Qué hace ReLU y qué hace Softmax?** ReLU (`max(0, z)`) deja pasar los valores positivos y anula los negativos. Sin ella, dos capas lineales equivaldrían a una sola; ReLU introduce la no linealidad que permite aprender fronteras de decisión más complejas. Softmax convierte las tres puntuaciones de salida en probabilidades positivas que suman 1; la exponencial amplifica la mayor, así que una puntuación claramente superior se acerca al 100 %.

**4. ¿Por qué una precisión alta puede ser engañosa?** Por varias razones:
- Con clases desbalanceadas, predecir siempre la clase común da precisión alta: en una simulación con 5 % de casos urgentes, «nunca urgente» acierta el 95.5 % y detecta 0 de 45 urgentes. Por eso el conjunto de entrenamiento debe estar equilibrado.
- Se mide frente a datos de la misma distribución y frente a una regla inventada, no frente al mundo real: 98 % significa «copia bien nuestra regla», no «decide bien».
- Los errores se concentran en las fronteras (prioridades casi iguales), que es donde más importa y donde menos pesan en el promedio.
- La precisión no dice nada sobre la confianza: acertar con 52 % o con 99 % cuenta igual.
