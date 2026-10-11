# Comparación de experimentos

Todos los experimentos usan **las mismas particiones** (entrenamiento 140.274 filas / validación 35.067 filas,
agrupadas por patrón de entradas, 0 patrones compartidos) y se evalúan con las mismas métricas.
Las hipótesis se escribieron en `src/experiment_configs.py` y se confirmaron en Git (commit `8e44467`)
**antes** de entrenar. Cada ejecución está en MLflow (experimento `defiende-el-servidor`) con su hipótesis como etiqueta.

Métricas en validación con umbral 0,5 (el umbral de alerta se elige después, ver `DECISION.md`):

| Experimento | Cambio vs referencia | Macro-F1 | Recall | Precisión | FPR | AUC | Tiempo |
|---|---|---|---|---|---|---|---|
| E0 baseline «siempre ataque» | — | 0,405 | 1,000 | 0,681 | 1,000 | 0,500 | 0,5 s |
| E1 Regresión logística cruda | modelo lineal vs E0 | 0,922 | 0,988 | 0,922 | 0,178 | 0,984 | 8 s |
| E2 Reg. logística + log(1+x) | + transformación vs E1 | 0,928 | 0,975 | 0,938 | 0,138 | 0,987 | 9 s |
| E3 Reg. logística + log + balanceo | + class_weight vs E2 | 0,927 | 0,943 | 0,962 | 0,080 | 0,987 | 12 s |
| E4 Random Forest | otro tipo de modelo vs E2 | 0,948 | 0,980 | 0,956 | 0,096 | 0,993 | 19 s |
| E5 HistGradientBoosting | boosting vs E4 | 0,949 | 0,976 | 0,961 | 0,086 | 0,994 | 13 s |
| E6 Random Forest con 25 % de los datos | menos datos vs E4 | 0,942 | 0,980 | 0,949 | 0,112 | 0,992 | 4 s |

## Hipótesis, resultado y conclusión

**E0 — baseline.** *Hipótesis:* siempre «ataque» da ~68 % de accuracy con 100 % de falsas alarmas.
*Resultado:* accuracy 68,06 %, FPR 100 %, macro-F1 0,405. *Conclusión:* se cumple; confirma que la accuracy sola engaña
(coincide con la lámina 2 del EDA).

**E1 — regresión logística cruda.** *Hipótesis:* mucho mejor que el baseline, pero limitada por las escalas sin transformar.
*Resultado:* macro-F1 0,922, FPR 17,8 %. *Conclusión:* parcialmente contradicha: es mucho mejor que el baseline, pero
**no tan limitada** como esperaba; un modelo lineal simple ya recupera casi todo el recall. Esto bajó mi expectativa
de cuánto aportaría la complejidad.

**E2 — log(1+x).** *Hipótesis:* sube el macro-F1 y baja la FPR frente a E1. *Resultado:* +0,63 puntos de macro-F1, FPR 17,8 % → 13,8 %
(con algo menos de recall, 98,8 % → 97,5 %). *Conclusión:* se cumple, pero la mejora es modesta. Se conserva la transformación
para los modelos lineales.

**E3 — balanceo de clases.** *Hipótesis:* menos falsas alarmas a costa de algo de recall. *Resultado:* FPR 13,8 % → 8,0 %,
recall 97,5 % → 94,3 %, macro-F1 sin mejora (−0,14 puntos). *Conclusión:* cambia el **tipo** de error, no la calidad
global. Con el umbral ajustado a FPR ≤ 5 % (tabla de `DECISION.md`) E2 y E3 quedan casi idénticos (0,9157 vs 0,9161):
**balancear clases se parece mucho a mover el umbral.** Este fue el experimento que más cambió mi opinión.

**E4 — Random Forest.** *Hipótesis:* los árboles capturan interacciones y superan a la mejor regresión logística.
*Resultado:* +1,97 puntos de macro-F1 vs E2 (0,928 → 0,948), FPR 13,8 % → 9,6 %. *Conclusión:* se cumple; es la mayor mejora
entre experimentos y viene de cambiar de familia de modelo, no de afinar el lineal.

**E5 — HistGradientBoosting.** *Hipótesis:* mejora a E4 en AUC con ganancia pequeña (<1 punto) por el techo que imponen
duplicados y etiquetas ambiguas. *Resultado:* +0,14 puntos de macro-F1; con el umbral final la diferencia es 0,0016
(IC 95 % por bootstrap pareado: −0,0001 a 0,0032). *Conclusión:* se cumple, y la ventaja sobre Random Forest es
**un empate técnico**: el IC incluye ≈0. Se elige E5 por la regla declarada, no porque la diferencia sea concluyente.

**E6 — 25 % de los datos.** *Hipótesis:* pérdida <1 punto y ~4x menos tiempo. *Resultado:* −0,64 puntos de macro-F1, tiempo
19 s → 4 s (≈4-5x). *Conclusión:* se cumple; usar más datos compensa poco en este dataset (muchas filas repetidas),
aunque a igual umbral la FPR sube de 9,6 % a 11,2 %, así que no es gratis.

## Limitaciones de esta comparación
- Una sola partición de validación: diferencias menores a ~0,3 puntos (E4 vs E5) no son concluyentes.
- Los modelos usan hiperparámetros razonables por defecto, sin búsqueda sistemática: la comparación responde a preguntas
  de diseño, no busca el máximo absoluto.
- Validación aleatoria agrupada, no cronológica (el dataset preparado no trae `Stime`/`Ltime`).
