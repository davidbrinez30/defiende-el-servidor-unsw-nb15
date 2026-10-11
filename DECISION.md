# Decisión final

## Modelo y umbral elegidos
- **Modelo:** HistGradientBoosting (E5) dentro de un `Pipeline` de scikit-learn con su preprocesamiento
  (one-hot de `proto`/`service`/`state`, categorías raras y desconocidas agrupadas como «infrecuente»).
  Entrada: las 42 variables predictoras; se excluyen `id` (orden del archivo) y `attack_cat` (respuesta relacionada).
- **Umbral de alerta: 0,6544.**
- **Regla, declarada en `params.yaml` antes de ver resultados:** entre los umbrales con tasa de falsos positivos (FPR) ≤ 5 %
  en validación, elegir el de mayor recall; entre modelos, el de mayor macro-F1 con ese umbral.
- **Por qué ese equilibrio:** un ataque ignorado cuesta más que una falsa alarma, pero una alerta en cada 10 conexiones normales
  vuelve inútil la herramienta. Los 5 % son un tope de decisión, no un óptimo; el máximo macro-F1 sin tope estaría en umbral 0,5
  (0,9494, FPR 8,6 %).

| Umbral | Recall | Precisión | FPR | Macro-F1 (validación) |
|---|---|---|---|---|
| 0,5 | 97,6 % | 96,1 % | 8,6 % | 0,949 |
| 0,6 | 96,2 % | 97,2 % | 5,9 % | 0,949 |
| **0,6544 (elegido)** | **95,4 %** | **97,6 %** | **5,0 %** | **0,946** |
| 0,7 | 94,4 % | 98,0 % | 4,1 % | 0,942 |
| 0,8 | 92,0 % | 98,8 % | 2,3 % | 0,931 |

**Fuerza de la elección:** E5 supera a Random Forest (E4) por solo 0,0016 de macro-F1 (IC 95 %: −0,0001 a 0,0032): es un empate
técnico. Se eligió E5 por aplicar la regla; no hay evidencia de que sea mejor que E4.

## Resultado en test (evaluado UNA vez, con modelo y umbral congelados)
| Métrica | Validación | **Test** |
|---|---|---|
| Recall | 95,4 % | **96,9 %** |
| Precisión | 97,6 % | **86,6 %** |
| FPR (normales alarmadas) | 5,0 % | **18,3 %** |
| Macro-F1 | 0,946 | **0,898** |
| AUC ROC | 0,994 | 0,984 |

Matriz en test: 30.219 normales bien clasificadas, **6.781 falsas alarmas**, **1.390 ataques perdidos**, 43.942 ataques detectados.
El resultado se conserva tal como salió: no se reajustó nada después de verlo.

## Errores relevantes
1. **La validación fue optimista.** El tope de FPR del 5 % no se sostuvo: en test fue 18,3 %. La partición de validación es
   aleatoria dentro del entrenamiento (agrupada por patrón idéntico) y el test oficial tiene otra distribución: 0 % de sus filas
   repite un patrón de entradas visto en entrenamiento y los normales son 44,9 % (frente a 31,9 %). **La causa exacta no se
   demostró**; son las diferencias que se observaron.
2. **Las falsas alarmas se concentran:** 5.638 de 6.781 (83 %) tienen `service = «-»` y 962 son `http`.
3. **Familia más débil: Fuzzers, recall 79,9 %** (6.062 ejemplos). El resto de familias supera el 94 %
   (Analysis 94,1 %; Generic, Backdoor, Worms ≈ 100 %). El análisis por familia es posterior a la elección y no se usó para decidir.
4. Aceptado: ~4,6 % de ataques perdidos en validación (3,1 % en test) a cambio de menos falsas alarmas.

## Limitaciones del dataset y del trabajo
- Benchmark histórico de laboratorio con ataques sintéticos: no prueba eficacia en una red actual.
- 74.301 filas (42,4 %) del entrenamiento son repeticiones de un vector de entrada y 229 patrones tienen etiquetas normal y ataque a la vez:
  parte del error es irreducible con estas variables. No se eliminaron duplicados (se agruparon para la partición).
- Validación no cronológica (no hay `Stime`/`Ltime`). Una sola partición de validación; sin búsqueda de hiperparámetros.
- Los CSV vienen de una copia pública fijada a un commit; no se contrastaron con la descarga oficial de UNSW.
- Siguiente paso razonable (no hecho, para no contaminar el test): una validación que simule el cambio de distribución,
  por ejemplo agrupando por tipo de servicio o familia de ataque, y elegir el umbral con ella.
