# Deep Hedging: Informe de Lógica y Capacidades Actuales

## 1) Propósito del documento
Este informe resume, en lenguaje funcional (más que de implementación), qué hace hoy el pipeline de Deep Hedging en este repositorio, cómo está organizado, qué variantes experimentales existen y qué capacidades operativas están disponibles para entrenar, comparar y reportar resultados.

El foco está en:
- lógica de negocio y flujo end-to-end,
- configuración y reproducibilidad,
- capacidades experimentales activas (A1, A2, A3, A4),
- artefactos y métricas que se generan.

---

## 2) Vista general del sistema
La plataforma está organizada alrededor de cuatro bloques:

1. **Generación/obtención de paths**  
   Permite simular (GBM) o evaluar sobre ventanas históricas reales.

2. **Agentes de cobertura**  
   Incluye agentes entrenables (Recurrent, LSTM, GRU, WaveNet, etc.) y benchmarks no entrenables (Delta Hedging, variantes asiáticas).

3. **Entorno de entrenamiento/evaluación**  
   Gestiona PnL, costos de transacción, riesgo objetivo, batching, semillas, contexto histórico y cálculo de métricas.

4. **Runners de consola + sistema de configs**  
   Ejecutan `train` / `compare` / `report` con JSON estricto e herencia (`extends`), guardando artefactos en estructura reproducible.

---

## 3) Runners principales y para qué se usan

### 3.1 Runners unificados (recomendados)
- `examples/train_console.py`
- `examples/compare_console.py`

Estos detectan pipeline automáticamente y resuelven configs desde `configs/` con herencia.

### 3.2 Runners especializados
- `examples/thesis_result1_train_console.py`, `examples/thesis_result1_compare_console.py`
- `examples/thesis_result1b_train_console.py`, `examples/thesis_result1b_compare_console.py`
- `examples/thesis_result1b_report_console.py`
- `examples/thesis_result1b_option_market_compare_console.py`
- `examples/thesis_result1b_plot_paths_console.py`

Se mantienen por compatibilidad y por casos específicos de evaluación.

---

## 4) Lógica de configuración (JSON)

## 4.1 Herencia y composición
El sistema de configs soporta:
- `extends` (string o lista),
- merge jerárquico (padres -> hijo),
- sobreescritura explícita por run.

Esto permite definir:
- **bases** reutilizables (`configs/bases/...`),
- **runs** concretos (`configs/runs/...`) con cambios mínimos.

## 4.2 Filosofía de validación
- Esquema estricto por runner (faltantes/extra -> error).
- Tipos y rangos validados.
- Campos opcionales solo cuando aplican.

---

## 5) Universo de agentes y claims

## 5.1 Agentes entrenables (deep hedging)
- `SimpleAgent`
- `RecurrentAgent`
- `LSTMAgent`
- `GRUAgent`
- `WaveNetAgent`

## 5.2 Benchmarks no entrenables
- `DeltaHedgingAgent` (Black-Scholes)
- agentes de asiáticas (geométricas/ariméticas, variantes MC/control variate y numéricas).

## 5.3 Claims soportados
- Europeas call/put.
- Asiáticas geométricas call/put.
- Asiáticas aritméticas call/put.

Los claims definen payoff y selección del underlying dentro del tensor de paths.

---

## 6) Flujo lógico de entrenamiento (`train`)

1. **Carga y validación de config**.
2. **Resolución de run** y creación de carpetas de salida.
3. **Calibración de parámetros de mercado**:
   - `r` (risk-free),
   - `sigma` (histórica o implícita, según config).
4. **Construcción de instrumento** (usualmente GBM en Result1/Result1b).
5. **Construcción de claim y risk measure**.
6. **Construcción del agente** (incluyendo transformaciones y contexto).
7. **Entrenamiento por epochs**:
   - con/ sin `resample_each_epoch`,
   - LR strategy configurable (`constant`, `exponential_decay`, `reduce_on_plateau`),
   - early stopping opcional.
8. **Guardado de artefactos**:
   - modelo,
   - optimizer,
   - historial de entrenamiento,
   - manifests de calibración/config.

---

## 7) Flujo lógico de comparación (`compare`)

1. **Carga y validación de config de comparación**.
2. **Construcción de benchmark + carga de modelos entrenados**.
3. **Preparación de paths de test**:
   - simulados, o
   - ventanas históricas reales.
4. **Asignación por path de parámetros** (`r`, `sigma`) según estrategia configurada.
5. **Cálculo de acciones por agente** (con cache opcional de acciones para acelerar reprocesos).
6. **Cálculo de PnL / hedge error**.
7. **Métricas puntuales y de cola** + bootstrap.
8. **Guardado de tablas y plots comparativos**.

---

## 8) Contexto histórico (sin y con uso por el agente)

Hay tres modos conceptuales:

1. **No context**  
   No se genera ni usa historia previa.

2. **Context generated only (`ctxgen`)**  
   Se genera historia para construir trayectorias más realistas, pero el agente no la ve como input.

3. **Context seen (`seen`)**  
   El agente recibe información histórica como features de contexto.

Features de contexto hoy:
- `log_returns`
- `log_moneyness`

Además, se puede elegir cómo tratar `T_minus_t` en pre-contexto (`calculated` o `zero`).

---

## 9) A4: Encoder Conv1D de contexto (entrenado en conjunto)

En A4 se introdujo un encoder de historia configurable mediante:
- `history_conv1d_enabled`
- `history_conv1d_layers` (lista de capas)
- `history_conv1d_pooling`

### Lógica funcional
1. El contexto histórico se arma de forma causal.
2. Si Conv1D está activo, ese contexto se codifica antes de entrar al modelo principal.
3. El embedding resultante se concatena al input base (precio transformado + `T_minus_t` + opcional `log(K)`).
4. Encoder Conv1D + modelo principal se entrenan con la misma loss y el mismo optimizer.
5. Se guarda/carga junto al modelo principal (sidecar de encoder).

### Nota de diseño importante
Para agentes secuenciales (LSTM/GRU/WaveNet), cuando Conv1D de contexto está activo se prioriza el modo de contexto como feature (`seen`) y no el modo de prefijo temporal, para evitar caminos ambiguos.

---

## 10) Medidas de riesgo objetivo y optimización

El entrenamiento puede minimizar distintas funciones objetivo:
- `CVaR` (incluyendo shorthand como `CVaR95`, `CVaR50`),
- `MSE`,
- `MAE`.

Esto habilita estudios de sensibilidad sobre:
- robustez en cola (CVaR),
- ajuste medio/cuadrático (MAE/MSE).

---

## 11) Volatilidad y tasa libre de riesgo: modos soportados

## 11.1 Sigma
- fija,
- uniforme por path (`uniform`),
- discreta por regímenes (`discrete`),
- histórica/implied según calibración.

## 11.2 Risk-free
- fijo,
- por serie externa (con modos de mapeo por ventana cuando corresponde).

Esto permite comparar:
- mundos simples controlados (GBM),
- evaluación histórica 2020–2024,
- escenarios heterogéneos por path.

---

## 12) Monte Carlo y eficiencia
Para agentes/benchmarks que usan MC, hay soporte de:
- vectorización,
- chunking,
- paralelización configurable (`n_workers`, backend, tamaños de bloque),
- logging de progreso por lotes (evitando spam por iteración).

También hay soporte de control variate y common random numbers en rutas específicas de pricing asiático.

---

## 13) Estructura de experimentos actual (`configs/runs`)

## 13.1 A1
Comparativa base en GBM sigma 0.2:
- noctx / ctxgen / seen,
- variantes `wavenet` y `wavenet_logk`.

## 13.2 A2
Mismo setup, barrido de objetivo de riesgo:
- `CVaR95`, `CVaR50`, `MSE`, `MAE`,
- con y sin costos de transacción (`tc1`).

## 13.3 A3
Entrenamiento con volatilidad aleatoria por path:
- `uniform_narrow`,
- `uniform_wide`,
- `discrete_regimes`.

## 13.4 A4
Extensión de A3 con contexto `seen` codificado por Conv1D:
- WaveNet + Recurrent,
- variantes de sigma (fixed_s02, uniform_narrow, uniform_wide, discrete para WaveNet).

---

## 14) Artefactos que produce el sistema

Según pipeline y runner, se guarda:
- snapshot del config usado,
- modelos y optimizers,
- tablas de métricas (`point_metrics`, `empirical_risk_metrics`, bootstrap),
- plots de distribución/errores/comparativas,
- manifests de calibración y metadatos de run,
- caches de acciones para acelerar recomputaciones.

La convención de outputs está orientada a reproducibilidad y trazabilidad de runs.

---

## 15) Estado funcional actual (resumen ejecutivo)

Hoy la plataforma ya permite:
1. Definir experimentos complejos por JSON con herencia.
2. Entrenar y comparar múltiples familias de agentes y benchmarks.
3. Alternar objetivos de riesgo (CVaR/MSE/MAE).
4. Trabajar con contextos históricos en distintos modos.
5. Incluir codificación de historia vía Conv1D entrenada conjuntamente.
6. Simular regímenes de volatilidad por path y comparar contra delta hedging.
7. Ejecutar reportes consolidados para análisis de resultados.

---

## 16) Limitaciones y puntos a vigilar

1. **Convivencia de roots de salida legacy vs nueva estructura**  
   Existen scripts legacy y estructura organizada; conviene estandarizar una sola ruta para evitar confusión operativa.

2. **Multiplicidad de runners legacy**  
   Aunque hay wrappers unificados, todavía coexisten entradas antiguas. Recomendado: consolidar uso operativo en `train_console.py` y `compare_console.py`.

3. **Complejidad de combinaciones de contexto**  
   Hay varios modos (prefijo temporal vs features vistas); para comparaciones limpias conviene fijar protocolos por experimento.

4. **Alineación de métricas de negocio final**  
   El sistema ya cubre métricas estadísticas/riesgo, pero siempre es clave vincular resultados con criterio económico final (cola/costo/robustez según caso de uso).

---

## 17) Recomendación de uso operativo
Para nuevos experimentos:
1. Definir base en `configs/bases/<SET>/...`.
2. Crear runs concretos en `configs/runs/<SET>/...` con nombres cortos.
3. Ejecutar con runners unificados:
   - `python examples/train_console.py <ref>`
   - `python examples/compare_console.py <ref>`
4. Consolidar con reportes para ranking final por criterio de riesgo objetivo.

---

## 18) Resultados observados en runs guardados (snapshot)

Esta sección agrega evidencia empírica leída directamente de carpetas de resultados ya guardadas en:
- `G:\Mi unidad\Tesis2026\Models\Organized\A1\...`
- `G:\Mi unidad\Tesis2026\Models\Organized\A2\...`
- `G:\Mi unidad\Tesis2026\Models\Organized\A3\...`
- `G:\Mi unidad\Tesis2026\Models\Organized\A4\...`

Métricas usadas aquí:
- `std_error` (dispersión del error de cobertura),
- `mae_error` (error absoluto medio),
- `es_99` (Expected Shortfall al 99%, foco en cola izquierda),
- `exceedance_left_1pct` (frecuencia de eventos por debajo del cuantil 1%).

### 18.1 A1 (`euro_gbm / wavenet_logk`) por modo de contexto

Fuente:
- `...A1/euro_gbm/wavenet_logk/compare/noctx/tables/empirical_risk_metrics.csv`
- `...A1/euro_gbm/wavenet_logk/compare/ctxgen/tables/empirical_risk_metrics.csv`
- `...A1/euro_gbm/wavenet_logk/compare/seen/tables/empirical_risk_metrics.csv`

| Run | Agente | std_error | mae_error | es_99 | exceedance_left_1pct |
|---|---|---:|---:|---:|---:|
| `noctx` | Delta europeo | 0.4283 | 0.3228 | -1.4801 | 0.0100 |
| `noctx` | WaveNet | 1.1454 | 0.8918 | -4.1751 | 0.1409 |
| `ctxgen` | Delta europeo | 0.3132 | 0.1943 | -1.2993 | 0.0100 |
| `ctxgen` | WaveNet | 1.6523 | 1.1944 | -6.4292 | 0.2265 |
| `seen` | Delta europeo | 0.3132 | 0.1943 | -1.2993 | 0.0100 |
| `seen` | WaveNet | 2.1908 | 1.5215 | -8.8275 | 0.2459 |

Comentario:
- En A1, con esta configuración, WaveNet empeora al pasar de `noctx` a `ctxgen` y a `seen`.
- El benchmark delta mantiene ventaja amplia en cola y estabilidad.

### 18.2 A2 (`euro_gbm / wavenet_mny`) barrido de objetivo de riesgo

Fuente:
- `...A2/euro_gbm/wavenet_mny/compare/*/tables/empirical_risk_metrics.csv`
- `...A2/euro_gbm/wavenet_mny_tc1/compare/*/tables/empirical_risk_metrics.csv`

Sin costos de transacción (representativos):

| Run | Agente | std_error | mae_error | es_99 | exceedance_left_1pct |
|---|---|---:|---:|---:|---:|
| `mse_noctx` | Delta europeo | 0.4283 | 0.3228 | -1.4801 | 0.0100 |
| `mse_noctx` | WaveNet | 0.4792 | 0.3652 | -1.6213 | 0.0156 |
| `cvar95_noctx` | Delta europeo | 0.4283 | 0.3228 | -1.4801 | 0.0100 |
| `cvar95_noctx` | WaveNet | 0.5441 | 0.4349 | -1.3296 | 0.0071 |
| `mae_ctxgen` | Delta europeo | 0.3132 | 0.1943 | -1.2993 | 0.0100 |
| `mae_ctxgen` | WaveNet | 2.0318 | 1.4251 | -7.0467 | 0.2340 |

Con costos 1% (representativos):

| Run | Agente | std_error | mae_error | es_99 | exceedance_left_1pct |
|---|---|---:|---:|---:|---:|
| `mse_noctx` | Delta europeo | 0.7122 | 1.9399 | -4.4455 | 0.0100 |
| `mse_noctx` | WaveNet | 1.3931 | 1.1689 | -4.9255 | 0.0155 |
| `cvar50_noctx` | Delta europeo | 0.7122 | 1.9399 | -4.4455 | 0.0100 |
| `cvar50_noctx` | WaveNet | 0.9958 | 0.9823 | -5.0721 | 0.0120 |
| `cvar50_ctxgen` | Delta europeo | 0.8464 | 1.3770 | -4.2457 | 0.0100 |
| `cvar50_ctxgen` | WaveNet | 3.3486 | 2.1374 | -13.9724 | 0.1011 |

Comentario:
- A2 muestra que el objetivo de entrenamiento sí cambia el perfil de riesgo de forma material.
- `mse_noctx` y `cvar95_noctx` son relativamente estables frente a otras variantes.
- Variantes `ctxgen` (en especial `mae_ctxgen` y `cvar50_ctxgen`) se ven mucho más frágiles en cola.

### 18.3 Escenario `fixed_s02` (A4, sigma fija 0.2)

Fuente:
- `...A4/euro_gbm/wavenet_mny_randvol_conv/compare/fixed_s02/tables/empirical_risk_metrics.csv`
- `...A4/euro_gbm/recurrent_mny_randvol_conv/compare/fixed_s02/tables/empirical_risk_metrics.csv`

| Agente | std_error | mae_error | es_99 | exceedance_left_1pct |
|---|---:|---:|---:|---:|
| Delta europeo | 0.3132 | 0.1943 | -1.2993 | 0.0100 |
| Recurrente | 0.7074 | 0.4537 | -1.9756 | 0.0390 |
| WaveNet | 0.7579 | 0.5715 | -2.5984 | 0.0853 |

Comentario:
- En este corte, el benchmark delta sigue claramente mejor en dispersión y cola.
- Recurrente mejora respecto de WaveNet en cola (`es_99` menos extremo) y en `mae_error`.

### 18.4 Escenario `uniform_wide` (A3/A4, sigma aleatoria amplia)

Fuente:
- `...A3/euro_gbm/wavenet_mny_randvol/compare/uniform_wide/tables/empirical_risk_metrics.csv`
- `...A4/euro_gbm/recurrent_mny_randvol_conv/compare/uniform_wide/tables/empirical_risk_metrics.csv`

| Agente | std_error | mae_error | es_99 | exceedance_left_1pct |
|---|---:|---:|---:|---:|
| Delta europeo | 0.4637 | 0.2619 | -2.1294 | 0.0100 |
| Recurrente (A4 Conv1D) | 1.2450 | 0.7480 | -4.6093 | 0.0531 |
| WaveNet (A3) | 3.7689 | 1.9644 | -21.2680 | 0.1449 |

Comentario:
- La heterogeneidad de volatilidad penaliza fuerte a los modelos entrenados, sobre todo WaveNet en este setup.
- El recurrente con Conv1D de contexto reduce deterioro frente a WaveNet, pero todavía queda lejos del benchmark delta.

### 18.5 Lectura rápida de entrenamiento (A4 WaveNet, `fixed_s02`)

Fuente:
- `...A4/euro_gbm/wavenet_mny_randvol_conv/train/fixed_s02/tables/training_history.csv`

Observación:
- `train_loss` baja aproximadamente de `8.97` a `8.43-8.55` en las primeras épocas registradas.
- `val_loss` se mueve alrededor de `8.50-8.60`, señal de mejora inicial con meseta temprana.

Interpretación operativa:
- El pipeline está aprendiendo (no está colapsando), pero la generalización en cola sigue siendo el cuello de botella principal en escenarios de sigma heterogénea.
