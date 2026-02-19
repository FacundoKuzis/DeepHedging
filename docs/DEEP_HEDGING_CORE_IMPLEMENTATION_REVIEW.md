# Deep Hedging Core: Implementacion, Comparaciones, Monte Carlo y Simulacion de Precios

## 1. Objetivo de este documento
Este documento describe, de forma operativa y paso a paso, como funciona el core de Deep Hedging de este repo, dejando de lado la parte GAN/Diffusion.

Incluye:
1. Implementacion de agentes.
2. Logica de entrenamiento y evaluacion/comparacion.
3. Pricing y delta por Monte Carlo.
4. Simulacion de precios (GBM/Heston).
5. Errores de implementacion encontrados y mejoras necesarias.

## 2. Arquitectura funcional del core
Los bloques principales son:
1. Instrumentos (generan paths): `src/DeepHedging/HedgingInstruments/stock.py`.
2. Claims (definen payoff): `src/DeepHedging/ContingentClaims/*.py`.
3. Costos de transaccion: `src/DeepHedging/CostFunctions/*.py`.
4. Medidas de riesgo: `src/DeepHedging/RiskMeasures/risk_measures.py`.
5. Agentes (politicas de cobertura): `src/DeepHedging/Agents/*.py`.
6. Entorno (loop de entrenamiento/evaluacion): `src/DeepHedging/Environments/environment.py`.
7. Scripts CLI de entrenamiento y comparacion: `examples/train.py`, `examples/evaluate_agents.py`, `examples/compare_hedging_strategies.py`, `examples/bootstrap_confidence_intervals.py`.

La interfaz comun esperada por el entorno es:
1. Instrumento: `generate_paths(num_paths, random_seed)`.
2. Claim: `calculate_payoff(paths)`.
3. CostFunction: `calculate(actions, paths)`.
4. RiskMeasure: `calculate(pnl)`.
5. Agent: `process_batch(paths, T_minus_t)` que devuelve acciones por paso.

## 3. Flujo end-to-end (step by step)

### Paso 1: Configuracion del experimento
En los scripts de ejemplo se define:
1. Mercado: `T`, `N`, `r`, `S0`, `sigma`.
2. Claim: tipo de opcion y strike.
3. Agente(s): entrenable(s) y/o analitico(s).
4. Risk measure para entrenar/comparar (ej. CVaR).

### Paso 2: Generacion de paths
`Environment.generate_data(...)` itera por instrumentos y arma un tensor final con forma:
- `(n_paths, N+1, n_instruments)`
Referencia: `src/DeepHedging/Environments/environment.py:32`.

### Paso 3: Construccion de input temporal
Se calcula `T_minus_t` con forma `(batch, N)`:
- `T_minus_t[k, t] = (N - t) * dt`
Referencia: `src/DeepHedging/Environments/environment.py:169`.

### Paso 4: Politica del agente
Cada agente produce `actions` con forma `(batch, N+1, n_instruments)`:
1. `SimpleAgent`/`RecurrentAgent`: paso a paso.
2. `LSTM`/`GRU`/`WaveNet`: secuencial de una vez.
3. Al final se concatena una accion cero en `T`.

### Paso 5: PnL de cobertura
`Environment.calculate_pnl(...)` hace:
1. Posicion acumulada: `cumsum(actions)`.
2. Valor de cartera por instrumento: `cumsum(actions) * paths`.
3. Cashflows de trading y costos.
4. Capitalizacion del cash con tasa libre de riesgo.
5. Resta payoff final del claim.
Referencia: `src/DeepHedging/Environments/environment.py:54`.

### Paso 6: Loss de entrenamiento
`loss = risk_measure.calculate(pnl)`
Referencia: `src/DeepHedging/Environments/environment.py:91`.

### Paso 7: Optimizacion
`BaseAgent.train_batch(...)`:
1. `process_batch(...)`.
2. Evalua loss.
3. Backprop con `GradientTape`.
Referencia: `src/DeepHedging/Agents/base_agent.py:121`.

## 4. Implementacion de agentes

## 4.1 Agentes entrenables (redes)

### SimpleAgent
- MLP densa por timestep.
- Input: `[features_instrumento, T_minus_t]`.
Referencia: `src/DeepHedging/Agents/simple_agent.py`.

### RecurrentAgent
- Similar a Simple, pero agrega en input la posicion acumulada.
- Estado interno: `accumulated_position`.
Referencia: `src/DeepHedging/Agents/recurrent_agent.py`.

### LSTMAgent y GRUAgent
- Modelos recurrentes secuenciales con `return_sequences=True`.
- Procesan la trayectoria completa (hasta `N`).
Referencia: `src/DeepHedging/Agents/lstm_agent.py`, `src/DeepHedging/Agents/gru_agent.py`.

### WaveNetAgent
- Conv1D causal con bloques residuales y dilataciones.
- Alternativa temporal no recurrente.
Referencia: `src/DeepHedging/Agents/wavenet_agent.py`.

## 4.2 Agentes analiticos/no entrenables

### DeltaHedgingAgent (Black-Scholes)
- Delta cerrada usando `d1` y CDF normal.
- Accion = `delta_t - delta_{t-1}`.
Referencia: `src/DeepHedging/Agents/delta_hedging_agent.py`.

### Variantes asiaticas
- Geometric Asian (formula cerrada o numerica/QuantLib segun agente).
- Arithmetic Asian por Monte Carlo/QuantLib para delta.
Referencias: `src/DeepHedging/Agents/geometric_asian_*.py`, `src/DeepHedging/Agents/arithmetic_asian_*.py`.

### MonteCarloAgent
- Recalcula delta por finite differences sobre pricing Monte Carlo.
- Sin red neuronal.
Referencia: `src/DeepHedging/Agents/montecarlo_agent.py`.

## 4.3 Transformacion de features de path
`BaseAgent.transform_paths(...)` soporta:
1. `log`.
2. `log_moneyness`.
Luego concatena `T_minus_t`.
Referencia: `src/DeepHedging/Agents/base_agent.py:19` y `src/DeepHedging/Agents/base_agent.py:80`.

## 5. Comparaciones entre agentes

## 5.1 Error terminal y metricas
`Environment.terminal_hedging_error_multiple_agents(...)`:
1. Genera paths comunes para todos.
2. Obtiene acciones por agente.
3. Calcula PnL y error terminal (con precio inicial segun `pricing_method`).
4. Grafica histogramas y exporta stats.
Referencia: `src/DeepHedging/Environments/environment.py:182`.

## 5.2 Metodos de pricing para comparar
- `fixed`: todos pagan el precio del primer agente.
- `individual`: cada agente usa su propio precio.
Referencia: `src/DeepHedging/Environments/environment.py:211`.

## 5.3 Bootstrap de intervalos de confianza
`bootstrap_confidence_intervals(...)`:
1. Re-muestrea errores terminales.
2. Calcula estadisticos y CI por agente.
Referencia: `src/DeepHedging/Environments/environment.py:609`.

## 6. Monte Carlo en este repo

## 6.1 Pricer
`MonteCarloPricer`:
1. Simula paths (GBM/Heston).
2. Evalua payoff.
3. Descuenta al presente.
4. Delta por diferencia finita central.
Referencia: `src/DeepHedging/utils/monte_carlo_pricer.py`.

## 6.2 Uso dentro de agentes
`MonteCarloAgent` usa el pricer para obtener delta por cada estado `(S_t, T-t)`.
Referencia: `src/DeepHedging/Agents/montecarlo_agent.py:107`.

## 7. Simulacion de precios

## 7.1 GBM
Ecuacion discretizada:
- `S_t = S_{t-1} * exp((r - 0.5*sigma^2)dt + sigma*dW)`
Referencia: `src/DeepHedging/HedgingInstruments/stock.py:56`.

## 7.2 Heston
Sistema discreto:
1. Varianza: mean reversion + ruido.
2. Precio: drift `r` y volatilidad `sqrt(v_t)`.
3. Correlacion `rho` entre shocks.
Referencia: `src/DeepHedging/HedgingInstruments/stock.py:120`.

## 8. Errores de implementacion encontrados y mejoras necesarias

## 8.1 Errores/fallos funcionales (alta prioridad)

1. `SimpleAgent` incompatible con wrappers CLI actuales.
- Evidencia:
  - `examples/train.py:33-36`
  - `examples/evaluate_agents.py:156-160`
  - `examples/compare_hedging_strategies.py:131-136`
  - `examples/bootstrap_confidence_intervals.py:163-167`
- Problema: esos helpers pasan `n_hedging_timesteps=...` a todo agente entrenable, pero `SimpleAgent.__init__` no acepta ese argumento (`src/DeepHedging/Agents/simple_agent.py:21`).
- Impacto: seleccionar `SimpleAgent` desde CLI puede romper con `TypeError`.
- Mejora: aplicar introspeccion de firma como ya se hizo en `examples/run_with_globals.py:287-295`.

2. Semilla compartida puede forzar correlacion artificial entre instrumentos.
- Evidencia:
  - `Environment.generate_data(..., random_seed=seed)` pasa la misma seed a cada instrumento (`src/DeepHedging/Environments/environment.py:38`).
  - GBM/Heston resetean RNG global con esa seed (`src/DeepHedging/HedgingInstruments/stock.py:73`, `src/DeepHedging/HedgingInstruments/stock.py:144`).
- Impacto: si hay varios instrumentos, pueden terminar con ruido identico no deseado.
- Mejora: usar `np.random.Generator` por instrumento o derivar seeds por instrumento (`seed + i`).

3. Logging por timestep en agentes analiticos.
- Evidencia:
  - `src/DeepHedging/Agents/delta_hedging_agent.py:104`
  - `src/DeepHedging/Agents/arithmetic_asian_montecarlo_agent.py:188`
- Impacto: fuerte degradacion de performance y ruido de consola en corridas grandes.
- Mejora: remover prints por paso o usar logging con nivel debug.

## 8.2 Riesgos metodologicos importantes

1. Reuso del mismo dataset durante todos los epochs.
- Evidencia: `train_data = self.generate_data(train_paths)` se genera una sola vez antes del loop de epochs (`src/DeepHedging/Environments/environment.py:99-105`).
- Impacto: riesgo de sobreajuste a una muestra MC fija.
- Mejora: opcion `resample_each_epoch=True` para regenerar paths por epoch.

2. Suposicion fuerte de que el claim depende solo del instrumento 0.
- Evidencia: `calculate_payoff(paths[:, :, 0])` (`src/DeepHedging/Environments/environment.py:81`).
- Impacto: limita claims multi-activo y puede producir errores silenciosos si el usuario espera otra cosa.
- Mejora: permitir mapping explicito de instrumentos por claim.

3. Eliminacion de agentes duplicados por `agent.name`.
- Evidencia:
  - `terminal_hedging_error_multiple_agents`: `src/DeepHedging/Environments/environment.py:199-206`
  - `bootstrap_confidence_intervals`: `src/DeepHedging/Environments/environment.py:651-658`
- Impacto: no se pueden comparar dos variantes del mismo tipo de agente con distinto checkpoint/hyperparams si comparten nombre.
- Mejora: deduplicar por identificador unico (`agent_id`) o no deduplicar.

4. Convenciones temporales inconsistentes entre scripts.
- Evidencia:
  - `train.py` usa default `T=63/252` (`examples/train.py:59`).
  - `evaluate_agents.py` y otros defaults usan `T=22/365` (`examples/evaluate_agents.py:85`, `examples/compare_hedging_strategies.py:70`, `examples/bootstrap_confidence_intervals.py:85`).
- Impacto: comparaciones no consistentes si se usan defaults sin cuidado.
- Mejora: unificar base de tiempo (idealmente 252 para trading days) o forzarla explicita.

## 8.3 Mejoras de eficiencia y robustez

1. `MonteCarloAgent` recalcula pricing por cada path y timestep re-instanciando pricer.
- Evidencia: `src/DeepHedging/Agents/montecarlo_agent.py:118-144`.
- Impacto: costo computacional muy alto.
- Mejora: vectorizar deltas, cachear simulaciones, o usar aproximador de delta offline.

2. `risk_measure` no se valida en constructor.
- Evidencia: `loss_function` asume `self.risk_measure` no nulo (`src/DeepHedging/Environments/environment.py:91-93`).
- Impacto: falla tardia si se olvida configurar.
- Mejora: validar en `__init__` y lanzar error temprano.

3. Alineacion de documentacion y scripts.
- Evidencia: README indica `INSTRUMENT_NAME` con opciones limitadas (`README.md:39`) mientras codigo ya soporta mas.
- Impacto: confusion operativa.
- Mejora: actualizar README con opciones y rutas reales de salida.

## 9. Recomendacion de ejecucion para trabajo futuro
Orden recomendado:
1. Corregir wrappers CLI para `SimpleAgent` (bloqueante funcional).
2. Corregir estrategia de semillas multi-instrumento (bloqueante metodologico).
3. Limpiar prints por timestep en agentes analiticos (performance).
4. Agregar opcion de resample por epoch (mejora estadistica).
5. Unificar convencion temporal (`252` vs `365`) y documentarla.

Con esto, el core de Deep Hedging queda mas coherente para experimentacion comparativa y para resultados defendibles frente a revision tecnica.
