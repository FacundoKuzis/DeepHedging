# THESIS_FINAL: Escenarios Definitivos, Metodología y Runbook (Comité)

Este documento es la especificación y bitácora oficial de los escenarios `THESIS_FINAL`. Está escrito para un comité que **no verá el código**, por lo que incluye:

- Qué se genera en cada mundo (modelos estocásticos, distribuciones de parámetros, shocks, etc.)
- Qué ve cada agente (inputs causales) y qué está prohibido (no look-ahead)
- Qué se entrena exactamente (pérdida, dinámica de portafolio, costos)
- Qué se compara exactamente (métricas, bootstrap, fairness de pricing)
- Qué artefactos se guardan para análisis **offline** sin re-ejecutar

**Fuente de verdad operativa:** los JSON bajo `configs/runs/THESIS_FINAL/` y las bases `configs/bases/THESIS_FINAL/`.

---

## 0. Estructura de Configs (THESIS_FINAL)

Raíces:
- Bases de entrenamiento: `configs/bases/THESIS_FINAL/train/`
- Bases de comparación: `configs/bases/THESIS_FINAL/compare/`
- Runs concretos: `configs/runs/THESIS_FINAL/<MUNDO>/<SUBESCENARIO>/{train,compare,calibration}/*.json`

Mundos (carpetas bajo `configs/runs/THESIS_FINAL/`):
- `W1_gbm_fixed/`
- `W2_gbm_random_ctx50/`
- `W3_garch_t_random_ctx50/`
- `W4_garch_t_tail_ctx50/`
- `W5_hmm3_garch_t_ctx100/`

Subescenarios:
- W1: `1a_1`, `1a_1p`, `1a_2`, `1b_1`, `1b_2`, `1c_1`, `1c_2`
- W2–W5: `1a_1`, `1a_2`, `1c_1`, `1c_2`

Convención de outputs (carpeta destino real):
- Los runners guardan todo bajo: `G:\Mi unidad\Tesis2026\Models\Organized\...`
- El `run_dir` se define por el path relativo del config: por ejemplo
  `configs/runs/THESIS_FINAL/W3_garch_t_random_ctx50/1a_1_euro_tc0_cvar50/compare/vs_bs.json`
  produce outputs en:
  `G:\Mi unidad\Tesis2026\Models\Organized\THESIS_FINAL/W3_garch_t_random_ctx50/1a_1_euro_tc0_cvar50/compare/vs_bs/`

---

## 1. Notación y Grilla Temporal

Variables base:
- `S_t`: precio del subyacente en el paso `t` (nivel, no log)
- `K`: strike
- `x_t = ln(S_t / K)`: log-moneyness (log natural)
- `N`: número de pasos de hedge dentro de la ventana (en configs `n`)
- `trading_days_per_year = 252`
- `T = N / 252` (años)
- `dt = T / N = 1/252`
- `r`: tasa libre de riesgo anual
- `sigma`: volatilidad anual (interpretación depende del modelo)

**Importante (log):** en todas las features y plots, `log` significa `ln` (log natural), no `log10`.

---

## 2. Modelos de Generación de Paths (W1–W5)

Todos los mundos generan paths en **espacio log-precio** y luego convierten a precio:

- Se simulan log-retornos `lr_t`
- Se acumulan `log S_t = log S_0 + sum_{j=0..t-1} lr_j`
- Se devuelve `S_t = S_0 * exp(log S_t)`

En los mundos con “contexto”, se genera también una pre-historia de `L` días (50 o 100) consistente con el mismo generador. Así, en un mismo path se obtiene:

- Pre-historia: `S_{-L}, ..., S_{-1}` (solo para features/estimación)
- Ventana de hedge: `S_0, ..., S_N` (se usa para PnL/payoff/decisiones)

Nota sobre logs de entrenamiento (`calibrated r=..., sigma=...`):
- El framework siempre imprime un par “calibrado” `(r, sigma)` a partir de los campos del config (`fixed_risk_free`, `fixed_implied_vol`) para inicializar el objeto instrumento y para compatibilidad con pricing/benchmarks.
- En mundos con parámetros **random por path** (W2–W5), esos valores son **placeholders** y **no** representan el `sigma_i` real de cada path ni la dinámica completa (GARCH/HMM).

### 2.1 W1: `W1_gbm_fixed` (GBM, parámetros fijos)

Dinámica (GBM con innovaciones normales):

```
lr_t = (r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z_t
Z_t ~ N(0,1)
```

Parámetros:
- `r = 0.03` (fijo)
- `sigma = 0.20` (fijo)
- `S0 = 100`
- No hay contexto (`context_length = 0`)

### 2.2 W2: `W2_gbm_random_ctx50` (GBM, sigma aleatoria por path, contexto 50)

Misma dinámica GBM que W1, pero `sigma` se samplea **por path**:

- `sigma_i ~ Uniform(0.08, 0.60)` para cada path `i`
- `r` es **fijo**: `r = 0.03` (no se estima ni se randomiza en THESIS_FINAL)

Contexto visto por agentes:
- `context_length = 50`
- Se genera pre-historia de 50 días por path

### 2.3 W3: `W3_garch_t_random_ctx50` (GARCH(1,1) sin leverage + innovaciones t, contexto 50)

Este mundo usa volatilidad condicional tipo GARCH(1,1) y shocks t-Student **estandarizados a varianza 1**.

Innovaciones t estandarizadas:

```
t_raw ~ t_ν
z_t = t_raw * sqrt((ν-2)/ν)   => Var(z_t)=1 (ν>2)
```

Ecuaciones (anualizadas):

```
eps_t = sqrt(v_t) * z_t
v_{t+1} = omega + alpha*eps_t^2 + beta*v_t + leverage*1_{eps_t<0}*eps_t^2
lr_t = (r - 0.5*v_t)*dt + sqrt(v_t*dt)*z_t
```

En THESIS_FINAL:
- `leverage = 0` (sin asimetría por signo), pero la condición de estacionariedad se chequea igual.
- `omega` se define para que la varianza de largo plazo coincida con `sigma_long_run^2`:

```
omega = (1 - alpha - beta) * sigma_long_run^2
```

Randomización por path (parámetros sampleados independientemente, con chequeos de estabilidad):
- `sigma_long_run ~ Uniform(0.04, 0.80)`
- `alpha ~ Uniform(0.03, 0.12)`
- `beta ~ Uniform(0.55, 0.78)`
- `leverage = 0.0`
- `ν ~ Uniform(3.2, 18.0)`
- `r = 0.03` (fijo)

Chequeos (se aplican por path):
- `alpha + beta < 1`
- `alpha + beta + 2*leverage < 1` (aquí es equivalente al anterior porque `leverage=0`)

Contexto:
- `context_length = 50` (pre-historia de 50 días)

### 2.4 W4: `W4_garch_t_tail_ctx50` (W3 + tail shocks aleatorios)

Es W3, más shocks negativos en el log-retorno:

```
lr_t <- lr_t + shock_t
shock_t <= 0
```

Generación de shocks (por path):
- Magnitud: `mag ~ Uniform(0.02, 0.10)`
- Gap entre shocks: `gap ~ UniformInt(10, 30)`
- Se arranca en un `t` inicial aleatorio en `[10,30]` y luego se suma otro gap hasta `t >= N`.
- Cada shock es `shock_t = -mag` en el instante elegido.

Contexto:
- `context_length = 50`

### 2.5 W5: `W5_hmm3_garch_t_ctx100` (HMM 3 estados + GARCH+t, contexto 100)

Es un generador **HMM + GARCH** donde un estado discreto modula la volatilidad de largo plazo.

Estados:
- `s_t ∈ {0,1,2}` (3 estados)
- Distribución inicial `π` (vector tamaño 3) por path
- Matriz de transición `P` (3x3) por path

Randomización por path (modo `uniform_random`, con normalización para que sumen 1):
- `P_ij ~ Uniform(0.01, 1.0)`, luego se renormaliza cada fila
- `π_i ~ Uniform(0.01, 1.0)`, luego se renormaliza
- Multiplicadores de volatilidad por estado: `m_k ~ Uniform(0.35, 2.8)`, y se ordenan (`sort=true`) para imponer un ranking estable (baja->alta vol)

Varianza de largo plazo por estado:

```
sigma_long_run_path ~ Uniform(0.04, 0.80)
var_long_run(state=k) = (sigma_long_run_path * m_k)^2
omega_k = (1 - alpha - beta) * var_long_run(k)
```

La dinámica GARCH se ejecuta usando `omega_{s_t}` en cada paso.

**Nota crítica sobre `r` en W5 (THESIS_FINAL):**
- Aunque el generador samplea multiplicadores `r_multiplier[k] ~ Uniform(0.5, 1.8)`, en THESIS_FINAL el `r_per_path_mode` es `"fixed"` y por implementación **se fuerza** `r_multiplier[k] = 1`.
- Resultado: `r` queda fijo en `0.03` en todos los estados; el HMM condiciona solo la volatilidad (no el drift).

Contexto:
- `context_length = 100`

---

## 3. Payoffs (Claims)

Los claims usan `fixing_indices`. Si `fixing_indices = null`, se usan **todos** los timestamps disponibles.

### 3.1 Europea (EuropeanCall)

Con `S_T = S_N`:

```
payoff = max(S_T - K, 0)
```

### 3.2 Asiática aritmética (AsianArithmeticCall)

Con fijaciones `S_{t_j}`:

```
A = mean_j S_{t_j}
payoff = max(A - K, 0)
```

### 3.3 Asiática geométrica (AsianGeometricCall)

```
G = exp( mean_j ln(S_{t_j}) )
payoff = max(G - K, 0)
```

---

## 4. Trading, Costos y PnL (Cómo se Evalúa un Hedge)

### 4.1 Acciones y posiciones

Los agentes producen una secuencia de **trades** (incrementos de posición):
- `a_t` = trade ejecutado en `t` (cambio de posición) para `t=0..N-1`
- Por convención, `a_N = 0` (no se opera en el último timestamp)

La posición acumulada es:

```
phi_t = sum_{j=0..t} a_j
```

### 4.2 Costos de transacción (proporcionales)

Con tasa `lambda` (`proportional_cost`):

```
cost_t = lambda * |a_t| * S_t
```

En escenarios TC=0: `lambda=0.0`
En escenarios TC=1%: `lambda=0.01`

### 4.3 Dinámica de caja (cash account)

Se computa una cuenta de caja `B_t` con capitalización **discreta**:

```
growth = (1 + r)^(dt)
```

Cashflows de compras/ventas y costos:

```
cf_t = -a_t * S_t - cost_t
```

Recurrencia:

```
B_0 = cf_0
B_t = B_{t-1} * growth + cf_t,   t=1..N
```

### 4.4 Valor terminal, PnL y descuento

Valor terminal del portafolio:

```
V_T = phi_N * S_N + B_N
```

PnL terminal (replicación perfecta => PnL ~ 0):

```
PnL = V_T - payoff
```

Factor de descuento usado para métricas:
- Si `r` es fijo por path (THESIS_FINAL): `D = exp(-r*T)`
- Si `r` fuera stepwise: `D = exp(-sum_{t} r_t * dt)`

### 4.5 “Error” terminal usado en plots y métricas

En comparaciones estándar (`compare_mode="benchmark_vs_targets"`), se usa:

```
error = price + D * PnL
```

Alternativas (usadas solo en caso especial CVaR sweep):
- `discounted_pnl`: `error = D * PnL`
- `pnl`: `error = PnL`

---

## 5. Agentes Entrenables (Deep Hedgers)

En THESIS_FINAL se entrenan (según escenario):
- `RecurrentAgent` (feedforward por paso, con posición acumulada)
- `LSTMAgent` (secuencia completa con LSTM)

### 5.1 Features base (por paso t)

Transformación del path (para agentes entrenables):
- `path_transformation_type = log_moneyness` => el input del precio es `x_t = ln(S_t/K)`

Feature de tiempo:
- `tau_t = T - t*dt` (time-to-maturity)

### 5.2 Contexto causal (pre-historia + historial observado)

Si `use_price_history_context=true` y `context_length=L>0`, se construye por cada path un vector causal:

```
h_t = [ ln(S_{t-L+1}/K), ..., ln(S_t/K) ]   (orden: más viejo -> más nuevo)
```

Donde para `t=0`, esos `L` valores provienen de la pre-historia simulada `S_{-L}..S_{-1}`.

**Causalidad:** en el paso `t` nunca se usa información de `S_{t+1}` ni de retornos futuros.

### 5.3 Encoder Conv1D del contexto (CNN sobre el vector de lags)

Cuando `history_conv1d_enabled=true`, el vector `h_t` (largo `L`) se “ve” por una CNN 1D:

- Input a Conv1D: secuencia de longitud `L` con 1 canal
- Padding: `causal` (Keras) para mantener causalidad en la dirección “viejo -> nuevo”
- Residuales opcionales por capa
- Pooling final: `global_max` (en THESIS_FINAL)

Arquitectura usada en THESIS_FINAL (W2–W5):
- 3 capas Conv1D, cada una produce 16 filtros
- Dilataciones: 1, 2, 4
- Kernel size: 3
- Dropout: 0.00, 0.03, 0.03
- Residuales: off, on, on

Salida del encoder:
- Embedding `e_t ∈ R^16` que se concatena a los features base del agente.

### 5.4 RecurrentAgent (estructura)

Input por paso:

```
u_t = [ x_t, tau_t, e_t, phi_{t-1} ]
```

Red:
- Dense(64, relu) -> Dense(64, relu) -> Dense(1, linear)

Output:
- `sequence_output_mode="trade"` => el output se interpreta directamente como trade `a_t`.

### 5.5 LSTMAgent (estructura)

Input secuencial (N pasos):

```
U = [u_0, u_1, ..., u_{N-1}]     con u_t = [ x_t, tau_t, e_t ]
```

Red:
- LSTM(30, return_sequences=True)
- LSTM(30, return_sequences=True)
- Dense(1, linear) por paso

Output:
- `sequence_output_mode="trade"` => el output por paso es el trade `a_t`.

Nota de implementación importante:
- Cuando `history_conv1d_enabled=true`, se fuerza `context_as_timesteps=false` para agentes secuenciales (LSTM/GRU/WaveNet).
- Resultado: el contexto entra **por features** (via `e_t`), no como prefijo temporal extra.

---

## 6. Objetivo de Entrenamiento (Risk Measure)

En THESIS_FINAL la pérdida se calcula sobre `PnL` (definido arriba) en la distribución de paths de training.

### 6.1 CVaR (como está implementado)

Sea `PnL` la variable aleatoria de PnL terminal (por path). Definimos:

```
VaR_alpha = quantile_{(1-alpha)}(PnL)
CVaR_alpha = - E[ PnL | PnL <= VaR_alpha ]
```

El entrenamiento minimiza `CVaR_alpha` (por ejemplo `alpha=0.50` en “CVaR50”).

Interpretación:
- Penaliza la cola izquierda (los peores PnL).
- El signo “-” hace que minimizar la pérdida equivalga a “subir” la cola izquierda (menos pérdidas extremas).

### 6.2 Hiperparámetros de entrenamiento (comunes)

De `configs/bases/THESIS_FINAL/train/common.json`:
- `train_paths=60000`, `val_paths=8000` (W1–W4)
- `n_epochs=100` (base), con extensión dinámica (ver abajo)
- `batch_size=2000` (W1–W4)
- Seeds: `global_random_seed=233`
- `resample_each_epoch=true` (cada epoch genera un nuevo set de paths)

W5 ajusta para memoria/tiempo:
- `train_paths=45000`, `val_paths=5000`, `batch_size=1200`

### 6.3 Early stopping y extensión dinámica del máximo de épocas

Early stopping (independiente):
- `early_stopping_enabled=true`
- `early_stopping_patience=20`
- `early_stopping_min_delta=1e-4`
- Monitor: `val_loss` (si existe), si no `train_loss`

Extensión dinámica del máximo de épocas (“late stopping”):
- `max_epochs_extension_enabled=true`
- `max_epochs_extension_window_epochs=10`
- `max_epochs_extension_by=10`
- `max_added_epochs=100`

Regla:
- Se arranca con `planned_total_epochs = n_epochs = 100`.
- Si al llegar a `planned_total_epochs` el mejor epoch (según `val_loss`) está dentro de los últimos 10 epochs,
  se extiende `planned_total_epochs += 10`.
- Esto se puede repetir hasta consumir el presupuesto: `max_total_epochs = 100 + 100 = 200`.

Propósito:
- Evitar cortar justo cuando el mejor val aparece cerca del final, sin necesidad de fijar un tope enorme desde el inicio.

### 6.4 Checkpoints (anti “perder horas”)

Se guardan checkpoints de modelo + optimizador:
- Cada `checkpoint_every_epochs` (default 5) se guarda `latest`.
- Si mejora el mejor `val_loss`, se guarda `best`.
- Se guarda también un sidecar para el encoder Conv1D (`*.history_conv.pkl`).
- Si existe checkpoint `latest`, el runner puede reanudar (`checkpoint_resume_if_available=true`).

### 6.5 Loop de entrenamiento (paso a paso, exacto a nivel conceptual)

En cada run de entrenamiento se ejecuta este flujo (con `resample_each_epoch=true` en THESIS_FINAL):

1) **Inicialización**
   - Se construye el instrumento (generador de paths) según `instrument_model` y el mundo W1–W5.
   - Se construye el claim (European/Asian).
   - Se construye el agente (`RecurrentAgent` o `LSTMAgent`) con sus capas y encoder Conv1D si aplica.
   - Se inicializa el optimizador y el scheduler (`reduce_on_plateau` por default).

2) **Por epoch `e = 1..planned_total_epochs`**
   - Se generan paths **nuevos** de training:

     ```
     paths_train: shape (train_paths, N+1, 1)     # ventana de hedge
     pre_history_train: shape (train_paths, L, 1) # si context_length=L>0
     ```

   - Se generan paths **nuevos** de validación:

     ```
     paths_val: shape (val_paths, N+1, 1)
     pre_history_val: shape (val_paths, L, 1)
     ```

   - Se recorre training en batches (`batch_size`):
     - Para cada batch se computa la secuencia de trades `a_t` del agente:
       - Si el agente es secuencial (LSTM): produce `a_t` para todos los pasos en una sola pasada forward.
       - Si el agente es por paso (Recurrent): produce `a_t` iterando `t=0..N-1` (con `phi_{t-1}` en el input).
     - Se computa `PnL` por path con la dinámica de caja (sección 4).
     - Se calcula la pérdida `Loss = RiskMeasure(PnL)` (sección 6.1).
     - Se hace backprop y update del optimizador.

   - Se evalúa validación (sin gradientes):
     - Se calcula `val_loss` con el mismo pipeline.
     - Se guarda en history (append, no replace).

3) **Scheduler, early stopping, y extensión dinámica**
   - Scheduler `reduce_on_plateau`:
     - Si `val_loss` no mejora por `reduce_on_plateau_patience` epochs, multiplica LR por `reduce_on_plateau_factor`.
   - Early stopping (independiente):
     - Si `val_loss` no mejora por `early_stopping_patience` epochs (con `min_delta`), se corta.
   - Extensión dinámica del máximo de epochs (independiente):
     - Al llegar a `planned_total_epochs`, si el mejor `val_loss` está dentro de los últimos `max_epochs_extension_window_epochs`,
       se extiende `planned_total_epochs += max_epochs_extension_by`, hasta `n_epochs + max_added_epochs`.

4) **Guardado**
   - Checkpoints:
     - `latest` cada `checkpoint_every_epochs`
     - `best` cuando mejora el mejor `val_loss`
   - Modelo final:
     - Se guarda siempre (si `save_after_train=true`) aun si no fue el best.
   - Sidecars:
     - Se guarda el encoder Conv1D (`*.history_conv.pkl`) para poder reconstruir el input del modelo en compare.

---

## 7. Benchmarks (Qué hacen y qué parámetros “ven”)

### 7.1 Regla de fairness de pricing (obligatoria)

En todas las comparaciones estándar se fuerza:
- `pricing_method="fixed"`

Eso significa:
- Se computa el **precio por path** del **primer agente** (benchmark del run).
- Ese mismo precio se usa para calcular `error = price + D*PnL` para todos los agentes comparados dentro de ese run.

Motivo:
- Evita que dos agentes “compitan” usando distintos precios de referencia en el mismo experimento.

### 7.2 Black-Scholes Delta (Europea) – `DeltaHedgingAgent`

Delta (call):

```
d1 = [ ln(S/K) + (r + 0.5*sigma^2)*tau ] / [ sigma*sqrt(tau) ]
Delta_call = N(d1)
```

Usa `sigma` de dos formas posibles:
- Si `batch_path_sigma` es vector (n_paths): usa sigma constante por path.
- Si `batch_path_sigma` es matriz (n_paths, N): usa `sigma_t` en cada paso `t`.

En THESIS_FINAL (W2–W5):
- El benchmark recibe `sigma_t` estimada causalmente (ver sección 8).

Precio (para el `price` de fairness):
- BS closed-form con `S0`, `r` y `sigma_t0` (la sigma de t=0, estimada desde el contexto).

Acciones (trades) por paso:
- El benchmark calcula **posición target** `Δ_t` y la convierte a trade usando estado interno:

```
a_t = Δ_t - Δ_{t-1}     (con Δ_{-1} := 0)
phi_t = sum_{j<=t} a_j = Δ_t
```

- Si se habilita un no-trade-band interno del agente (`no_trade_band`), primero se aplica sobre la **posición target** y luego se toma el trade:

```
Δ̃_t = Δ_{t-1} si |Δ_t - Δ_{t-1}| es “chico” (modo absolute/percentage)
a_t = Δ̃_t - Δ̃_{t-1}
```

En THESIS_FINAL, la “banda NI” pedida para escenarios TC=1% se implementa **externamente** reutilizando acciones guardadas de `.1` y aplicando la regla de sección 10.

### 7.3 Delta Asiática geométrica – `GeometricAsianDeltaHedgingAgent`

Benchmark para **asiática geométrica discreta** bajo GBM.

Idea:
- Condiciona en los fixings observados hasta `t` y en el calendario de fixings restante.
- Modela el log de la media geométrica futura como Normal (por propiedades de GBM).

Pricing condicional (intuición + fórmula usada):
- Sea `M` el número total de fixings del contrato.
- Sea `L_t = sum_{j: fixing<=t} ln(S_{t_j})` la suma de logs ya fijados (incluye el fixing de `t` si `t` es fixing).
- Sean los fixings futuros expresados como offsets `τ_i = (t_i - t)*dt`, para `i=1..m_future`.

Bajo GBM con `r` y `sigma` (en el benchmark, `sigma` puede ser `sigma_t`):
- La suma de logs futuros es Normal, y por lo tanto el log de la media geométrica futura también es Normal.
- El benchmark construye:

```
drift = r - 0.5*sigma^2
mu = (m_future/M)*ln(S_t) + drift * (sum_i τ_i)/M
var = sigma^2 * [ sum_{i,j} min(τ_i, τ_j) ] / M^2
m = (L_t/M) + mu
```

Con eso, el precio condicional se computa como opción sobre una variable lognormal con parámetros `(m, var)`:

```
d1 = (m - ln(K) + var) / sqrt(var)
d2 = (m - ln(K)) / sqrt(var)
Price_t = exp(-r * max_i τ_i) * ( exp(m + 0.5*var) * N(d1) - K * N(d2) )
```

Delta:
- Se computa numéricamente por bump (diferencia finita central):

```
Delta_t ≈ (Price(S_t + eps) - Price(S_t - eps)) / (2*eps)
```

Con `eps = max(|S_t|*bump_rel, 1e-6)`.

Sigma:
- Igual que BS: puede ser vector o matriz stepwise.

### 7.4 Local Risk Minimization (LRM) – `LocalRiskMinimizationAgent`

Implementa la razón de hedge de LRM (un paso):

```
h_t = Cov(C_{t+1}, dS_{t+1}) / Var(dS_{t+1})
dS_{t+1} = S_{t+1} - S_t * exp(r*dt)
```

Donde `C_{t+1}` es el “continuation value” a `t+1` estimado por un provider.

Simulación:
- Outer simulation (`outer_paths`): simula `S_{t+1}` desde `S_t` asumiendo GBM con `r_t` y `sigma_t` del benchmark.
- Provider calcula `C_{t+1}` (forma depende del claim):
  - `monte_carlo` para Europeas.
  - `asian_monte_carlo` para Asiáticas (usa `path_prefix` para promedios ya fijados).

Algoritmo (por path `i` y paso `t`, sin look-ahead):

1) Inputs causales disponibles:
   - `S_t^i`
   - `path_prefix^i = [S_0^i, ..., S_t^i]` (para asiáticas: fija promedios ya observados)
   - `r` y `sigma_t` del benchmark (en THESIS_FINAL: `r=0.03` fijo; `sigma_t` estimada por sección 8)

2) Outer simulation (GBM, un paso):

```
Z_{i,k} ~ N(0,1)              (k=1..outer_paths)
S_{t+1}^{i,k} = S_t^i * exp( (r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z_{i,k} )
```

3) Continuation value:
- El provider estima `C_{t+1}^{i,k}`.
- En provider `monte_carlo` / `asian_monte_carlo`, esto implica una simulación interna desde `t+1` hasta `T` (nested MC):

```
C_{t+1}^{i,k} ≈ E[ discounted_payoff | state at (t+1), path_prefix ]
```

4) Hedge ratio por covarianza muestral:

```
dS_{t+1}^{i,k} = S_{t+1}^{i,k} - S_t^i*exp(r*dt)
h_t^i = Cov_k(C_{t+1}^{i,k}, dS_{t+1}^{i,k}) / Var_k(dS_{t+1}^{i,k})
```

5) Conversión a trades:
- Se usa estado interno `last_delta` para producir trades:

```
a_t^i = h_t^i - h_{t-1}^i
```

Nota de model-mismatch (deliberada):
- En mundos GARCH/HMM, el benchmark LRM sigue simulando con **GBM** usando la `sigma` estimada.
- Esto representa un benchmark “clásico” fuera de especificación.

---

## 8. Estimación Causal de Parámetros para Benchmarks (Sigma / df / estados)

En THESIS_FINAL, **r es fijo** en todos los mundos:
- `r = 0.03`
- `benchmark_delta_r_mode = "none"` (no se estima r)

El único input extra “semi-modelado” para benchmarks es la `sigma` (y diagnósticos de colas/regímenes).

### 8.1 Datos disponibles para el benchmark en el paso t (causalidad)

Para cada path, en el paso de decisión `t` el benchmark puede usar:
- Pre-historia completa (L días): `S_{-L}..S_{-1}`
- Historial observado en la ventana hedge hasta `S_t`

Nunca puede usar:
- `S_{t+1}` ni retornos futuros
- parámetros verdaderos sampleados del generador

### 8.2 Estimación `sigma` con filtro GARCH sobre retornos observados (modo stepwise)

En W2–W4 se usa `benchmark_delta_sigma_mode="garch_context_stepwise"` con:
- `benchmark_delta_sigma_garch_fit_from_context=true`
- `benchmark_delta_sigma_context_days=L` (50 en W2–W4, 100 en W5)

Definimos retornos log y shocks anualizados:

```
lr_j = ln(S_j / S_{j-1})
eps_j = sqrt(252) * lr_j
```

Varianza inicial desde contexto:
- Se toma la serie `[S_{-L}..S_{-1}, S_0]` y se computan los `eps_j` asociados.
- Si hay al menos `min_obs` observaciones, se usa `sample_var = Var(eps)` (ddof=1 cuando aplica).
- Si no, se usa `default_sigma^2`.

Omega:

```
omega = (1 - alpha - beta) * sample_var
```

Filtro:

```
v <- sample_var
para cada eps del contexto:
  v <- omega + alpha*eps^2 + leverage*1_{eps<0}*eps^2 + beta*v
sigma_t0 = sqrt(v)
```

Stepwise (dentro del hedge):
- Para decisión en `t`, se actualiza con el retorno observado del paso anterior `eps_{t-1}`:

```
sigma_t = sqrt(v_t), con v_{t} actualizado solo con observaciones hasta t
```

Clips numéricos:
- `sigma_floor` para evitar 0
- `sigma_cap` opcional

### 8.3 Fit robusto de (alpha, beta, leverage) desde contexto (heurístico)

Cuando `benchmark_delta_sigma_garch_fit_from_context=true`, se estiman parámetros por path usando momentos del contexto:

- Persistencia proxy: autocorrelación lag-1 de `eps^2` => `ac1`
- Se mapea a un “persistence” `p`:

```
p = clip(0.15 + 0.80*ac1, persistence_floor, persistence_cap)
beta = 0.80*p
alpha = p - beta
```

- Leverage proxy: diferencia de varianza entre shocks negativos y positivos:

```
asym = (E[eps^2 | eps<0] - E[eps^2 | eps>=0]) / (E[eps^2 | eps<0] + E[eps^2 | eps>=0])
leverage = clip(0.20*asym, 0, leverage_cap)
```

- Se escala `(alpha,beta,leverage)` para cumplir estacionariedad:

```
alpha + beta + 2*leverage < 0.995
```

### 8.4 Fit de grados de libertad `ν` (Student-t) desde shocks estandarizados (diagnóstico)

En W3–W5 se habilita además:
- `benchmark_delta_student_t_fit_enabled=true`
- `benchmark_delta_student_t_mode="stepwise"`

Se construyen shocks estandarizados:

```
z_j = eps_j / sqrt(v_j)
```

Y se estima `ν` por método de momentos (kurtosis):

```
excess = E[z^4]/E[z^2]^2 - 3  (se trunca a >=0)
si excess > 0:
  nu_hat = 4 + 6/excess
si excess == 0:
  nu_hat = nu_cap (p.ej. 200)
```

Se clipea a `[df_floor, df_cap]`.

**Importante:**
- En W3–W4, `ν` se guarda como diagnóstico (`tables/student_t_df_simulated_stepwise.csv`) pero el benchmark BS sigue siendo BS: solo usa `sigma_t`.
- En W5, la estimación HMM+GARCH+Student sí usa `ν` para ajustar `sigma` (ver siguiente sección).

### 8.5 W5: estimador combinado `hmm_garch_student_context_stepwise`

En W5 se usa un estimador combinado que produce:
- `sigma_garch_stepwise` (GARCH puro)
- `student_t_df_stepwise` (ν por paso)
- `hmm_states_stepwise` (estado/regla de régimen)
- `sigma_stepwise` final (lo que se pasa a benchmark)

Regímenes (estilo HMM, inferencia heurística):
- Para cada path y paso `t`, se computan umbrales como cuantiles del historial `sigma_garch[:, :t+1]`.
- Estado `state_t` es la cantidad de umbrales superados por `sigma_garch_t` (0..k-1).
- Se cuentan transiciones `state_{t-1} -> state_t` y se suavizan con `smoothing`.

Multiplicadores por estado:
- Se mantiene la media de `sigma` por estado y la media global.
- Se define:

```
mult_state = clip(mean_sigma_state / mean_sigma_global, floor, cap)
```

Probabilidad de próximo estado:

```
p(next | state_t) ∝ counts(state_t, next) + smoothing
```

Multiplicador esperado:

```
mult_t = sum_next p(next|state_t) * mult_state[next]
sigma_hat_t = sigma_garch_t * mult_t
```

Ajuste por colas (usa ν):

```
tail_mult = clip( sqrt(ν / (ν-2)), 1.0, tail_cap )
sigma_final_t = sigma_hat_t * tail_mult
```

Esto produce una `sigma_final_t` más grande cuando el ajuste de colas indica heavy tails.

---

## 9. Escenarios THESIS_FINAL (Qué se corre exactamente)

Cada subescenario tiene `train/` y `compare/`. Los `.2` tienen además `calibration/`.

Comparaciones por claim (la “regla” pedida por la tesis se interpreta como: un compare por benchmark relevante, sin mezclar benchmarks en el mismo JSON):

- **Europea**
  - `compare/vs_bs.json`: Deep vs `DeltaHedgingAgent` (Black-Scholes).
  - `compare/vs_lrm_mc.json`: Deep vs `LocalRiskMinimizationAgent` (provider MC).
- **Asiática geométrica (solo en W1)**
  - `compare/vs_geo_bs.json`: Deep vs `GeometricAsianDeltaHedgingAgent`.
  - `compare/vs_lrm_mc.json`: Deep vs `LocalRiskMinimizationAgent` con provider `asian_monte_carlo`.
- **Asiática aritmética**
  - Solo `compare/vs_lrm_mc.json` (no hay benchmark “Delta/BS” útil para aritmética en este repositorio).

En escenarios `.2` (TC=1% + banda NI), hay versiones “banded”:
- `compare/*_banded.json`

Caso especial `1a.1'` (solo W1):
- `compare_mode="trained_only"`: compara solo modelos entrenados (LSTM CVaR50/90/99) sin benchmark.

Ejemplos de paths (source-of-truth son los JSON):
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1a_1_euro_tc0_cvar50/train/recurrent.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1a_1_euro_tc0_cvar50/train/lstm.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1a_1_euro_tc0_cvar50/compare/vs_bs.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1a_1_euro_tc0_cvar50/compare/vs_lrm_mc.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1a_1p_euro_tc0_lstm_cvar_sweep/compare/lstm_cvar_sweep.json`

---

## 10. No-Intervention Band (escenarios `.2`, TC=1%)

Objetivo:
- En TC=1%, los benchmarks “clásicos” (Delta/BS y LRM) suelen sobre-operar.
- En `.2` se reutilizan sus acciones del escenario `.1` (sin costos) y se aplica una banda “no-intervention” que reduce la frecuencia de trades.
- La banda se calibra por grilla para optimizar el benchmark (CVaR50).

### 10.1 Reuso de acciones

Se usa:
- `benchmark_actions_reuse_from_run`: apunta a un run `.1` existente (carpeta `actions_cache/`).
- `benchmark_actions_reuse_agent_name`: nombre del agente a buscar (p.ej. `DeltaHedgingAgent`).
- `benchmark_actions_reuse_apply_no_intervention=true`: aplica banda sobre la **trayectoria de posiciones**.

### 10.2 Regla de banda (modo percentage)

Sea `p_t` la posición target acumulada del benchmark (cumsum de trades). Se construye una posición ejecutada `e_t`:

```
e_0 = p_0
para t=1..N:
  diff = |p_t - e_{t-1}|
  scale = max(|p_t|, |e_{t-1}|, eps)
  hold si (diff/scale) < eta
  e_t = e_{t-1} si hold, sino p_t
```

Los trades ejecutados resultan de diferenciar:

```
a_0 = e_0
a_t = e_t - e_{t-1}   (t>=1)
```

### 10.3 Grilla obligatoria (tesis)

La grilla de búsqueda para `eta` es:
- Desde 0.5% a 10% inclusive
- Step 0.5%

Conjunto:

```
eta ∈ {0.005, 0.010, 0.015, ..., 0.100}
```

En cada carpeta `.2/calibration/` existen configs:
- `band_0005.json ... band_0100.json`
- `band_template.json` y `band_template_lrm.json` son solo plantillas (no se ejecutan automáticamente por el runner).

Cardinalidad (por carpeta de calibración):
- 20 runs reales: `band_0005..band_0100` (grilla 0.5%..10% step 0.5%).
- 2 templates: `band_template*.json` (no ejecutar).

Selección:
- Elegir el `eta` que minimiza el `cvar_50` del benchmark en `tables/empirical_risk_metrics.csv` para los runs de calibración.

---

## 11. Comparaciones, Plots y Métricas Guardadas

### 11.1 Qué se plotea

Por comparación (`compare_mode="benchmark_vs_targets"`):
- Histograma de `error` terminal por agente (en español, sin título).
- Scatter “acciones benchmark vs acciones agente” (por paso) para cada par benchmark-vs-agente.
- Plots de paths usados en la comparación (niveles y log-moneyness) generados por `run_scenarios_console.py`.

Caso especial CVaR sweep (trained-only):
- Un único plot overlay con 3 histogramas (CVaR50/90/99) con colores fijos:
  - CVaR50: verde `#2ca02c`
  - CVaR90: naranja `#ff7f0e`
  - CVaR99: azul `#1f77b4`

### 11.2 Rango automático del histograma (si no se fija en config)

Si `plot_min_x` y `plot_max_x` son null:
- Se toma el rango `[q0.1%, q99.9%]` sobre los errores finitos de todos los agentes del plot.
- Se agrega 5% de padding a cada lado.

### 11.3 Bootstrap

Por default se corre bootstrap i.i.d. con:
- `bootstrap_n_bootstraps=500`
- `bootstrap_confidence_level=0.95`
- `bootstrap_batch_size=100`

Estadísticos bootstrap calculados (por agente):
- Mean
- StdDev
- CVaR(50%)
- CVaR(90%)
- CVaR(95%)
- CVaR(99%)
- MAE
- WorstCase (mínimo)

### 11.4 Archivos por compare (para análisis offline sin re-run)

En cada `.../compare/<run>/` se guardan:

- `actions_cache/*.npy`: acciones (trades) por agente. Si se reusa banda, aparece `*_actions_reused_band.npy`.
- `plots/*.jpg`: histogramas y scatter de acciones.
- `tables/point_metrics.csv`: media y desvío (y losses auxiliares si aplica).
- `tables/empirical_risk_metrics.csv`: métricas empíricas incluyendo `cvar_50`, `cvar_90`, `cvar_95`, `cvar_99`.
- `tables/bootstrap_metrics_wide.csv`: tabla wide con CIs bootstrap.
- `tables/pairwise_terminal_stats.csv`: stats por par benchmark-vs-agente.
- `tables/calibration_manifest.csv`: manifiesto de calibración/parametrización del run.
- `run_metadata.json`: metadata resumida del run.
- `raw/` (payload completo):
  - `raw/eval_paths.npy`
  - `raw/eval_pre_history.npy` (si hay contexto)
  - `raw/per_path_sigma.npy` (vector o matriz stepwise si se estimó sigma)
  - `raw/per_path_r.npy` (en THESIS_FINAL: vector constante 0.03)
  - `raw/per_path_student_t_df.npy` (si se fiteó df)
  - `raw/hmm_states_stepwise.npy` (si aplica W5)
  - `raw/raw_payload_manifest.json`

---

## 12. Comandos de Ejecución (Runbook)

### 12.1 Correr un train individual

```
python examples/train_console.py THESIS_FINAL/<MUNDO>/<SUBESCENARIO>/train/<config_sin_json>
```

### 12.2 Correr un compare individual

```
python examples/compare_console.py THESIS_FINAL/<MUNDO>/<SUBESCENARIO>/compare/<config_sin_json>
```

### 12.3 Correr todas las corridas dentro de un mundo (secuencial)

Ejemplo:

```
python examples/run_scenarios_console.py THESIS_FINAL/W1_gbm_fixed --checkpoint-every-epochs 5
```

Comportamiento default del runner:
- Ejecuta primero todos los `train/*.json`, luego todos los `compare/*.json` (incluye `calibration/*.json`).
- Si el output ya está completo, saltea (no re-ejecuta).
- Si en compare falta solo bootstrap pero existe payload raw, corre solo bootstrap.
- `--force-run` fuerza re-ejecución completa ignorando artefactos existentes.

---

## 13. Checklist de Validez (para comité)

1) Los mundos y subescenarios incluidos son exactamente los definidos en THESIS_FINAL.
2) En todas las comparaciones estándar se usa `pricing_method="fixed"` (precio del primer agente por path).
3) En W2–W5, `r` está fijo a `0.03` y no se estima (benchmark_delta_r_mode="none").
4) La sigma del benchmark en W2–W4 se estima causalmente con `garch_context_stepwise` usando contexto + historia hasta `t`.
5) En W5, el benchmark usa `hmm_garch_student_context_stepwise` y guarda `sigma`, `df` y `states` stepwise.
6) En escenarios `.2`, el benchmark reusa acciones de `.1` y aplica banda NI calibrada en grilla 0.5%..10% step 0.5%.
7) Todas las métricas necesarias quedan persistidas (raw payload + tablas) para análisis offline.


---

## 14. Inventario Exhaustivo de Configs (THESIS_FINAL)

Esta sección enumera **todo** lo que se corre en THESIS_FINAL (sin agregar ni quitar escenarios), con paths relativos bajo `configs/runs/THESIS_FINAL/`.

Convenciones:
- `train/*.json`: entrenamientos (modelos guardados bajo `.../train/<run>/models/`).
- `compare/*.json`: comparaciones finales (artefactos bajo `.../compare/<run>/`).
- `calibration/*.json`: calibración de banda NI (solo escenarios `.2`).
- En `calibration/`:
  - Se ejecutan solo `band_0005..band_0100` (20 configs).
  - `band_template*.json` son plantillas (no ejecutar).

### W1_gbm_fixed

- `1a_1_euro_tc0_cvar50/`
  - `train/recurrent.json`
  - `train/lstm.json`
  - `compare/vs_bs.json`
  - `compare/vs_lrm_mc.json`
- `1a_1p_euro_tc0_lstm_cvar_sweep/`
  - `train/lstm_cvar90.json`
  - `train/lstm_cvar99.json`
  - `compare/lstm_cvar_sweep.json`
- `1a_2_euro_tc1_band/`
  - `train/recurrent.json`
  - `train/lstm.json`
  - `calibration/band_0005.json ... calibration/band_0100.json` (20)
  - `calibration/band_template.json` (template)
  - `calibration/band_template_lrm.json` (template)
  - `compare/vs_bs.json`
  - `compare/vs_bs_banded.json`
  - `compare/vs_lrm_mc.json`
  - `compare/vs_lrm_mc_banded.json`
- `1b_1_asian_geo_tc0_cvar50/`
  - `train/recurrent.json`
  - `train/lstm.json`
  - `compare/vs_geo_bs.json`
  - `compare/vs_lrm_mc.json`
- `1b_2_asian_geo_tc1_band/`
  - `train/recurrent.json`
  - `train/lstm.json`
  - `calibration/band_0005.json ... calibration/band_0100.json` (20)
  - `calibration/band_template.json` (template)
  - `calibration/band_template_lrm.json` (template)
  - `compare/vs_geo_bs.json`
  - `compare/vs_geo_bs_banded.json`
  - `compare/vs_lrm_mc.json`
  - `compare/vs_lrm_mc_banded.json`
- `1c_1_asian_arith_tc0_cvar50/`
  - `train/recurrent.json`
  - `train/lstm.json`
  - `compare/vs_lrm_mc.json`
- `1c_2_asian_arith_tc1_band/`
  - `train/recurrent.json`
  - `train/lstm.json`
  - `calibration/band_0005.json ... calibration/band_0100.json` (20)
  - `calibration/band_template.json` (template)
  - `calibration/band_template_lrm.json` (template)
  - `compare/vs_lrm_mc.json`
  - `compare/vs_lrm_mc_banded.json`

### W2_gbm_random_ctx50

- `1a_1_euro_tc0_cvar50/`
  - `train/recurrent.json`
  - `train/lstm.json`
  - `compare/vs_bs.json`
  - `compare/vs_lrm_mc.json`
- `1a_2_euro_tc1_band/`
  - `train/recurrent.json`
  - `train/lstm.json`
  - `calibration/band_0005.json ... calibration/band_0100.json` (20)
  - `calibration/band_template.json` (template)
  - `calibration/band_template_lrm.json` (template)
  - `compare/vs_bs.json`
  - `compare/vs_bs_banded.json`
  - `compare/vs_lrm_mc.json`
  - `compare/vs_lrm_mc_banded.json`
- `1c_1_asian_arith_tc0_cvar50/`
  - `train/recurrent.json`
  - `train/lstm.json`
  - `compare/vs_lrm_mc.json`
- `1c_2_asian_arith_tc1_band/`
  - `train/recurrent.json`
  - `train/lstm.json`
  - `calibration/band_0005.json ... calibration/band_0100.json` (20)
  - `calibration/band_template.json` (template)
  - `calibration/band_template_lrm.json` (template)
  - `compare/vs_lrm_mc.json`
  - `compare/vs_lrm_mc_banded.json`

### W3_garch_t_random_ctx50

- `1a_1_euro_tc0_cvar50/`
  - `train/recurrent.json`
  - `train/lstm.json`
  - `compare/vs_bs.json`
  - `compare/vs_lrm_mc.json`
- `1a_2_euro_tc1_band/`
  - `train/recurrent.json`
  - `train/lstm.json`
  - `calibration/band_0005.json ... calibration/band_0100.json` (20)
  - `calibration/band_template.json` (template)
  - `calibration/band_template_lrm.json` (template)
  - `compare/vs_bs.json`
  - `compare/vs_bs_banded.json`
  - `compare/vs_lrm_mc.json`
  - `compare/vs_lrm_mc_banded.json`
- `1c_1_asian_arith_tc0_cvar50/`
  - `train/recurrent.json`
  - `train/lstm.json`
  - `compare/vs_lrm_mc.json`
- `1c_2_asian_arith_tc1_band/`
  - `train/recurrent.json`
  - `train/lstm.json`
  - `calibration/band_0005.json ... calibration/band_0100.json` (20)
  - `calibration/band_template.json` (template)
  - `calibration/band_template_lrm.json` (template)
  - `compare/vs_lrm_mc.json`
  - `compare/vs_lrm_mc_banded.json`

### W4_garch_t_tail_ctx50

- `1a_1_euro_tc0_cvar50/`
  - `train/recurrent.json`
  - `train/lstm.json`
  - `compare/vs_bs.json`
  - `compare/vs_lrm_mc.json`
- `1a_2_euro_tc1_band/`
  - `train/recurrent.json`
  - `train/lstm.json`
  - `calibration/band_0005.json ... calibration/band_0100.json` (20)
  - `calibration/band_template.json` (template)
  - `calibration/band_template_lrm.json` (template)
  - `compare/vs_bs.json`
  - `compare/vs_bs_banded.json`
  - `compare/vs_lrm_mc.json`
  - `compare/vs_lrm_mc_banded.json`
- `1c_1_asian_arith_tc0_cvar50/`
  - `train/recurrent.json`
  - `train/lstm.json`
  - `compare/vs_lrm_mc.json`
- `1c_2_asian_arith_tc1_band/`
  - `train/recurrent.json`
  - `train/lstm.json`
  - `calibration/band_0005.json ... calibration/band_0100.json` (20)
  - `calibration/band_template.json` (template)
  - `calibration/band_template_lrm.json` (template)
  - `compare/vs_lrm_mc.json`
  - `compare/vs_lrm_mc_banded.json`

### W5_hmm3_garch_t_ctx100

- `1a_1_euro_tc0_cvar50/`
  - `train/recurrent.json`
  - `train/lstm.json`
  - `compare/vs_bs.json`
  - `compare/vs_lrm_mc.json`
- `1a_2_euro_tc1_band/`
  - `train/recurrent.json`
  - `train/lstm.json`
  - `calibration/band_0005.json ... calibration/band_0100.json` (20)
  - `calibration/band_template.json` (template)
  - `calibration/band_template_lrm.json` (template)
  - `compare/vs_bs.json`
  - `compare/vs_bs_banded.json`
  - `compare/vs_lrm_mc.json`
  - `compare/vs_lrm_mc_banded.json`
- `1c_1_asian_arith_tc0_cvar50/`
  - `train/recurrent.json`
  - `train/lstm.json`
  - `compare/vs_lrm_mc.json`
- `1c_2_asian_arith_tc1_band/`
  - `train/recurrent.json`
  - `train/lstm.json`
  - `calibration/band_0005.json ... calibration/band_0100.json` (20)
  - `calibration/band_template.json` (template)
  - `calibration/band_template_lrm.json` (template)
  - `compare/vs_lrm_mc.json`
  - `compare/vs_lrm_mc_banded.json`
