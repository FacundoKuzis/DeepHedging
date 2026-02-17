# DeepHedging

Repositorio para experimentar estrategias de cobertura (deep hedging) con:
- Agentes entrenables (`Simple`, `Recurrent`, `LSTM`, `GRU`, `WaveNet`)
- Agentes analíticos/no entrenables (delta hedging BS y variantes asiáticas)
- Simulación de paths (`GBM`, `Heston`)
- Evaluación de error terminal y estadísticas de riesgo

## Instalación rápida

```bash
python install.py
```

`install.py` ahora crea el entorno con **Python 3.11** de forma obligatoria.
Si ya tenías un `env` creado con Python 3.12, elimínalo y vuelve a correr `python install.py`.

o manualmente:

```bash
py -3.11 -m venv env
env\Scripts\python -m pip install -e .
```

## Script más fácil de correr (sin CLI)

Editar variables globales en:

`examples/run_with_globals.py`

y ejecutar:

```bash
python examples/run_with_globals.py
```

Variables principales:
- `RUN_MODE`: `train`, `evaluate`, `train_and_evaluate`
- `INSTRUMENT_NAME`: `GBMStock` o `TimeGANStock`
- `MAIN_AGENT_NAME`
- `CONTINGENT_CLAIM_NAME`
- `TRAIN_PATHS`, `N_EPOCHS`, `BATCH_SIZE`
- `COMPARE_AGENT_NAMES`
- `MODEL_NAME`, `MODELS_DIR`, `OPTIMIZERS_DIR`

### TimeGAN (sin key/licencia corporativa)
Este repo integra simulación con `ydata-synthetic` (open source). No usa `ydata-sdk` ni token de plataforma.

Requisitos:
- `pip install ydata-synthetic`
- Python 3.11 para todo el repo (setup validado para evitar conflictos entre dependencias)
- `yfinance` queda acotado a `<0.2.27` por compatibilidad de dependencias con `ydata-synthetic`

Si trabajas en Python 3.12, la instalación del proyecto fallará explícitamente por `python_requires`.

### Configuración rápida TimeGAN en `run_with_globals.py`
1. Poner `INSTRUMENT_NAME = "TimeGANStock"`
2. Configurar bloque `TIMEGAN_*`:
   - ticker/rango (`TIMEGAN_TICKER`, `TIMEGAN_START_DATE`, `TIMEGAN_END_DATE`, `TIMEGAN_INTERVAL`)
   - cache CSV/modelo (`TIMEGAN_CSV_PATH`, `TIMEGAN_MODEL_PATH`)
   - entrenamiento (`TIMEGAN_TRAIN_EPOCHS`, `TIMEGAN_BATCH_SIZE`, etc.)
   - recomendación: `TIMEGAN_TRAINING_TARGET="log_returns"` y `TIMEGAN_MATCH_RETURN_MOMENTS=True` para estabilidad estadística
3. Ejecutar `python examples/run_with_globals.py`

### Demo aislada del simulador
Para entrenar/cargar solo el simulador y generar plots + métricas:

```bash
python examples/timegan_simulator_demo.py
```

Salidas:
- Plots: `assets/plots/timegan/*.pdf`
- Métricas: `assets/csvs/timegan/summary_metrics.csv`

## Scripts CLI

- Entrenamiento: `examples/train.py`
- Evaluación: `examples/evaluate_agents.py`
- Comparar estrategias (trayectoria): `examples/compare_hedging_strategies.py`
- Bootstrap: `examples/bootstrap_confidence_intervals.py`
