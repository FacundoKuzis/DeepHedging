# Plan de reescritura de `main_tesis.tex` para THESIS_FINAL

Este plan detalla la etapa inicial de edición del documento. El foco está en el título, el resumen y las primeras cinco secciones. El objetivo es alinear el texto con THESIS_FINAL y eliminar material que ya no corresponde.

## Alcance de esta etapa
- Nuevo título y ajustes en la portada
- Resumen
- Sección Introducción
- Sección Descripción del Problema
- Sección Aprendizaje por refuerzo
- Sección El modelo de Deep Hedging
- Sección Experimentos numéricos

## Fuera de alcance por ahora
- Sección Resultados
- Sección Conclusiones
- Tablas y figuras que dependan de corridas todavía incompletas

## Restricciones de contenido
- WaveNet se elimina por completo
- No se explica LSTM en esta versión
- No se explica Longstaff Schwartz ni proveedores LSMC
- No se incluye Monte Carlo con Control Variates
- Para asiáticas aritméticas el benchmark relevante es Local Risk Minimization con proveedor Monte Carlo
- El texto no discute el código ni herramientas de software

## Restricciones de estilo para la redacción
- Prosa académica directa
- Evitar construcciones de contraste repetidas del tipo no solo X sino Y
- Evitar la raya larga
- Evitar el uso de dos puntos en la prosa

## Criterios de éxito de esta etapa
- El documento compila con `build_tesis.ps1`
- Las primeras secciones no hacen referencia a WaveNet, LSTM, Control Variates ni Longstaff Schwartz
- Los supuestos del diseño experimental coinciden con THESIS_FINAL
- El resumen y la introducción describen el alcance actual sin depender de resultados numéricos

---

## 1. Nuevo título y portada

### 1.1 Objetivo del título
El título debe reflejar cobertura dinámica en tiempo discreto con fricciones y con dinámicas de volatilidad más realistas que GBM fijo. Debe admitir que se estudian opciones europeas y asiáticas.

### 1.2 Opciones de título recomendadas
- Deep Hedging en tiempo discreto con fricciones y regímenes de volatilidad
- Deep Hedging con contexto causal para cobertura de opciones europeas y asiáticas
- Cobertura dinámica de opciones bajo volatilidad con colas y cambios de régimen mediante Deep Hedging

### 1.3 Cambios concretos en la portada
- Actualizar el título principal
- Revisar la fecha si se busca coherencia con el estado actual del trabajo
- Hacer opcional el logo mediante `\\IfFileExists` para evitar fallas de compilación si el archivo no está

---

## 2. Resumen

### 2.1 Objetivo del resumen
Explicar el problema, el método y el diseño de evaluación. Evitar reportar métricas o conclusiones cuantitativas mientras falten corridas.

### 2.2 Estructura sugerida
- Párrafo 1 con motivación y problema de cobertura con rebalanceo discreto y costos de transacción
- Párrafo 2 con el enfoque Deep Hedging y el conjunto de mundos THESIS_FINAL
- Párrafo 3 con benchmarks y métricas de evaluación, incluyendo CVaR y bootstrap, sin números

### 2.3 Puntos que deben aparecer
- Evaluación en tiempo discreto con costos proporcionales
- Comparación con delta hedging y con Local Risk Minimization
- Uso de historia causal en mundos con parámetros no observables por trayectoria

---

## 3. Introducción

### 3.1 Objetivo de la sección
Enmarcar el problema, justificar el enfoque, delimitar el aporte y guiar la lectura del documento.

### 3.2 Contenido a reescribir
- Motivar por qué la cobertura continua idealizada no es una referencia suficiente cuando hay rebalanceo discreto y fricciones
- Justificar la necesidad de evaluar en entornos donde la volatilidad cambia en el tiempo, presenta colas y puede alternar regímenes
- Introducir la idea de contexto causal como sustituto de parámetros no observables

### 3.3 Referencias recientes para una mención breve
Incluir pocas y de forma breve para mostrar continuidad con literatura reciente
- Cao, Cui y Liu 2020 como ejemplo de deep hedging en entornos tipo GARCH
- Neagu, Godin y Kosseim 2025 como ejemplo de RL aplicado a cobertura bajo GJR GARCH
Opcionales si encajan con el hilo
- Raj 2025 o Kop 2025 para discusión de decisiones adaptativas con señales de regímenes mediante HMM
- Tan, Roberts y Zohren 2024 o Stone 2019 para discutir uso de convoluciones causales en tareas financieras

### 3.4 Preguntas de investigación
Definir preguntas sin depender de resultados
- Robustez de Deep Hedging al pasar de GBM fijo a mundos con heterogeneidad, colas y regímenes
- Comparación frente a benchmarks cuando hay costos de transacción
- Aporte del contexto causal en mundos con parámetros por trayectoria

### 3.5 Estructura del documento
Actualizar el párrafo de organización del trabajo
- Eliminar referencias a WaveNet y a explicaciones extensas de arquitecturas no utilizadas
- Evitar adelantar conclusiones de resultados

---

## 4. Descripción del Problema

### 4.1 Ajuste del contenido base
Mantener definiciones y recortar lo no usado
- Opciones y cobertura
- Modelo Black Scholes Merton y delta hedging como benchmark
- Limitaciones relevantes para el trabajo, tiempo discreto y fricciones

### 4.2 Derivados y payoffs incluidos
Mantener la definición de los tres instrumentos
- Call europea
- Call asiática geométrica
- Call asiática aritmética

### 4.3 Métodos numéricos y benchmarks
En esta etapa solo se describe lo que se usa
- Monte Carlo como herramienta para aproximar expectativas condicionales en benchmarks
- Local Risk Minimization como benchmark basado en covarianza y varianza de un paso
- Evitar Longstaff Schwartz y evitar Control Variates

### 4.4 Acciones concretas de limpieza en esta sección
Buscar y eliminar o mover a apéndice
- Sub-secciones que expliquen Control Variates
- Sub-secciones que expliquen Longstaff Schwartz o LSMC
Si se conserva material como referencia, dejarlo fuera del cuerpo principal y aclarar que no se usa en los experimentos actuales.

---

## 5. Aprendizaje por refuerzo

### 5.1 Objetivo de la sección
Dar el mínimo necesario para interpretar Deep Hedging como un problema de control estocástico en tiempo discreto.

### 5.2 Recorte recomendado del contenido
- Mantener la formulación de política, secuencia de estados, acciones y recompensas
- Mantener una explicación breve de métodos de gradiente de política
- Q learning puede quedar como mención breve o moverse a un apéndice si no aporta a la lectura

### 5.3 Redes neuronales en esta etapa
Eliminar lo que no se usa y explicar solo lo necesario
- Quitar WaveNet por completo
- Quitar explicación detallada de LSTM
- Mantener una descripción general de una política paramétrica que usa variables observables y contexto causal
- Si se explica el contexto, describir convoluciones causales uno dimensionales como extractor de características, sin detallar arquitecturas específicas

---

## 6. El modelo de Deep Hedging

### 6.1 Notación y grilla temporal
Alinear con THESIS_FINAL
- Definir N, T, dt con 252 días hábiles
- Aclarar que el log es log natural
- Usar log moneyness como variable de escala para el precio

### 6.2 Estado, acción y dinámica de portafolio
Explicar en términos económicos y matemáticos
- Estado observable en cada decisión, precio transformado, tiempo a vencimiento, posición previa
- Acción como ajuste de la posición en el subyacente
- Restricción de la posición en rango compatible con calls cuando aplique, por ejemplo mediante una función sigmoidal

### 6.3 Costos, cuenta de caja y PnL
Describir claramente
- Costos proporcionales de transacción
- Dinámica de cuenta de caja con acumulación a tasa libre de riesgo
- Definición de valor terminal del portafolio, PnL y descuento

### 6.4 Objetivo de entrenamiento
- Función objetivo basada en CVaR con nivel 50 por ciento
- Interpretación de por qué CVaR es relevante en cobertura

### 6.5 Regla de precio usada para evaluar el error
Describir la comparación de forma consistente
- Un único precio por trayectoria dentro de un mismo experimento para comparar estrategias

---

## 7. Experimentos numéricos

### 7.1 Mundos THESIS_FINAL
Describir en términos conceptuales
- W1 GBM con parámetros fijos
- W2 GBM con sigma por trayectoria y con contexto causal
- W3 GARCH con innovaciones t
- W4 W3 con shocks de cola
- W5 HMM con tres estados sobre GARCH con innovaciones t

### 7.2 Grilla de escenarios
Definir qué se compara sin incluir resultados
- Derivados incluidos por mundo
- Costos con TC igual a 0 y TC igual a 1 por ciento
- Benchmarks por derivado, delta hedging y Local Risk Minimization
- Métricas reportables, media, desvío, CVaR, peor caso, MAE
- Intervalos de confianza mediante bootstrap

### 7.3 Banda de no intervención con costos
Explicar motivo y regla
- En presencia de costos, benchmarks clásicos tienden a operar con frecuencia excesiva
- Aplicar una banda de no intervención sobre la trayectoria de posiciones del benchmark
- Calibrar el umbral por grilla usando un criterio de riesgo del benchmark, por ejemplo CVaR 50

### 7.4 Qué se documentará cuando estén los resultados
Dejar asentado el formato sin completar números
- Tablas de métricas con intervalos de confianza
- Figuras de distribución del error
- Medidas de intensidad de trading cuando sea relevante

---

## 8. Bibliografía

### 8.1 Referencias canónicas necesarias
Agregar bibliografía para sustentar el marco
- ARCH y GARCH
- HMM y cambios de régimen
- Local Risk Minimization y cobertura de varianza
- No trade regions bajo costos de transacción

### 8.2 Referencias recientes para la introducción
Agregar pocas referencias recientes con una mención breve
- Cao, Cui y Liu 2020
- Neagu, Godin y Kosseim 2025
Opcionales si se usan en el texto
- Raj 2025 o Kop 2025
- Tan, Roberts y Zohren 2024 o Stone 2019

### 8.3 Limpieza de bibliografía
- Eliminar entradas y citas asociadas a WaveNet
- Eliminar referencias a Control Variates y Longstaff Schwartz del cuerpo principal si no se usan

---

## 9. Limpieza LaTeX para asegurar compilación en esta etapa
- Resolver imágenes faltantes en portada y en secciones 1 a 5
- Si hay figuras útiles pero no disponibles, envolverlas con `\\IfFileExists` y dejar una nota interna para completarlas luego
- Evitar que la compilación dependa de figuras de la sección de resultados mientras esa sección no se trabaja

