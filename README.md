# TP2: Diseño evolutivo de un rover marciano con Algoritmos Genéticos

Inteligencia Artificial, UTN FRBA, Grupo 7.

Un Algoritmo Genético implementado con DEAP busca el diseño de un rover de seis ruedas con suspensión rocker-bogie
(chasis, suspensión, ruedas, motores, batería y parámetros de manejo) que recorra la mayor distancia posible sobre una
pista construida con el relieve real del cráter Jezero, respetando límites de masa, estabilidad, tamaño y altura libre.
Cada individuo es un cromosoma de 15 genes reales en [0, 1]. La aptitud, que se maximiza, es la distancia media que el
rover recorre en una simulación física (MuJoCo) sobre los terrenos de la generación. Las restricciones se tratan con
las reglas de factibilidad de Deb.

## Video

https://github.com/user-attachments/assets/8bbe9b0d-7f06-4bfe-b936-95525c10cda5

La corrida `c3-binomial` en 3D: la población de las generaciones 0, 15 y 29 recorriendo la pista y la evolución de la
aptitud máxima. El visor 3D con el que se grabó no forma parte de este repositorio.

## Contenido

| Carpeta o archivo | Qué contiene |
|---|---|
| `TP2_AG_Rover.ipynb` | El Algoritmo Genético completo, explicado paso a paso en seis secciones. Está guardado ya ejecutado con los resultados del informe. |
| `logs-corridas/` | Las corridas del informe: `generations.csv` (log por generación), `run.json` (configuración y validación) y los gráficos de aptitud y de diversidad de cada una. |
| `codigo/` | El simulador (caja negra para el notebook) y el programa de terminal que generó las corridas, con los datos de la pista. Cada archivo empieza con una explicación en español de para qué existe. |
| `pyproject.toml`, `uv.lock` | Dependencias: Python 3.12, DEAP, MuJoCo, NumPy y Matplotlib. |

## Corridas del informe

Todas usan 30 individuos, 30 generaciones, 2 terrenos por generación y la semilla 1, así que comparten la población
inicial y los terrenos. Cada una cambia un solo método respecto de la base. El mejor diseño se valida en 5 terrenos
que no se usaron en la evolución, y el problema se considera resuelto si llega a la meta en al menos 3 de ellos.

| Corrida | Cambio respecto de la base | Llegadas en validación | ¿Resuelve? |
|---|---|---|---|
| `c1-base` | Base: torneo k = 3, BLX-α 0,3, mutación gaussiana con σ adaptativa, elitismo 2 | 3/5 | Sí |
| `c2-ruleta` | Selección por ruleta | 3/5 | Sí |
| `c3-binomial` | Cruzamiento binomial | 5/5 | Sí |
| `c4-sigma-fija` | Mutación con σ fija | 3/5 | Sí |
| `c5-sin-elite` | Sin elitismo | 4/5 | Sí |
| `random-search` | Búsqueda aleatoria con el mismo presupuesto (línea de base) | 1/5 | No |

El diseño de referencia, de escala similar al Perseverance, no llega a la meta en ninguno de los 5 terrenos.

## Ver los resultados

- **GitHub:** `TP2_AG_Rover.ipynb` se ve directamente en el navegador, con las tablas y los gráficos ya ejecutados.
- **Google Colab:** [abrir el notebook en Colab](https://colab.research.google.com/github/RemersaroRamiro/tp2-ag-rover-marciano/blob/main/TP2_AG_Rover.ipynb)
  y usar **Entorno de ejecución → Ejecutar todas**. La primera celda clona el repositorio e instala DEAP; el notebook
  corre en modo resultados y no necesita MuJoCo.
- **Local:** con [uv](https://docs.astral.sh/uv/) instalado, `uv sync` y luego
  `uv run --with jupyterlab jupyter lab TP2_AG_Rover.ipynb`.
- **Tabla comparativa en la terminal:** `uv run python codigo/evolve.py --summarize logs-corridas`.

## Reproducir una corrida

`uv sync` y luego, desde la carpeta del repositorio:

| Corrida | Comando |
|---|---|
| C1 | `uv run python codigo/evolve.py --preset normal --seed 1 --out runs/c1-base` |
| C2 | `uv run python codigo/evolve.py --preset normal --seed 1 --selection roulette --out runs/c2-ruleta` |
| C3 | `uv run python codigo/evolve.py --preset normal --seed 1 --crossover binomial --out runs/c3-binomial` |
| C4 | `uv run python codigo/evolve.py --preset normal --seed 1 --no-adaptive-mutation --out runs/c4-sigma-fija` |
| C5 | `uv run python codigo/evolve.py --preset normal --seed 1 --elite-count 0 --out runs/c5-sin-elite` |
| Aleatoria | `uv run python codigo/evolve.py --preset normal --seed 1 --random-search --out runs/random-search` |

Cada corrida del AG son 1800 simulaciones y tardó entre 39 y 48 minutos con 16 procesos en paralelo; la búsqueda
aleatoria, unos 17 minutos. `uv run python codigo/evolve.py --help` lista todos los parámetros.

## Mapa del código

El Algoritmo Genético se lee en el notebook; `codigo/` es el simulador y el programa de terminal.

| Parte del AG | En el notebook | En `codigo/` (lo que usó `evolve.py` para las corridas) |
|---|---|---|
| Cromosoma: los 15 genes y su decodificación en valores físicos | sección 3, "Estructura del cromosoma" | `genome_spec.py` |
| Función de aptitud y reglas de Deb | sección 3, "Función de aptitud" | `fitness.py` |
| Operadores genéticos en DEAP (selección, cruzamiento, mutación), reflexión, σ adaptativa y elitismo | sección 3, "Operadores genéticos" | `ga.py` |
| Ciclo de generaciones | sección 4, "Ciclo de generaciones" | `ga.py` |
| Corrida completa, validación en terrenos nuevos y criterio de éxito | sección 4, "Ejecución de la corrida" | `evolve.py` |
| Búsqueda aleatoria (línea de base) | resultados en la sección 4 | `baselines.py` |
| Simulación de cada rover (MuJoCo, sin gráficos): pista, rocas, rover, controlador y energía | caja negra | `evaluate.py` y los módulos que importa (`headless.py`, `track_*.py`, `mars_params.py`, `rover*.py`, `controller.py`, `energy.py`) |
| Gráficos | sección 5 | `plot_run.py` (los PNG de `logs-corridas/`) |

La celda "Comprobación" de la sección 4 demuestra que el AG del notebook es el mismo que el de `codigo/ga.py`: con la
misma semilla generan exactamente las mismas poblaciones, y la aptitud del notebook coincide con la registrada en las
corridas.

## Créditos

El relieve de la pista es un recorte del modelo digital de elevación HiRISE del sitio de aterrizaje del Mars 2020
(USGS Astrogeology); ver `codigo/data/track/CREDITS.md`. Para la implementación se usó Claude Code (Anthropic) como
asistente de programación.
