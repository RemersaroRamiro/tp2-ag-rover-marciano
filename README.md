# TP2: Diseño evolutivo de un rover marciano con Algoritmos Genéticos

Inteligencia Artificial, UTN FRBA, Grupo 7.

Un Algoritmo Genético implementado con DEAP busca el diseño de un rover de seis ruedas con suspensión rocker-bogie
(chasis, suspensión, ruedas, motores, batería y parámetros de manejo) que recorra la mayor distancia posible sobre una
pista construida con el relieve real del cráter Jezero, respetando límites de masa, estabilidad, tamaño y altura libre.
Cada individuo es un cromosoma de 15 genes reales en [0, 1]. La aptitud, que se maximiza, es la distancia media que el
rover recorre en una simulación física (MuJoCo) sobre los terrenos de la generación. Las restricciones se tratan con
las reglas de factibilidad de Deb.

## Contenido

| Carpeta o archivo | Qué contiene |
|---|---|
| `TP2_AG_Rover.ipynb` | Notebook con las secciones de la demo de la cátedra. Está guardado ya ejecutado con los resultados del informe. |
| `logs-corridas/` | Las corridas del informe: `generations.csv` (log por generación), `run.json` (configuración y validación) y los gráficos de aptitud y de diversidad de cada una. |
| `codigo/` | El código del Algoritmo Genético y del simulador, con los datos de la pista. |
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
  y, en la sección 0, descomentar las líneas que clonan el repositorio e instalan DEAP. El notebook corre en modo
  "solo resultados" y no necesita MuJoCo.
- **Local:** con [uv](https://docs.astral.sh/uv/) instalado, `uv sync` y luego
  `uv run --with jupyterlab jupyter lab TP2_AG_Rover.ipynb`.
- **Tabla comparativa en la terminal:** `uv run python codigo/evolve.py --summarize logs-corridas`.

## Reproducir una corrida

`uv sync` y luego, desde la carpeta del repositorio:

| Corrida | Comando |
|---|---|
| C1 | `uv run python codigo/evolve.py --preset normal --seed 1 --replay none --out runs/c1-base` |
| C2 | `uv run python codigo/evolve.py --preset normal --seed 1 --replay none --selection roulette --out runs/c2-ruleta` |
| C3 | `uv run python codigo/evolve.py --preset normal --seed 1 --replay none --crossover binomial --out runs/c3-binomial` |
| C4 | `uv run python codigo/evolve.py --preset normal --seed 1 --replay none --no-adaptive-mutation --out runs/c4-sigma-fija` |
| C5 | `uv run python codigo/evolve.py --preset normal --seed 1 --replay none --elite-count 0 --out runs/c5-sin-elite` |
| Aleatoria | `uv run python codigo/evolve.py --preset normal --seed 1 --replay none --random-search --out runs/random-search` |

Cada corrida del AG son 1800 simulaciones y tardó entre 39 y 48 minutos con 16 procesos en paralelo; la búsqueda
aleatoria, unos 17 minutos. `uv run python codigo/evolve.py --help` lista todos los parámetros.

## Mapa del código

| Parte del AG | Archivo |
|---|---|
| Cromosoma: los 15 genes y su decodificación en valores físicos | `codigo/genome_spec.py`, `codigo/rover_params.py` |
| Función de aptitud y reglas de Deb | `codigo/fitness.py` |
| Operadores genéticos en DEAP (selección, cruzamiento, mutación), elitismo y ciclo de generaciones | `codigo/ga.py` |
| Corrida completa, validación en terrenos nuevos y criterio de éxito | `codigo/evolve.py` |
| Búsqueda aleatoria (línea de base) | `codigo/baselines.py` |
| Simulación de cada rover (MuJoCo): pista, rocas, rover, controlador y energía | `codigo/evaluate.py` y los módulos que importa (`track_*.py`, `mars_*.py`, `rover*.py`, `controller.py`, `energy.py`, `scene.py`, `headless.py`) |
| Gráficos y tablas del notebook | `codigo/plot_run.py`, `codigo/notebook_support.py` |

## Créditos

El relieve de la pista es un recorte del modelo digital de elevación HiRISE del sitio de aterrizaje del Mars 2020
(USGS Astrogeology); ver `codigo/data/track/CREDITS.md`. Las texturas del terreno provienen de imágenes de la NASA;
ver `codigo/assets/CREDITS.md`. Para la implementación se usó Claude Code (Anthropic) como asistente de programación.
