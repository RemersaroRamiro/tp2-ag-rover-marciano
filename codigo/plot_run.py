"""
plot_run.py: dibuja los gráficos de una corrida a partir de su CSV.

Para qué existe
    Los gráficos del informe salen solo de generations.csv, sin simulador ni MuJoCo, así que se pueden
    volver a dibujar en cualquier momento a partir de una corrida guardada.

Qué contiene
    - read_generations: lee las columnas del CSV.
    - fitness_figure: el gráfico "Evolución de la Aptitud" (máximo y promedio por generación).
    - diversity_figure: la diversidad de la población y el paso de mutación σ, ambos en unidades de gen.
    - plot_run: guarda los dos PNG junto al CSV (generations_fitness.png y generations_diversity.png).

Cómo funciona
    Los puntos no finitos (-inf o NaN) se omiten de la curva y una nota al pie indica cuántos son.
    Los textos de los gráficos están en español porque van al informe.
    uv run python codigo/plot_run.py runs/<corrida>

Dónde encaja
    Etapa: resultados. No importa otros archivos del proyecto; evolve.py lo llama al terminar cada
    corrida.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

CSV_NAME = "generations.csv"
REQUIRED_COLUMNS = ("generation", "best", "mean")

FITNESS_TITLE = "Evolución de la Aptitud"
DIVERSITY_TITLE = "Diversidad y paso de mutación"
GENERATION_LABEL = "Generación"
FITNESS_LABEL = "Aptitud"
DIVERSITY_Y_LABEL = "Valor (unidades de gen, 0 a 1)"
BEST_LABEL = "Máximo"
MEAN_LABEL = "Promedio"
DIVERSITY_LABEL = "Diversidad (desvío medio por gen)"
SIGMA_LABEL = "σ de mutación"

FIGSIZE = (8.0, 5.0)
DPI = 150


@dataclass(frozen=True)
class RunCharts:
    fitness_png: Path
    diversity_png: Path
    fitness_figure: Figure
    diversity_figure: Figure
    skipped: dict[str, int]


def _number(text: str | None) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def read_generations(csv_path: str | Path) -> dict[str, list[float]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in columns]
        if missing:
            raise ValueError(f"{csv_path}: missing column(s) {', '.join(missing)}; expected the columns of ga.CSV_COLUMNS")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{csv_path}: no generation rows")
    data = {c: [_number(row.get(c)) for row in rows] for c in columns}
    data["generation"] = [int(g) for g in data["generation"]]
    return data


def _draw_series(ax, generations, values, label) -> int | None:
    points = [(g, v) for g, v in zip(generations, values) if math.isfinite(v)]
    if not points:
        return None
    xs, ys = zip(*points)
    ax.plot(xs, ys, label=label, marker="o" if len(points) == 1 else None)
    return len(values) - len(points)


def _chart(series: list[tuple[str, list[float]]], generations, title, y_label) -> tuple[Figure, dict[str, int]]:
    fig = Figure(figsize=FIGSIZE)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    skipped, missing = {}, []
    for label, values in series:
        left_out = _draw_series(ax, generations, values, label)
        if left_out is None:
            missing.append(label)
        elif left_out:
            skipped[label] = left_out
    ax.set_title(title)
    ax.set_xlabel(GENERATION_LABEL)
    ax.set_ylabel(y_label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True)
    if ax.get_lines():
        ax.legend()
    notes = []
    if skipped:
        notes.append("Se omitieron puntos no finitos (simulaciones fallidas): "
                     + ", ".join(f"{label} {n}" for label, n in skipped.items()))
    notes += [f"{label}: sin datos" for label in missing]
    if notes:
        ax.annotate("; ".join(notes), xy=(0, 0), xycoords="axes fraction", xytext=(0, -42), textcoords="offset points",
                    fontsize=8, color="0.3", va="top")
    fig.tight_layout()
    return fig, skipped


def _titled(title: str, suffix: str) -> str:
    return f"{title} ({suffix})" if suffix else title


def fitness_figure(data: dict[str, list[float]], title_suffix: str = "") -> tuple[Figure, dict[str, int]]:
    series = [(BEST_LABEL, data["best"]), (MEAN_LABEL, data["mean"])]
    return _chart(series, data["generation"], _titled(FITNESS_TITLE, title_suffix), FITNESS_LABEL)


def diversity_figure(data: dict[str, list[float]], title_suffix: str = "") -> tuple[Figure, dict[str, int]]:
    nothing = [math.nan] * len(data["generation"])
    series = [(DIVERSITY_LABEL, data.get("diversity", nothing)), (SIGMA_LABEL, data.get("mutation_sigma", nothing))]
    return _chart(series, data["generation"], _titled(DIVERSITY_TITLE, title_suffix), DIVERSITY_Y_LABEL)


def plot_run(csv_path: str | Path, out_dir: str | Path | None = None, title_suffix: str = "") -> RunCharts:
    csv_path = Path(csv_path)
    folder = Path(out_dir) if out_dir is not None else csv_path.parent
    folder.mkdir(parents=True, exist_ok=True)
    data = read_generations(csv_path)
    fitness, skipped_fitness = fitness_figure(data, title_suffix)
    diversity, skipped_diversity = diversity_figure(data, title_suffix)
    fitness_png, diversity_png = folder / f"{csv_path.stem}_fitness.png", folder / f"{csv_path.stem}_diversity.png"
    fitness.savefig(fitness_png, dpi=DPI)
    diversity.savefig(diversity_png, dpi=DPI)
    return RunCharts(fitness_png, diversity_png, fitness, diversity, {**skipped_fitness, **skipped_diversity})


def find_csvs(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if (path / CSV_NAME).is_file():
        return [path / CSV_NAME]
    return sorted(path.glob(f"*/{CSV_NAME}")) if path.is_dir() else []


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Regenera los graficos de una corrida (aptitud maxima y promedio; diversidad y sigma de mutacion) "
                    "desde su CSV por generacion, sin simulador. Escribe <nombre>_fitness.png y <nombre>_diversity.png "
                    "junto al CSV.")
    p.add_argument("paths", nargs="+", metavar="RUTA",
                   help="un CSV con las columnas de ga.CSV_COLUMNS, una carpeta de corrida (usa su generations.csv) o una "
                        "carpeta con varias corridas (seed-1, seed-2, ...)")
    p.add_argument("--out", default=None, help="carpeta de salida (default: la del CSV)")
    p.add_argument("--title-suffix", default="", help="texto agregado al titulo entre parentesis, p. ej. \"semilla 1\"")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    code, csvs = 0, []
    for path in args.paths:
        found = find_csvs(path)
        if not found:
            print(f"Error: no hay CSV de generaciones en {path}", file=sys.stderr)
            code = 1
        csvs += found
    if args.out and len(csvs) > 1:
        parser.error("--out solo sirve con un unico CSV")
    for csv_path in csvs:
        try:
            charts = plot_run(csv_path, args.out, args.title_suffix)
        except (OSError, ValueError) as error:
            print(f"Error: {error}", file=sys.stderr)
            code = 1
            continue
        print(f"{charts.fitness_png}\n{charts.diversity_png}")
    return code


if __name__ == "__main__":
    sys.exit(main())
