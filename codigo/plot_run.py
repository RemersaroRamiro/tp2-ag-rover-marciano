"""Charts of one GA run, drawn from its per-generation CSV alone (no simulator, no MuJoCo).

    uv run python plot_run.py runs/<run>                  # the run folder: reads its generations.csv
    uv run python plot_run.py runs/tp                     # every run folder inside (seed-1, seed-2, ...)
    uv run python plot_run.py results/random_search.csv --title-suffix "busqueda aleatoria"

Two PNGs per CSV, written next to it (or into ``--out``), named after the CSV: ``<stem>_fitness.png`` and
``<stem>_diversity.png`` (``generations.csv`` -> ``generations_fitness.png``). Any CSV with the columns of
``ga.CSV_COLUMNS`` works (a baseline's too); ``generation``, ``best`` and ``mean`` are required, ``diversity`` and
``mutation_sigma`` are drawn when present.

- fitness chart, in the style of the course's approved example: title "Evolución de la Aptitud", x "Generación",
  y "Aptitud", series "Máximo" (``best``) and "Promedio" (``mean``), grid, whole-number generation ticks;
- diversity chart: "Diversidad y paso de mutación", the population diversity (``diversity`` = mean per-gene standard
  deviation) and the mutation step (``mutation_sigma``). Both are in normalised gene units ([0, 1]), so they share one
  y axis.

The chart text is Spanish because the charts go into the Spanish report. Generations are numbered as in the CSV (0 =
the initial population, as in DEAP's logbook). Fitness may be negative (an illegal design scores -(1 + violation)) and
non-finite (-inf: every simulation of the generation failed; NaN: no finite fitness to average): non-finite points are
left out of their line and a note under the chart says how many, per series; a series without any finite value is not
drawn and the note says so ("sin datos").
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

CSV_NAME = "generations.csv"  # the file ``ga.RunLogger`` writes in a run folder
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

FIGSIZE = (8.0, 5.0)  # inches
DPI = 150  # crisp enough for a printed report


@dataclass(frozen=True)
class RunCharts:
    """What ``plot_run`` drew: the two PNG paths, their figures, and per series how many non-finite points were left
    out (only series with at least one)."""

    fitness_png: Path
    diversity_png: Path
    fitness_figure: Figure
    diversity_figure: Figure
    skipped: dict[str, int]


def _number(text: str | None) -> float:
    """A CSV cell as a float; empty or unreadable is NaN ("-inf", "inf" and "nan" read as themselves)."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def read_generations(csv_path: str | Path) -> dict[str, list[float]]:
    """The CSV's columns as lists of floats (``generation`` as ints). ``ValueError`` without the required columns or
    without any row."""
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
    """One line of the finite points; returns how many points were left out, or ``None`` if nothing was drawn."""
    points = [(g, v) for g, v in zip(generations, values) if math.isfinite(v)]
    if not points:
        return None
    xs, ys = zip(*points)
    ax.plot(xs, ys, label=label, marker="o" if len(points) == 1 else None)  # a lone point needs a marker to be seen
    return len(values) - len(points)


def _chart(series: list[tuple[str, list[float]]], generations, title, y_label) -> tuple[Figure, dict[str, int]]:
    fig = Figure(figsize=FIGSIZE)
    FigureCanvasAgg(fig)  # headless (Agg); no pyplot, so no global state and no window
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
    """Best ("Máximo") and mean ("Promedio") fitness per generation, and the non-finite points left out per series."""
    series = [(BEST_LABEL, data["best"]), (MEAN_LABEL, data["mean"])]
    return _chart(series, data["generation"], _titled(FITNESS_TITLE, title_suffix), FITNESS_LABEL)


def diversity_figure(data: dict[str, list[float]], title_suffix: str = "") -> tuple[Figure, dict[str, int]]:
    """Population diversity and mutation sigma per generation (one axis: both in gene units)."""
    nothing = [math.nan] * len(data["generation"])
    series = [(DIVERSITY_LABEL, data.get("diversity", nothing)), (SIGMA_LABEL, data.get("mutation_sigma", nothing))]
    return _chart(series, data["generation"], _titled(DIVERSITY_TITLE, title_suffix), DIVERSITY_Y_LABEL)


def plot_run(csv_path: str | Path, out_dir: str | Path | None = None, title_suffix: str = "") -> RunCharts:
    """Both charts of the run logged in ``csv_path``, saved as ``<stem>_fitness.png`` and ``<stem>_diversity.png``
    in ``out_dir`` (default: the CSV's folder)."""
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
    """The CSVs a path names: a CSV file; a run folder (its ``generations.csv``); or a folder of run folders."""
    path = Path(path)
    if path.is_file():
        return [path]
    if (path / CSV_NAME).is_file():
        return [path / CSV_NAME]
    return sorted(path.glob(f"*/{CSV_NAME}")) if path.is_dir() else []


# -- command line -----------------------------------------------------------------------------------------------------


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
    if args.out and len(csvs) > 1:  # several runs' generations.csv would overwrite each other's charts there
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
