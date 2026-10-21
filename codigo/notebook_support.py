"""Helpers of the course-style notebook ``TP2_AG_Rover.ipynb``: a finished run folder (or a folder of several runs) as
the notebook's tables and charts, and the notebook's parameters as the command line of ``evolve.py``.

Results mode needs only numpy, DEAP and matplotlib: this module never imports the simulator (``evolve``, ``evaluate``,
MuJoCo) at module level. Only ``run_evolve`` (simulation mode) imports ``evolve``, when it is called.

    run = load_run("runs/tp-final/c1-base")        # run.json + generations.csv of one run
    generation_table(run)                          # Markdown tables, shown with IPython.display.Markdown
    runs = load_experiment("runs/tp-final")        # every run folder inside, at any depth
    comparison_table(runs); overlay_figure(runs)   # the cross-run table and the "Máximo" curves of every run

The tables read what the run wrote (``run.json``: methods, validation, success flag; ``generations.csv``: the curves);
nothing is recomputed, so the notebook shows exactly what ``evolve.py`` decided. Text is Spanish (the report's language).
"""

from __future__ import annotations

import json
import math
import shlex
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

import ga
import genome_spec
import plot_run

NOT_AVAILABLE = "n/d"
NOT_APPLICABLE = "—"

# Spanish description and unit of every gene (genome_spec.GENES order); a test checks that no gene is missing.
GENE_INFO: dict[str, tuple[str, str]] = {
    "chassis_length_m": ("largo del chasis", "m"),
    "track_width_m": ("distancia lateral entre las ruedas izquierdas y derechas", "m"),
    "chassis_height_m": ("alto del chasis", "m"),
    "rocker_pivot_height_m": ("altura del pivote del balancín sobre los ejes de las ruedas", "m"),
    "rocker_pivot_offset": ("posición del pivote del balancín a lo largo del chasis", "fracción del largo del chasis"),
    "rocker_front_arm_m": ("brazo del balancín: pivote a rueda delantera", "m"),
    "rocker_rear_arm_m": ("brazo del balancín: pivote al pivote del bogie", "m"),
    "bogie_length_m": ("largo del bogie: entre ruedas media y trasera", "m"),
    "wheel_diameter_m": ("diámetro de las seis ruedas", "m"),
    "wheel_width_m": ("ancho de las seis ruedas", "m"),
    "motor_peak_torque_nm": ("torque pico del motor de cada rueda", "N·m"),
    "battery_energy_wh": ("energía de la batería", "Wh"),
    "target_speed_mps": ("velocidad crucero", "m/s"),
    "torque_limit_fraction": ("fracción del torque pico que usa el controlador", "fracción (0 a 1)"),
    "slope_brake": ("cuánto frena el controlador según la pendiente (0 = nunca)", "adimensional (0 a 1)"),
}

# How a simulation ended (``evolve.status_label``: "finished" or "dead:<reason>"), in words.
CAUSES = {
    "finished": "llegó a la meta",
    "dead:stuck": "atascado",
    "dead:tipped": "volcó",
    "dead:energy": "sin energía",
    "dead:out_of_lane": "salió del carril",
    "dead:timeout": "tiempo agotado",
    "dead:unstable": "simulación inestable",
}

KIND_LABELS = {"ga": "AG", "random_search": "búsqueda aleatoria"}

# The GA parameters of run.json (``ga``) that section 2 of the notebook sets.
COMPARED_GA_PARAMETERS = ("population_size", "seed", "selection", "tournament_size", "crossover", "crossover_rate",
                          "blx_alpha", "binomial_indpb", "mutation_gene_rate", "mutation_sigma", "sigma_max",
                          "adaptive_mutation", "elite_count")


# -- loading -----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    """A finished (or interrupted) run folder: its manifest (``run.json``) and its per-generation CSV columns."""

    path: Path
    name: str
    manifest: dict
    generations: dict[str, list[float]]

    @property
    def kind(self) -> str:
        """``ga`` or ``random_search`` (a run saved before baselines existed is a GA run)."""
        return self.manifest.get("kind", "ga")

    @property
    def success(self) -> dict:
        return self.manifest.get("success") or {}

    @property
    def validation(self) -> dict:
        return self.manifest.get("validation") or {}


def load_run(path: str | Path, name: str | None = None) -> Run:
    """The run in folder ``path`` (``run.json`` and ``generations.csv``); ``FileNotFoundError`` if it is not one (the
    message is Spanish: the notebook shows it)."""
    path = Path(path)
    if not (path / "run.json").is_file():
        raise FileNotFoundError(f"{path} no es una carpeta de corrida (falta run.json): indicar una corrida terminada en RUN_DIR")
    manifest = json.loads((path / "run.json").read_text(encoding="utf-8"))
    return Run(path, name or path.name, manifest, plot_run.read_generations(path / plot_run.CSV_NAME))


def load_experiment(path: str | Path) -> list[Run]:
    """Every run folder under ``path`` (a ``run.json`` at any depth), sorted by path and named by it. A folder without
    its ``generations.csv`` yet (a run still starting or in progress) is left out."""
    path = Path(path)
    return [load_run(m.parent, m.parent.relative_to(path).as_posix()) for m in sorted(path.rglob("run.json"))
            if (m.parent / plot_run.CSV_NAME).is_file()]


# -- formatting --------------------------------------------------------------------------------------------------------


def markdown_table(headers: list[str], rows: list[list]) -> str:
    """A Markdown table (shown as a table by Jupyter, Colab and GitHub)."""
    def cell(value) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(cell(h) for h in headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def number(value, spec: str) -> str:
    """``value`` formatted with ``spec``; missing or non-finite values are a dash."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return NOT_APPLICABLE
    return format(value, spec)


def format_value(value: float) -> str:
    """A physical value without trailing zeros: three decimals below 10, two below 100, one above (0.525, 12.25,
    120, 1772.1)."""
    decimals = 3 if abs(value) < 10 else 2 if abs(value) < 100 else 1
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def yes_no(flag) -> str:
    return NOT_AVAILABLE if flag is None else ("sí" if flag else "no")


# -- one run -----------------------------------------------------------------------------------------------------------


def generation_table(run: Run) -> str:
    """Per generation: best ("Máximo") and mean fitness, diversity (mean per-gene std) and the mutation sigma used to
    breed the next generation (a dash for a random search, which has no mutation)."""
    d = run.generations
    nothing = [math.nan] * len(d["generation"])
    rows = [[g, number(best, ".2f"), number(mean, ".2f"), number(div, ".3f"), number(sigma, ".3f")]
            for g, best, mean, div, sigma in zip(d["generation"], d["best"], d["mean"], d.get("diversity", nothing),
                                                 d.get("mutation_sigma", nothing))]
    return markdown_table(["Generación", plot_run.BEST_LABEL, plot_run.MEAN_LABEL, "Diversidad", "σ de mutación"], rows)


def design_rows(genome) -> list[list[str]]:
    """One row per gene: name, description, value in [0, 1], physical value (``genome_spec.decode``) and unit."""
    genome = np.asarray(genome, dtype=float)
    decoded = genome_spec.decode(genome)
    return [[gene.name, GENE_INFO[gene.name][0], f"{u:.3f}", format_value(decoded[gene.name]), GENE_INFO[gene.name][1]]
            for gene, u in zip(genome_spec.GENES, genome)]


def design_table(genome) -> str:
    """The chromosome decoded into the rover's physical values, with units."""
    return markdown_table(["Gen", "Descripción", "Valor en el cromosoma (0 a 1)", "Valor físico", "Unidad"], design_rows(genome))


def best_entry(run: Run) -> dict | None:
    """The validation of the run's best design (the first validated candidate: ``final_best`` of a GA run, the final
    population's best; ``random_best`` of a random search); ``None`` without validation."""
    candidates = run.validation.get("candidates") or []
    return candidates[0] if candidates else None


def reference_entry(run: Run) -> dict | None:
    """The validation of the Perseverance-like reference design on the run's validation terrains."""
    return run.validation.get("reference")


def best_design_table(run: Run) -> str:
    """The run's best design decoded (the validated one; the final generation's best without validation)."""
    entry = best_entry(run)
    genome = entry["genome"] if entry else (run.manifest.get("final") or {}).get("best_genome")
    if genome is None:
        return "La corrida no tiene un mejor diseño registrado."
    return design_table(genome)


def validation_table(entry: dict | None) -> str:
    """One row per validation terrain: its seed, the distance driven, how the simulation ended, and whether it finished."""
    if not entry:
        return "Sin validación registrada."
    rows = [[seed, f"{distance:.1f}", CAUSES.get(status, status), yes_no(finished)]
            for seed, distance, status, finished in zip(entry["seeds"], entry["distances_m"], entry["statuses"],
                                                        entry["finished_by_seed"])]
    return markdown_table(["Terreno (semilla)", "Distancia (m)", "Causa de fin", "¿Llegó a la meta?"], rows)


def verdict(flag: dict | None, required: int | None) -> str:
    """**Sí** / **No** and why, for one success flag of run.json (``success.best`` or ``success.reference``)."""
    if not flag:
        return f"{NOT_AVAILABLE}: la corrida no tiene criterio de éxito registrado (sin validación, o anterior a ese criterio)."
    legal = "es legal" if flag["legal"] else "es ilegal (no cumple las restricciones de la misión)"
    return (f"**{'Sí' if flag['success'] else 'No'}**: el diseño `{flag['label']}` {legal} y llegó a la meta en "
            f"{flag['finished']} de {flag['seeds']} terrenos de validación (el criterio pide al menos {required}).")


def success_line(run: Run) -> str:
    """The run's success flag (partial delivery, section 6.3) as the answer to "¿Resuelve el problema?"."""
    return f"¿Resuelve el problema? {verdict(run.success.get('best'), run.success.get('required_finishes'))}"


def reference_line(run: Run) -> str:
    return f"Diseño de referencia (tipo Perseverance): {verdict(run.success.get('reference'), run.success.get('required_finishes'))}"


def top3_line(run: Run) -> str:
    """How many validation terrains each of the top-3 validated designs finished."""
    top3 = run.success.get("top3") or []
    if not top3:
        return f"Top-3 validados: {NOT_AVAILABLE}"
    return "Top-3 validados, llegadas: " + ", ".join(f"{t['label']} {t['finished']}/{t['seeds']}" for t in top3)


def selection_text(params: dict) -> str:
    if "selection" not in params:
        return NOT_AVAILABLE
    return f"torneo (k={params['tournament_size']})" if params["selection"] == "tournament" else "ruleta"


def crossover_text(params: dict) -> str:
    if "crossover" not in params:
        return NOT_AVAILABLE
    kind = f"BLX-α (α={params['blx_alpha']:g})" if params["crossover"] == "blx" else f"binomial (p_gen={params['binomial_indpb']:g})"
    return f"{kind}, p={params['crossover_rate']:g}"


def mutation_text(params: dict) -> str:
    if "adaptive_mutation" not in params:
        return NOT_AVAILABLE
    sigma = (f"σ {params['mutation_sigma']:g}–{params['sigma_max']:g} adaptativa" if params["adaptive_mutation"]
             else f"σ fija {params['mutation_sigma']:g}")
    return f"gaussiana, p_gen={params['mutation_gene_rate']:g}, {sigma}"


def methods(run: Run) -> list[str]:
    """Selection, crossover, mutation and elitism of a GA run (dashes for a random search)."""
    if run.kind != "ga":
        return [NOT_APPLICABLE] * 4
    params = run.manifest.get("ga") or {}
    return [selection_text(params), crossover_text(params), mutation_text(params), str(params.get("elite_count", NOT_AVAILABLE))]


def run_summary(run: Run) -> str:
    """What the loaded run is: kind, size, seed, status and methods (from its run.json)."""
    m, preset = run.manifest, run.manifest.get("preset") or {}
    selection, crossover, mutation, elitism = methods(run)
    terrains = preset.get("seeds_per_generation", NOT_AVAILABLE)
    lines = [
        f"- **Corrida:** `{run.path}` ({KIND_LABELS.get(run.kind, run.kind)}, estado `{m.get('status', NOT_AVAILABLE)}`)",
        f"- **Tamaño:** {preset.get('population', NOT_AVAILABLE)} individuos x {len(m.get('generations') or [])} generaciones, "
        f"{terrains} {'terreno' if terrains == 1 else 'terrenos'} por generación, semilla {m.get('run_seed', NOT_AVAILABLE)}",
    ]
    if run.kind == "ga":
        lines.append(f"- **Métodos:** selección {selection}; cruce {crossover}; mutación {mutation}; elitismo {elitism}")
    validation_seeds = len(run.validation.get("seeds") or [])
    lines.append(f"- **Validación:** {validation_seeds} terrenos no vistos durante la evolución")
    return "\n".join(lines)


# -- several runs ------------------------------------------------------------------------------------------------------


def comparison_rows(runs: list[Run]) -> list[list[str]]:
    rows = []
    for run in runs:
        entry, best = best_entry(run) or {}, run.success.get("best") or {}
        finishes = f"{entry['finished']}/{len(entry['seeds'])}" if entry else NOT_AVAILABLE
        rows.append([run.name, KIND_LABELS.get(run.kind, run.kind), *methods(run),
                     number((run.manifest.get("final") or {}).get("best_fitness"), ".2f"),
                     number(entry.get("mean_distance_m"), ".1f"), finishes, yes_no(best.get("success"))])
    return rows


def comparison_table(runs: list[Run]) -> str:
    """One row per run (GA runs and baselines): methods, best design's fitness, its validation and the success flag."""
    headers = ["Corrida", "Tipo", "Selección", "Cruce", "Mutación", "Elitismo", "Aptitud del mejor diseño",
               "Distancia media de validación (m)", "Llegadas", "¿Éxito?"]
    return markdown_table(headers, comparison_rows(runs))


def overlay_figure(runs: list[Run], title: str = "Comparación entre corridas: aptitud máxima por generación") -> Figure:
    """The "Máximo" curve of every run on one chart (non-finite points left out, as in ``plot_run``)."""
    fig = Figure(figsize=plot_run.FIGSIZE)
    FigureCanvasAgg(fig)  # headless; the notebook shows the figure with IPython.display
    ax = fig.add_subplot()
    for run in runs:
        points = [(g, v) for g, v in zip(run.generations["generation"], run.generations["best"]) if math.isfinite(v)]
        if points:
            xs, ys = zip(*points)
            ax.plot(xs, ys, label=run.name, marker="o" if len(points) == 1 else None)
    ax.set_title(title)
    ax.set_xlabel(plot_run.GENERATION_LABEL)
    ax.set_ylabel(plot_run.FITNESS_LABEL)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True)
    if ax.get_lines():
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# -- the DEAP toolbox --------------------------------------------------------------------------------------------------


def operator_lines(toolbox, aliases: tuple[str, ...] = ("select", "mate", "mutate")) -> list[str]:
    """What is registered on ``toolbox``, as the ``toolbox.register(...)`` line that would register it."""
    lines = []
    for alias in aliases:
        partial = getattr(toolbox, alias)
        func = partial.func
        module = "tools" if func.__module__.startswith("deap.tools") else func.__module__
        arguments = "".join(f", {key}={value!r}" for key, value in partial.keywords.items())
        lines.append(f'toolbox.register("{alias}", {module}.{func.__name__}{arguments})')
    return lines


def parameter_differences(run: Run, config: ga.GAConfig, generations: int, seeds_per_generation: int,
                          validation_seeds: int) -> list[str]:
    """The notebook's parameters (section 2) that differ from the ones the loaded run used, one line each."""
    differences = []
    params = run.manifest.get("ga") or {}
    wanted = asdict(config)
    for key in COMPARED_GA_PARAMETERS:
        if run.kind == "ga" and key in params and params[key] != wanted[key]:
            differences.append(f"{key}: la corrida usó {params[key]!r}, la sección 2 dice {wanted[key]!r}")
    preset = run.manifest.get("preset") or {}
    run_sizes = {"generations": preset.get("generations"), "seeds_per_generation": preset.get("seeds_per_generation"),
                 "validation_seeds": (run.manifest.get("validation_rules") or {}).get("seeds")}
    notebook_sizes = {"generations": generations, "seeds_per_generation": seeds_per_generation, "validation_seeds": validation_seeds}
    for key, value in run_sizes.items():
        if value is not None and value != notebook_sizes[key]:
            differences.append(f"{key}: la corrida usó {value!r}, la sección 2 dice {notebook_sizes[key]!r}")
    return differences


# -- simulation mode: the notebook's parameters through evolve.py's command line --------------------------------------------


def evolve_arguments(config: ga.GAConfig, generations: int, seeds_per_generation: int, validation_seeds: int,
                     out_dir: str | Path, max_sim_time_s: float | None = None, workers: int | None = None) -> list[str]:
    """The ``evolve.py`` arguments of the run the notebook describes: preset ``normal`` (Tabla 1) resized to the
    notebook's population and generations, with every GA method and parameter of ``config``."""
    arguments = [
        "--preset", "normal", "--seed", str(config.seed), "--population", str(config.population_size),
        "--generations", str(generations), "--seeds-per-generation", str(seeds_per_generation),
        "--validation-seeds", str(validation_seeds),
        "--selection", config.selection, "--tournament-size", str(config.tournament_size),
        "--crossover", config.crossover, "--blx-alpha", repr(config.blx_alpha),
        "--binomial-indpb", repr(config.binomial_indpb), "--crossover-rate", repr(config.crossover_rate),
        "--mutation-rate", repr(config.mutation_gene_rate), "--sigma-min", repr(config.mutation_sigma),
        "--sigma-max", repr(config.sigma_max), "--adaptive-mutation" if config.adaptive_mutation else "--no-adaptive-mutation",
        "--elite-count", str(config.elite_count), "--out", str(out_dir)]
    if max_sim_time_s is not None:
        arguments += ["--max-sim-time", repr(float(max_sim_time_s))]
    if workers is not None:
        arguments += ["--workers", str(workers)]
    return arguments


def command_line(arguments: list[str]) -> str:
    """The shell command that runs the same thing from a terminal."""
    return "uv run python evolve.py " + shlex.join(arguments)


def default_output_dir(prefix: str = "notebook") -> Path:
    """A new run folder name under ``runs/`` for a run launched from the notebook."""
    return Path("runs") / f"{prefix}-{datetime.now():%Y%m%d-%H%M%S}"


def run_evolve(arguments: list[str]) -> Path:
    """Run ``evolve.py`` with ``arguments`` in this process (its per-generation lines are printed as it goes) and return
    the run folder. Imports the simulator: only simulation mode calls it."""
    import evolve  # MuJoCo comes with it: never imported in results mode

    code = evolve.main(arguments)
    if code != 0:
        raise RuntimeError(f"evolve.py terminó con código {code}")
    return Path(arguments[arguments.index("--out") + 1])
