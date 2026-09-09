# ccagg/visualisation.py

"""
Figures for the multi-view aggregated CCA permutation results.

Functions
---------
plot_permutation_summary    : Fig 1 — violin distributions + mean null CIs.
plot_train_vs_test          : Fig 3 — train vs. test correlation scatter.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import gridspec  # noqa: F401
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch  # noqa: F401
from matplotlib.transforms import blended_transform_factory

# ─────────────────────────────────────────────────────────────────────────────
# Global style
# ─────────────────────────────────────────────────────────────────────────────

NATURE_RC: dict = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.0,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
}

ALPHA = 0.05
DIM_COLORS = ["#2166AC", "#D6604D", "#4DAC26"]
SPLIT_MARKERS = ["o", "s"]
SPLIT_LABELS = ["Split 1", "Split 2"]
DIM_LABELS = ["Dim 1", "Dim 2", "Dim 3"]


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────


def _apply_nature_style() -> None:
    plt.rcParams.update(NATURE_RC)


def _sig_stars(p: float, alpha: float = ALPHA) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < alpha:
        return "*"
    return "ns"


def _style_violin(parts: dict, color: str, alpha: float = 0.70) -> None:
    """Apply uniform colour / style to a violinplot parts dict."""
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(alpha)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color(color)
            parts[key].set_linewidth(0.9)
            
            
###############################################################################

# ─────────────────────────────────────────────────────────────────────────────
# Figure  — Variability of regularisation parameters across outer splits
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float_plot(x, default=np.nan) -> float:
    try:
        val = float(x)
        return val if np.isfinite(val) else default
    except Exception:
        return default
    
def _flatten_numeric_params(
    params: dict | None,
) -> dict[str, float]:
    """
    Flatten scalar or per-view/list hyperparameters into numeric columns.

    Example
    -------
    {"latent_dimensions": 3,
     "alpha": [1e-4, 3e-4, 1e-3],
     "l1_ratio": [0.75, 0.45, 0.30]}

    becomes

    {"latent_dimensions": 3.0,
     "alpha_v1": 1e-4,
     "alpha_v2": 3e-4,
     "alpha_v3": 1e-3,
     "l1_ratio_v1": 0.75,
     "l1_ratio_v2": 0.45,
     "l1_ratio_v3": 0.30}
    """
    flat = {}

    if params is None:
        return flat

    for key, value in params.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value, dtype=float).ravel()
            for i, val in enumerate(arr):
                flat[f"{key}_v{i + 1}"] = _safe_float_plot(val)
        else:
            flat[key] = _safe_float_plot(value)

    return flat


def _param_label(
    param_name: str,
    view_names: list[str] | None = None,
) -> str:
    """
    Convert internal flattened names into publication-friendly labels.
    """
    replacements = {
        "latent_dimensions": "Latent dimensions",
        "l1_ratio": "L1 ratio",
        "alpha": "Alpha",
    }

    if "_v" in param_name:
        base, suffix = param_name.rsplit("_v", 1)
        if suffix.isdigit():
            view_idx = int(suffix) - 1
            base_label = replacements.get(base, base.replace("_", " "))
            if view_names is not None and view_idx < len(view_names):
                return f"{base_label} ({view_names[view_idx]})"
            return f"{base_label} (View {view_idx + 1})"

    return replacements.get(param_name, param_name.replace("_", " ").title())


def _collect_hyperparameter_rows(
    results: dict,
) -> list[dict]:
    """
    Collect one row per evaluated hyperparameter setting per outer split.

    Uses:
    - results["cv_records_all_params_per_split"]
    - results["outer_test_scores_all_params_per_split"] (if available)
    - results["best_param_index_per_split"] (if available)
    """
    rows = []

    all_cv = results.get("cv_records_all_params_per_split", [])
    all_outer = results.get("outer_test_scores_all_params_per_split", [])
    all_best_idx = results.get("best_param_index_per_split", [])

    for split_idx, split_records in enumerate(all_cv):
        if split_records is None:
            continue

        outer_scores = (
            all_outer[split_idx]
            if split_idx < len(all_outer) and all_outer[split_idx] is not None
            else None
        )
        best_idx = (
            all_best_idx[split_idx]
            if split_idx < len(all_best_idx)
            else None
        )

        for param_idx, rec in enumerate(split_records):
            row = {
                "split_idx": split_idx,
                "param_idx": param_idx,
                "mean_cv": _safe_float_plot(rec.get("mean_cv")),
                "sd_cv": _safe_float_plot(rec.get("sd_cv")),
                "se_cv": _safe_float_plot(rec.get("se_cv")),
                "n_cv": _safe_float_plot(rec.get("n_cv")),
                "selected": int(best_idx == param_idx),
                "outer_test_score": (
                    _safe_float_plot(outer_scores[param_idx])
                    if (
                        outer_scores is not None
                        and param_idx < len(outer_scores)
                    )
                    else np.nan
                ),
            }
            row.update(_flatten_numeric_params(rec.get("params", {})))
            rows.append(row)

    return rows


def _summarise_xy(
    x: np.ndarray,
    y: np.ndarray,
    use_log_x: bool = False,
    max_unique: int = 10,
    n_bins: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Summarise y as a function of x using:
    - exact groups if x has few unique values
    - quantile bins otherwise

    Returns
    -------
    x_mid, y_mid, y_lo, y_hi
        Median and IQR summary.
    """
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[valid]
    y = np.asarray(y, dtype=float)[valid]

    if x.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty

    x_metric = np.log10(x) if use_log_x else x
    unique_metric = np.unique(np.round(x_metric, 12))

    x_mid = []
    y_mid = []
    y_lo = []
    y_hi = []

    if unique_metric.size <= max_unique:
        order = np.argsort(unique_metric)
        for xm in unique_metric[order]:
            mask = np.isclose(x_metric, xm, atol=1e-12, rtol=0.0)
            yy = y[mask]
            xx = x[mask]
            if yy.size == 0:
                continue
            x_mid.append(float(np.nanmedian(xx)))
            y_mid.append(float(np.nanmedian(yy)))
            y_lo.append(float(np.nanpercentile(yy, 25)))
            y_hi.append(float(np.nanpercentile(yy, 75)))
    else:
        edges = np.quantile(x_metric, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)

        if edges.size < 2:
            yy = y
            x_mid = [float(np.nanmedian(x))]
            y_mid = [float(np.nanmedian(yy))]
            y_lo = [float(np.nanpercentile(yy, 25))]
            y_hi = [float(np.nanpercentile(yy, 75))]
        else:
            bin_ids = np.digitize(x_metric, edges[1:-1], right=True)
            for b in range(len(edges) - 1):
                mask = bin_ids == b
                if not np.any(mask):
                    continue
                xx = x[mask]
                yy = y[mask]
                if yy.size == 0:
                    continue
                if use_log_x and np.all(xx > 0):
                    x_center = float(10 ** np.nanmean(np.log10(xx)))
                else:
                    x_center = float(np.nanmedian(xx))
                x_mid.append(x_center)
                y_mid.append(float(np.nanmedian(yy)))
                y_lo.append(float(np.nanpercentile(yy, 25)))
                y_hi.append(float(np.nanpercentile(yy, 75)))

    return (
        np.asarray(x_mid, dtype=float),
        np.asarray(y_mid, dtype=float),
        np.asarray(y_lo, dtype=float),
        np.asarray(y_hi, dtype=float),
    )


def _split_flat_param_name(name: str) -> tuple[str, int | None]:
    """
    Split a flattened name like 'alpha_v3' into ('alpha', 3).
    """
    if "_v" in name:
        base, suffix = name.rsplit("_v", 1)
        if suffix.isdigit():
            return base, int(suffix)
    return name, None


def _pretty_base_label(base: str) -> str:
    mapping = {
        "alpha": "Alpha",
        "l1_ratio": "L1 ratio",
        "latent_dimensions": "Latent dimensions",
    }
    
    if base.lower() == "tau":
        return "L1-sparsity"
    
    return mapping.get(base, base.replace("_", " ").title())


def _gaussian_kde(
    x: np.ndarray,
    grid: np.ndarray,
    bandwidth_scale: float = 1.0,
) -> np.ndarray:
    """
    Simple Gaussian KDE without SciPy.

    For discrete tuning grids this is best interpreted as a smoothed
    selection distribution rather than a truly continuous density.
    """
    x = np.asarray(x, dtype=float)
    grid = np.asarray(grid, dtype=float)

    if x.size == 0:
        return np.zeros_like(grid)

    uniq = np.unique(np.sort(x))

    # Constant case: use a narrow kernel
    if uniq.size == 1:
        center = uniq[0]
        span = max(abs(center) * 0.05, 1e-3)
        h = span * bandwidth_scale
    else:
        n = x.size
        std = float(np.nanstd(x, ddof=1))
        iqr = float(np.subtract(*np.nanpercentile(x, [75, 25])))

        sigma = min(std, iqr / 1.34) if iqr > 0 else std
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = max(std, 1e-3)

        h = 0.9 * sigma * n ** (-1.0 / 5.0)

        min_spacing = float(np.min(np.diff(uniq)))
        h = max(h, 0.30 * min_spacing)

        h *= bandwidth_scale

    h = max(float(h), 1e-6)

    z = (grid[:, None] - x[None, :]) / h
    dens = np.exp(-0.5 * z**2).sum(axis=1)
    dens /= x.size * h * np.sqrt(2 * np.pi)
    return dens


def _infer_param_names(
    rows: list[dict],
    param_order: list[str] | None = None,
) -> list[str]:
    """
    Infer actual numeric hyperparameter column names from the flattened rows.

    Notes
    -----
    `param_order` is used later only for ordering the plotted panels. It is not
    used here to infer which numeric parameter columns actually exist.
    """
    reserved = {
        "split_idx",
        "param_idx",
        "mean_cv",
        "sd_cv",
        "se_cv",
        "n_cv",
        "selected",
        "outer_test_score",
    }

    names = []
    seen = set()

    for row in rows:
        for key, val in row.items():
            if key in reserved or key in seen:
                continue
            if np.isfinite(_safe_float_plot(val)):
                names.append(key)
                seen.add(key)

    def _sort_key(name: str) -> tuple[int, str, int, str]:
        base, view_idx = _split_flat_param_name(name)

        if base == "latent_dimensions":
            priority = 0
        elif base == "alpha":
            priority = 1
        elif base == "l1_ratio":
            priority = 2
        else:
            priority = 3

        return (
            priority,
            base,
            view_idx if view_idx is not None else 0,
            name,
        )

    return sorted(names, key=_sort_key)


def plot_hyperparameter(
    results: dict,
    param_order: list[str] | None = None,
    view_names: list[str] | None = None,
    include_outer_test: bool = True,
    log_param_prefixes: tuple[str, ...] = ("alpha",),
    exclude_from_density: tuple[str, ...] = ("latent_dimensions",),
    bandwidth_scale: float = 0.90,
    summary_stat: str = "median",
    curve_color: str = "#2166AC",
    summary_line_color: str = "#1a1a1a",
    summary_line_ls: str = "--",
    summary_line_lw: float = 0.85,
    rng_seed: int = 0,
    save_path: str = "fig_hyperparameter_combined",
) -> plt.Figure:
    """
    Combined hyperparameter visualisation: selection density (top) and
    inner-CV / outer-test score landscape (bottom) for each hyperparameter.

    Layout
    ------
    - Rows  : hyperparameters, each split into a density sub-row and a score
              sub-row.  Scalar hyperparameters (e.g. latent_dimensions) span
              all columns; per-view ones have one column per view.
    - Columns : views (for per-view hyperparameters).
    - Row labels appear on the left axis of each hyperparameter group.
    - Column titles appear above the topmost per-view density row.

    Density panels
    --------------
    - Filled area only (no solid outline curve) to keep the figure uncluttered.
    - Rug marks along the x-axis indicate the actual values selected in each
      outer split.
    - A dashed vertical line marks the median (or geometric mean for
      log-scaled axes) of the selected values; it is explicitly identified in
      the legend.

    Score panels
    ------------
    - Grey dots  : all evaluated hyperparameter configurations (jittered).
    - Open circles : the configuration selected in each outer split.
    - Blue line / band  : inner-CV median ± IQR.
    - Red dashed line / band : outer-test median ± IQR (when available).

    Parameters
    ----------
    results : dict
        Output of ``parallel_best_param_ensemble_multidim``.
    param_order : list of str or None
        Optional ordering by base name (e.g. ``["alpha", "l1_ratio"]``).
    view_names : list of str or None
        Column labels for per-view hyperparameters.
    include_outer_test : bool
        Overlay outer-test score summaries in the score panels.
    log_param_prefixes : tuple of str
        Hyperparameter prefixes that use a log x-axis.
    exclude_from_density : tuple of str
        Hyperparameters for which the density row is hidden
        (defaults to ``("latent_dimensions",)``).
    bandwidth_scale : float
        KDE bandwidth multiplier (see ``_gaussian_kde``).
    summary_stat : {"median", "mean"}
        Summary statistic shown as a vertical dashed line in density panels.
        For log-scaled axes, "mean" is the geometric mean.
    curve_color : str
        Fill colour for density panels.
    summary_line_color : str
        Colour of the vertical summary line.
    summary_line_ls : str
        Line style of the vertical summary line.
    summary_line_lw : float
        Line width of the vertical summary line.
    rng_seed : int
        Seed for reproducible jitter in score panels.
    save_path : str
        File stem; ``.pdf`` and ``.png`` are written automatically.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if summary_stat not in {"median", "mean"}:
        raise ValueError("summary_stat must be 'median' or 'mean'.")

    from matplotlib.legend_handler import HandlerBase

    _apply_nature_style()
    rng = np.random.default_rng(rng_seed)

    # ── Legend handler: jittered cloud for "all sampled" proxy ───────────────
    class _HandlerJitter(HandlerBase):
        def create_artists(
            self, legend, orig_handle,
            xdescent, ydescent, width, height, fontsize, trans,
        ):
            offsets = [
                (0.36, 0.48), (0.45, 0.66), (0.53, 0.36),
                (0.61, 0.58), (0.49, 0.26), (0.67, 0.44),
            ]
            alpha = (
                orig_handle.get_alpha()
                if orig_handle.get_alpha() is not None
                else 0.75
            )
            return [
                Line2D(
                    [xdescent + ox * width],
                    [ydescent + oy * height],
                    marker="o",
                    linestyle="None",
                    color=orig_handle.get_color(),
                    markerfacecolor=orig_handle.get_markerfacecolor(),
                    markeredgecolor=orig_handle.get_markeredgecolor(),
                    markersize=orig_handle.get_markersize(),
                    alpha=alpha,
                    transform=trans,
                )
                for ox, oy in offsets
            ]

    # ── Collect rows ──────────────────────────────────────────────────────────
    rows = _collect_hyperparameter_rows(results)
    if not rows:
        raise ValueError("No hyperparameter search records found in results.")

    best_params_list = results.get("best_params_per_split", [])
    if not best_params_list:
        raise ValueError("No best_params_per_split found in results.")

    best_rows = [_flatten_numeric_params(p) for p in best_params_list]

    # ── Infer parameter names ─────────────────────────────────────────────────
    param_names = _infer_param_names(rows, param_order=param_order)
    if not param_names:
        raise ValueError("No numeric hyperparameters could be inferred.")

    scalar_params: list[str] = []
    per_view_params: dict[str, dict[int, str]] = {}

    for name in param_names:
        base, view_idx = _split_flat_param_name(name)
        if view_idx is None:
            scalar_params.append(name)
        else:
            per_view_params.setdefault(base, {})[view_idx] = name

    # Ordering
    if param_order is not None:
        seen_s, seen_b = set(), set()
        scalar_order, per_view_order = [], []
        for item in param_order:
            base, vidx = _split_flat_param_name(item)
            if vidx is None and item in scalar_params and item not in seen_s:
                scalar_order.append(item); seen_s.add(item)
            elif base in per_view_params and base not in seen_b:
                per_view_order.append(base); seen_b.add(base)
        scalar_order += [p for p in scalar_params if p not in seen_s]
        per_view_order += [b for b in per_view_params if b not in seen_b]
    else:
        scalar_order = [p for p in ["latent_dimensions"] if p in scalar_params]
        scalar_order += sorted(p for p in scalar_params if p not in scalar_order)
        per_view_order = [b for b in ["alpha", "l1_ratio"] if b in per_view_params]
        per_view_order += sorted(b for b in per_view_params if b not in per_view_order)

    # Drop constant scalar params from score panels
    def _extract_arrays(param_name):
        x, y_cv, y_te, sel = [], [], [], []
        for row in rows:
            if param_name not in row:
                continue
            xv = _safe_float_plot(row[param_name])
            if not np.isfinite(xv):
                continue
            x.append(xv)
            y_cv.append(_safe_float_plot(row.get("mean_cv")))
            y_te.append(_safe_float_plot(row.get("outer_test_score")))
            sel.append(bool(row.get("selected", 0)))
        return (
            np.asarray(x, dtype=float),
            np.asarray(y_cv, dtype=float),
            np.asarray(y_te, dtype=float),
            np.asarray(sel, dtype=bool),
        )

    scalar_order = [
        n for n in scalar_order
        if np.unique(
            (lambda a: a[np.isfinite(a)])(_extract_arrays(n)[0])
        ).size > 1
    ]

    # ── Grid dimensions ───────────────────────────────────────────────────────
    inferred_n_cols = 1
    if per_view_params:
        inferred_n_cols = max(
            max(d.keys()) for d in per_view_params.values()
        )

    if view_names is None:
        view_names = [f"View {i}" for i in range(1, inferred_n_cols + 1)]
    else:
        view_names = list(view_names)
        while len(view_names) < inferred_n_cols:
            view_names.append(f"View {len(view_names) + 1}")

    all_params_ordered = scalar_order + per_view_order
    if not all_params_ordered:
        raise ValueError("No hyperparameters remain to plot.")

    n_param_groups = len(all_params_ordered)

    # ── Shared y-limits (score panels) ────────────────────────────────────────
    all_y = []
    for row in rows:
        yc = _safe_float_plot(row.get("mean_cv"))
        if np.isfinite(yc):
            all_y.append(yc)
        if include_outer_test:
            yt = _safe_float_plot(row.get("outer_test_score"))
            if np.isfinite(yt):
                all_y.append(yt)

    if all_y:
        all_y = np.asarray(all_y)
        y_rng = all_y.max() - all_y.min()
        pad = 0.10 * y_rng if y_rng > 0 else 0.05
        ylim = (float(all_y.min()) - pad, float(all_y.max()) + pad)
    else:
        ylim = (-0.1, 0.1)

    # ── Per-row axis config (x-limits, log scale) ─────────────────────────────
    def _row_xcfg(param_name_or_base, is_per_view=False):
        if is_per_view:
            base = param_name_or_base
            xs = []
            for pname in per_view_params[base].values():
                a, _, _, _ = _extract_arrays(pname)
                xs.append(a)
            x_all = np.concatenate(xs) if xs else np.array([], dtype=float)
        else:
            x_all, _, _, _ = _extract_arrays(param_name_or_base)
            base = param_name_or_base

        finite = x_all[np.isfinite(x_all)]
        if finite.size == 0:
            return {"use_log_x": False, "xlim": (0.0, 1.0)}

        use_log = (
            any(base.startswith(p) for p in log_param_prefixes)
            and np.all(finite > 0)
        )

        if use_log:
            z = np.log10(finite)
            pad = max(0.10 * (z.max() - z.min()), 0.08) if np.unique(z).size > 1 else 0.18
            return {"use_log_x": True, "xlim": (10**(z.min()-pad), 10**(z.max()+pad))}

        is_int = np.all(np.isclose(finite, np.round(finite)))
        if np.unique(finite).size == 1:
            pad = 0.35 if is_int else max(abs(finite[0])*0.08, 0.05)
        else:
            min_pad = 0.35 if is_int else 0.03
            pad = max(0.10*(finite.max()-finite.min()), min_pad)
        return {"use_log_x": False, "xlim": (float(finite.min()-pad), float(finite.max()+pad))}

    row_cfg = {}
    for name in scalar_order:
        row_cfg[name] = _row_xcfg(name, is_per_view=False)
    for base in per_view_order:
        row_cfg[base] = _row_xcfg(base, is_per_view=True)

    # ── Figure + nested GridSpec ──────────────────────────────────────────────
    # Each hyperparameter group = inner gridspec with [density row, score row].
    # Density row is hidden for excluded params.
    density_h  = 0.80   # relative height of density sub-row
    score_h    = 2.20   # relative height of score sub-row

    fig_w = 1.95 * inferred_n_cols + 1.20
    fig_h = (density_h + score_h + 0.25) * n_param_groups + 1.00

    fig = plt.figure(figsize=(fig_w, fig_h))

    outer_gs = fig.add_gridspec(
        n_param_groups, 1,
        hspace=0.45,
        left=0.18, right=0.97,
        top=0.88,       
        bottom=0.20)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _compute_jitter_width(x):
        x = x[np.isfinite(x)]
        if x.size == 0:
            return 0.0
        uniq = np.unique(x)
        if uniq.size >= 2:
            sp = float(np.min(np.diff(uniq)))
            if sp > 0:
                return 0.12 * sp
        return max(abs(float(np.nanmedian(x))) * 0.03, 0.03)

    def _draw_density(ax, x, xcfg, show_title=False, col_title=""):
        """Filled area only — no outline, no lines of any kind."""
        use_log = xcfg["use_log_x"]
        xlim    = xcfg["xlim"]
        if use_log:
             xlim = (xlim[0] / 3.0, xlim[1])
        else:
             xlim = (xlim[0] - 0.20 * (xlim[1] - xlim[0]), xlim[1])
        x = x[np.isfinite(x)]

        if x.size == 0:
            ax.set_visible(False)
            return

        if use_log:
            ax.set_xscale("log")
            grid_z = np.linspace(np.log10(xlim[0]), np.log10(xlim[1]), 400)
            dens = _gaussian_kde(np.log10(x), grid_z, bandwidth_scale)
            if np.nanmax(dens) > 0:
                dens /= np.nanmax(dens)
            x_grid = 10**grid_z
            x_ref = (
                float(np.nanmedian(x))
                if summary_stat == "median"
                else float(10**np.nanmean(np.log10(x)))
            )
        else:
            grid = np.linspace(xlim[0], xlim[1], 400)
            dens = _gaussian_kde(x, grid, bandwidth_scale)
            if np.nanmax(dens) > 0:
                dens /= np.nanmax(dens)
            x_grid = grid
            x_ref = (
                float(np.nanmedian(x))
                if summary_stat == "median"
                else float(np.nanmean(x))
            )

        # Filled area only — no outline curve whatsoever
        ax.fill_between(
            x_grid, 0, dens,
            color=curve_color, alpha=0.32, linewidth=0, zorder=1,
        )
        # Rug marks
        ax.vlines(
            x, 0, 0.09,
            color=curve_color, lw=0.45, alpha=0.40, zorder=3,
        )
        # Summary line
        ax.axvline(
            x_ref,
            color=summary_line_color, ls=summary_line_ls,
            lw=summary_line_lw, alpha=0.95, zorder=4,
        )

        ax.set_xlim(xlim)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

        if show_title:
            ax.set_title(col_title, fontsize=5, fontweight="bold", pad=3)

    def _draw_score(
        ax, x, y_cv, y_te, sel, xcfg,
        x_label="",
        show_ylabel=False,
        force_int_ticks=False,
    ):
        use_log = xcfg["use_log_x"]
        xlim    = xcfg["xlim"]

        valid_cv = np.isfinite(x) & np.isfinite(y_cv)
        valid_te = np.isfinite(x) & np.isfinite(y_te)

        if not np.any(valid_cv):
            ax.set_visible(False)
            return

        # Jittered raw points
        x_plot = x.copy()
        if not use_log:
            jw = _compute_jitter_width(x)
            if jw > 0:
                x_plot = x + rng.uniform(-jw, jw, size=x.size)

        ax.scatter(
            x_plot[valid_cv], y_cv[valid_cv],
            s=7, color="0.75", linewidths=0, alpha=0.38,
            zorder=1, rasterized=True,
        )

        # Selected settings per split
        sel_ok = sel & valid_cv
        if np.any(sel_ok):
            ax.scatter(
                x[sel_ok], y_cv[sel_ok],
                s=13, facecolors="white", edgecolors="black",
                linewidths=0.55, alpha=0.95, zorder=4,
            )

        # Inner-CV summary
        xm, ym, ylo, yhi = _summarise_xy(x, y_cv, use_log_x=use_log)
        if xm.size > 0:
            o = np.argsort(xm)
            ax.fill_between(
                xm[o], ylo[o], yhi[o],
                color="#2166AC", alpha=0.14, linewidth=0, zorder=2,
            )
            ax.plot(
                xm[o], ym[o],
                color="#2166AC", lw=1.2, marker="o", ms=2.5, zorder=3,
            )

        # Outer-test summary
        if include_outer_test and np.any(valid_te):
            xmt, ymt, ylot, yhit = _summarise_xy(x, y_te, use_log_x=use_log)
            if xmt.size > 0:
                ot = np.argsort(xmt)
                ax.fill_between(
                    xmt[ot], ylot[ot], yhit[ot],
                    color="#D6604D", alpha=0.10, linewidth=0, zorder=2,
                )
                ax.plot(
                    xmt[ot], ymt[ot],
                    color="#D6604D", lw=1.0, ls="--",
                    marker="s", ms=2.3, zorder=3,
                )

        if use_log:
            ax.set_xscale("log")
        if force_int_ticks:
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        ax.axhline(0, color="black", lw=0.35, ls="--", zorder=0)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=5.5, width=0.5, length=2.5)
        ax.set_xlabel(x_label, fontsize=6.5)
        ax.set_ylabel("Score" if show_ylabel else "", fontsize=6)

    # ── Draw each hyperparameter group ────────────────────────────────────────
    first_per_view_group = (
        len(scalar_order) if per_view_order else None
    )

    for g_idx, key in enumerate(all_params_ordered):
        is_scalar   = key in scalar_order
        base        = key
        is_excluded = any(key == ex or key.startswith(ex) for ex in exclude_from_density)
        show_density_row = not is_excluded

        hr = [density_h, score_h] if show_density_row else [0.001, score_h]
        inner_gs = outer_gs[g_idx].subgridspec(
            2, inferred_n_cols,
            height_ratios=hr,
            hspace=0.08,
            wspace=0.35,
        )

        n_cols_this = 1 if is_scalar else inferred_n_cols

        for c in range(inferred_n_cols):
            # ── Density sub-row ──
            if is_scalar:
                d_ax = fig.add_subplot(inner_gs[0, :])
            else:
                d_ax = fig.add_subplot(inner_gs[0, c])

            if show_density_row:
                if is_scalar:
                    x_best = np.asarray(
                        [
                            _safe_float_plot(r.get(key))
                            for r in best_rows
                            if np.isfinite(_safe_float_plot(r.get(key, np.nan)))
                        ],
                        dtype=float,
                    )
                    _draw_density(
                        d_ax, x_best, row_cfg[key],
                        show_title=False,
                    )
                else:
                    pname = per_view_params[base].get(c + 1)
                    if pname is None:
                        d_ax.set_visible(False)
                    else:
                        x_best = np.asarray(
                            [
                                _safe_float_plot(r.get(pname))
                                for r in best_rows
                                if np.isfinite(
                                    _safe_float_plot(r.get(pname, np.nan))
                                )
                            ],
                            dtype=float,
                        )
                        show_col_title = (
                            g_idx == first_per_view_group
                        )
                        _draw_density(
                            d_ax, x_best, row_cfg[base],
                            show_title=show_col_title,
                            col_title=view_names[c],
                        )
            else:
                d_ax.set_visible(False)

            if is_scalar and c > 0:
                break  # span handled above

            # ── Score sub-row ──
            if is_scalar:
                s_ax = fig.add_subplot(inner_gs[1, :])
                x_s, yc_s, yt_s, sel_s = _extract_arrays(key)
                _draw_score(
                    s_ax, x_s, yc_s, yt_s, sel_s,
                    xcfg=row_cfg[key],
                    x_label=_pretty_base_label(key),   
                    show_ylabel=True,
                    force_int_ticks=(key == "latent_dimensions"))
                break
            else:
                s_ax = fig.add_subplot(inner_gs[1, c])
                pname = per_view_params[base].get(c + 1)
                if pname is None:
                    s_ax.set_visible(False)
                    continue
                x_b, yc_b, yt_b, sel_b = _extract_arrays(pname)
                _draw_score(
                    s_ax, x_b, yc_b, yt_b, sel_b,
                    xcfg=row_cfg[base],
                    x_label=_pretty_base_label(base),  
                    show_ylabel=(c == 0))

        # ── Row label (left margin) ───────────────────────────────────────────
        # Placed after layout so bbox coordinates are available at draw time.
        # We attach it as a figure-level annotation keyed to the first visible
        # score axis bbox; finalized in a post-draw pass below.
        # Store refs for the post-draw label pass.
        if g_idx == 0:
            _row_label_info = []
        _row_label_info.append(
            (g_idx, key, inner_gs, inferred_n_cols, is_scalar)
        )

    # ── Row labels (drawn after subplots are placed) ──────────────────────────
    def _add_group_titles(fig, label_info):
        fig.canvas.draw()
        for g_idx, key, igs, n_cols, is_scalar in label_info:
            all_inner = [
                ax for ax in fig.axes
                if ax.get_subplotspec() is not None
                and ax.get_subplotspec().get_gridspec() == igs
                and ax.get_visible()
            ]
            if not all_inner:
                continue

            y_top   = max(ax.get_position().y1 for ax in all_inner)
            x_left  = min(ax.get_position().x0 for ax in all_inner)
            x_right = max(ax.get_position().x1 for ax in all_inner)
            x_mid   = 0.5 * (x_left + x_right)

            fig.text(
                x_mid,
                y_top + 0.026,
                _pretty_base_label(key),
                ha="center",
                va="bottom",
                fontsize=8.0,
                fontweight="bold",
                transform=fig.transFigure,
            )

    _add_group_titles(fig, _row_label_info)
   
   

    # ── Legend ────────────────────────────────────────────────────────────────
    summary_label = (
        f"{'Geometric mean' if summary_stat == 'mean' else 'Median'} "
        "selected value"
    )

    sampled_proxy = Line2D(
        [], [], marker="o", linestyle="None",
        color="0.75", markerfacecolor="0.75", markeredgecolor="0.75",
        markersize=4, alpha=0.75, label="All sampled settings",
    )
    handles = [
        sampled_proxy,
        Line2D(
            [0], [0], marker="o", linestyle="None",
            color="black", markerfacecolor="white", markeredgecolor="black",
            markersize=3.5, label="Best setting per outer split"),
        Line2D(            [0], [0], color="#2166AC", lw=1.2,
                    marker="o", markersize=3, label="Inner-CV score — median (IQR)"),
    ]
    if include_outer_test:
        handles.append(
            Line2D(
                [0], [0], color="#D6604D", lw=1.0, ls="--",
                marker="s", markersize=3,
                label="Outer-test score — median (IQR)",
            )
        )
    handles.append(
        Line2D(
            [0], [0],
            color=summary_line_color, lw=summary_line_lw,
            ls=summary_line_ls, label=summary_label,
        )
    )
    handles.append(
        Patch(
            facecolor=curve_color, alpha=0.32, linewidth=0,
            label="Selection density (best settings)",
        )
    )

    fig.legend(
        handles=handles,
        handler_map={sampled_proxy: _HandlerJitter()},
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False,
        fontsize=5.8,
        columnspacing=1.2,
        handlelength=2.0,
    )

    # fig.suptitle(
    #     "Variability of the hyperparameters and scores across outer CV splits",
    #     fontsize=8.5, fontweight="bold",
    #     y=0.995)

    fig.savefig(f"{save_path}.pdf", bbox_inches="tight")
    fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
    return fig



###############################################################################

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — violin distributions across splits + mean null CIs
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_pvalue(p: float) -> str:
    if np.isnan(p):
        return "nan"
    if p < 1e-4:
        return "<1e-4"
    if p < 1e-3:
        return "<1e-3"
    if p < 1e-2:
        return f"{p:.3f}"
    return f"{p:.2f}"

def plot_permutation_summary(
    results: dict,
    alpha: float = ALPHA,
    dim_colors: list[str] = DIM_COLORS,
    split_markers: list[str] = SPLIT_MARKERS,
    split_labels: list[str] = SPLIT_LABELS,
    dim_labels: list[str] = DIM_LABELS,
    save_path: str = "fig_permtest_summary",
    rng_seed: int = 0,
) -> plt.Figure:
    """
    Violin plot of observed statistics across CV splits vs. the mean
    permutation-null 95% CI per latent dimension.
    """
    _apply_nature_style()
    rng = np.random.default_rng(rng_seed)

    K = int((~np.isnan(results["obs_corr_train"]).all(axis=0)).sum())

    _dim_colors = (dim_colors * K)[:K]
    _dim_labels = (dim_labels * K)[:K]

    # row, col, observed values, p-values, null CI
    panel_cfg = [
        (0, 0, "obs_corr_train", "p_corr_train", "ci_corr_train"),
        (0, 1, "obs_r2_train", "p_r2_train", "ci_r2_train"),
        (1, 0, "obs_corr_test", "p_corr_test", "ci_corr_test"),
        (1, 1, "obs_r2_test", "p_r2_test", "ci_r2_test"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(5.8, 5.4), sharey="col")
    fig.subplots_adjust(
        left=0.14,
        right=0.98,
        top=0.84,
        bottom=0.24,
        wspace=0.20,
        hspace=0.32,
    )

    for row, col, obs_key, p_key, ci_key in panel_cfg:
        ax = axes[row, col]
        trans = blended_transform_factory(ax.transData, ax.transAxes)

        for k in range(K):
            obs_vals = results[obs_key][:, k]
            p_vals = results[p_key][:, k]
            ci_vals = results[ci_key][:, k, :]
            color = _dim_colors[k]

            valid = ~np.isnan(obs_vals)
            if valid.sum() < 2:
                continue

            obs_valid = obs_vals[valid]
            p_valid = p_vals[valid]

            # Violin
            parts = ax.violinplot(
                obs_valid,
                positions=[k],
                widths=0.55,
                showmedians=True,
                showextrema=False,
            )
            _style_violin(parts, color)

            # Observed 95% interval
            p025 = float(np.nanpercentile(obs_valid, 2.5))
            p975 = float(np.nanpercentile(obs_valid, 97.5))
            ax.plot(
                [k, k],
                [p025, p975],
                color=color,
                lw=1.2,
                solid_capstyle="round",
                zorder=2,
            )

            # Split points
            jitter_width = 0.10
            x_jitter = rng.uniform(-jitter_width, jitter_width, size=valid.sum())
            sig_mask = p_valid < alpha

            ax.scatter(
                k + x_jitter[sig_mask],
                obs_valid[sig_mask],
                s=5,
                color=color,
                linewidths=0,
                alpha=0.80,
                zorder=5,
            )
            ax.scatter(
                k + x_jitter[~sig_mask],
                obs_valid[~sig_mask],
                s=5,
                facecolors="white",
                edgecolors=color,
                linewidths=0.6,
                alpha=0.85,
                zorder=5,
            )

            # Mean null 95% CI band
            mean_lo = float(np.nanmean(ci_vals[:, 0]))
            mean_hi = float(np.nanmean(ci_vals[:, 1]))
            half = 0.27

            ax.fill_between(
                [k - half, k + half],
                mean_lo,
                mean_hi,
                color=color,
                alpha=0.12,
                linewidth=0,
                zorder=0,
            )
            for y in (mean_lo, mean_hi):
                ax.plot(
                    [k - half, k + half],
                    [y, y],
                    color=color,
                    alpha=0.55,
                    lw=0.8,
                    ls="--",
                    zorder=1,
                )

            # % significant annotation
            p_mask = ~np.isnan(p_valid)
            med_stat = float(np.nanmedian(obs_valid))
            med_p = float(np.nanmedian(p_valid))
            pct_sig = (
                float(np.mean(p_valid[p_mask] < alpha)) * 100
                if np.any(p_mask)
                else np.nan
            )

            stat_label = "r" if "corr" in obs_key else "R²"
            pct_text = f"{pct_sig:.0f}% sig" if not np.isnan(pct_sig) else "NA sig"

            ax.text(
                k,
                0.97,
                f"median {stat_label}={med_stat:.2f}\nmedian p={_fmt_pvalue(med_p)}",
                ha="center",
                va="top",
                fontsize=5.2,
                color=color,
                fontweight="bold",
                transform=trans,
            )

            ax.text(
                k,
                0.84,
                pct_text,
                ha="center",
                va="top",
                fontsize=4.8,
                color=color,
                transform=trans,
            )
            
    for col in range(2):
        y_lows = []
        y_highs = []

        for row in range(2):
            lo, hi = axes[row, col].get_ylim()
            y_lows.append(lo)
            y_highs.append(hi)

        lo = min(y_lows)
        hi = max(y_highs)
        yr = hi - lo if hi > lo else 1.0

        new_lo = lo - 0.03 * yr
        new_hi = hi + 0.35 * yr

        axes[0, col].set_ylim(new_lo, new_hi)

    # Column headers: figure-level, centered above each column
    col_titles = ["Canonical correlation", "R²"]
    for col, title in enumerate(col_titles):
        bbox = axes[0, col].get_position()
        x_mid = 0.5 * (bbox.x0 + bbox.x1)
        y_top = bbox.y1 + 0.035
        fig.text(
            x_mid,
            y_top,
            title,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            transform=fig.transFigure,
        )

    # Row labels on left margin
    row_labels = ["Train", "Test"]
    for row, label in enumerate(row_labels):
        bbox = axes[row, 0].get_position()
        y_mid = 0.5 * (bbox.y0 + bbox.y1)
        x_left = bbox.x0 - 0.07
        fig.text(
            x_left,
            y_mid,
            label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=9.0,
            fontweight="bold",
            transform=fig.transFigure,
        )

    # Short dimension legend
    dim_handles = [
        Patch(
            facecolor=_dim_colors[k],
            edgecolor="none",
            alpha=0.75,
            label=_dim_labels[k],
        )
        for k in range(K)
    ]

    # Short encoding legend
    mean_null_handles = [
    Patch(
        facecolor="0.5",
        edgecolor="0.5",
        linestyle="--",
        linewidth=0.5,
        alpha=0.30,
        label="Mean Null 95% CI",
        )
    ]
    
    split_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=4,
            markerfacecolor="0.4",
            markeredgecolor="0.4",
            label=f"Significant (p < {alpha:g})",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=4,
            markerfacecolor="white",
            markeredgecolor="0.4",
            label="Not significant",
        ),
    ]
    
    common_legend_kwargs = dict(
        loc="lower center",
        frameon=False,
        fontsize=6.0,
        title_fontsize=6.4,
    )
    
    fig.subplots_adjust(bottom=0.16)
    
    fig.legend(
        handles=dim_handles,
        bbox_to_anchor=(0.18, 0.01),
        ncol=1,
        title="Latent dimension",
        **common_legend_kwargs,
    )
    
    fig.legend(
        handles=split_handles,
        bbox_to_anchor=(0.50, 0.01),
        ncol=1,
        title="Split",
        **common_legend_kwargs,
    )
    
    fig.legend(
        handles=mean_null_handles,
        bbox_to_anchor=(0.82, 0.01),
        ncol=1,
        **common_legend_kwargs,
    )

    fig.savefig(f"{save_path}.pdf", bbox_inches="tight")
    fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Train vs. test canonical correlation scatter
# ─────────────────────────────────────────────────────────────────────────────


def plot_train_vs_test(
    results: dict,
    alpha: float = ALPHA,
    dim_colors: list[str] = DIM_COLORS,
    split_markers: list[str] = SPLIT_MARKERS,  # kept for API compat
    split_labels: list[str] = SPLIT_LABELS,    # kept for API compat
    dim_labels: list[str] = DIM_LABELS,
    save_path: str = "fig_permtest_train_vs_test",
) -> plt.Figure:
    """
    Scatter plot of train vs. test canonical correlation (one point per split).

    One panel per latent dimension.  Symbol fill encodes joint significance:

    * **Filled**   — significant in both train *and* test (p < ``alpha``).
    * **Open**     — significant in train only.
    * **Grey**     — not significant in train.

    The legend reports the count of splits in each category.  The diagonal
    (identity line) represents perfect generalisation.

    Parameters
    ----------
    results : dict
        Output of either permutation function.
    alpha : float, default 0.05
        Significance threshold for fill encoding.
    dim_colors : list of str
        One colour per latent dimension.
    dim_labels : list of str
        Panel titles for each latent dimension.
    save_path : str
        File stem for PDF and PNG output.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_nature_style()

    S = results["obs_corr_train"].shape[0]
    K = int((~np.isnan(results["obs_corr_train"]).all(axis=0)).sum())

    _dim_colors = (dim_colors * K)[:K]
    _dim_labels = (dim_labels * K)[:K]

    fig, axes = plt.subplots(1, K, figsize=(3.2 * K, 3.2))
    if K == 1:
        axes = [axes]

    for k in range(K):
        ax    = axes[k]
        color = _dim_colors[k]

        obs_tr = results["obs_corr_train"][:, k]
        obs_te = results["obs_corr_test"][:, k]
        p_tr   = results["p_corr_train"][:, k]
        p_te   = results["p_corr_test"][:, k]

        valid      = ~(np.isnan(obs_tr) | np.isnan(obs_te))
        both_sig   = valid & (p_tr < alpha) & (p_te < alpha)
        train_only = valid & (p_tr < alpha) & ~(p_te < alpha)
        neither    = valid & ~(p_tr < alpha)

        categories = [
            (both_sig,   color,     f"Both sig. (n={int(both_sig.sum())})"),
            (train_only, "white",   f"Train only (n={int(train_only.sum())})"),
            (neither,    "#AAAAAA", f"Neither (n={int(neither.sum())})"),
        ]

        for mask, face, label in categories:
            if mask.any():
                ax.scatter(
                    obs_tr[mask], obs_te[mask],
                    s=22, marker="o",
                    facecolors=face, edgecolors=color,
                    linewidths=0.8, zorder=4,
                    label=label,
                )

        # ── Identity / reference lines ───────────────────────────────────────
        if valid.any():
            all_vals = np.concatenate([obs_tr[valid], obs_te[valid]])
            span = all_vals.max() - all_vals.min()
            pad  = span * 0.15 if span > 0 else 0.05
            lims = np.array([all_vals.min() - pad, all_vals.max() + pad])
        else:
            lims = np.array([-0.1, 0.1])

        ax.plot(
            lims, lims,
            ls="--", color="black", lw=0.8, label="Identity", zorder=1,
        )
        ax.axhline(0, color="grey", lw=0.4, ls=":", zorder=0)
        ax.axvline(0, color="grey", lw=0.4, ls=":", zorder=0)

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Train  $r$", fontsize=6)
        ax.set_ylabel("Test  $r$", fontsize=6)
        ax.set_title(_dim_labels[k], fontsize=7, fontweight="bold", color=color)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_aspect("equal")
        ax.legend(fontsize=5, frameon=False, loc="lower right")

    fig.suptitle(
        "Train vs. test canonical correlation\n"
        "(diagonal = perfect generalisation)",
        fontsize=7, fontweight="bold",
    )
    fig.subplots_adjust(
        left=0.10, right=0.97,
        top=0.82, bottom=0.15,
        wspace=0.38,
    )

    fig.savefig(f"{save_path}.pdf", bbox_inches="tight")
    fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
    plt.show()
    return fig

