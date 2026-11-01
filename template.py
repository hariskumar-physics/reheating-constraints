"""
Reheating-constraint plotting template (single-field slow-roll inflation)
=========================================================================

Give it ANY potential V(phi) and ANY n_s observables (Gaussian summary or a
Cobaya/GetDist chain); it produces

  fig1_reheating_vs_ns_<tag>.png   T_reh(n_s) [log] and N_reh(n_s), several w_reh,
                                    observable bands in the background, BBN line
  fig2_dN_phi_vs_ns.png            Delta N_*(n_s) and phi_*/M_Pl (n_s), all
                                    parameter sets overlaid, bands in background
  fig3_Treh_Nreh_w_<tag>.png       T_reh vs N_reh (several w_reh) and
                                    N_reh vs w_reh (several fixed T_reh)
  results.txt                      instant-reheating point and the T_reh / N_reh
                                    implied by each observable's n_s interval

Usage:  python reheating_template.py
To change the model or data edit ONLY section 1 (CONFIGURATION).

--------------------------------------------------------------------------
PHYSICS (units M_Pl = 1 inside the potential code; reduced Planck mass)
--------------------------------------------------------------------------
Potential slow-roll (first order):
    eps = (1/2)(V'/V)^2,  eta = V''/V,  n_s = 1 - 6 eps + 2 eta,  r = 16 eps.
End of inflation: eps(phi_end) = 1 (root found numerically).
e-folds from horizon exit of the pivot k to the end of inflation:
    N_k = integral_{phi_end}^{phi_*} (V/V') dphi        (signed integral; works
                                                          for either roll direction)
Normalisation:  P_s = V/(24 pi^2 eps) = A_s   =>   V_* = 24 pi^2 A_s eps_*,
                H_k = pi sqrt(8 eps_* A_s)                    (in M_Pl).

Reheating (constant effective w, instantaneous thermalisation at the end):
    rho_reh = rho_end exp[-3(1+w) N_reh],   rho_end = c * V_end,
    rho_reh = (pi^2/30) g_reh T_reh^4
    =>  T_reh = T_max exp[-3(1+w) N_reh / 4],
        T_max = (30 rho_end / (pi^2 g_reh))^(1/4).
(c = 3/2 for a canonical field: at eps_H = 1, K = rho/3, V = 2 rho/3.)

Horizon-crossing condition for the pivot k = a_k H_k:
    k/(a_0 H_0) = e^{-N_k} e^{-N_reh} (a_reh/a_0) (H_k/H_0),
    entropy conservation after reheating:
    a_reh/a_0 = (T_0/T_reh) (g_s0/g_reh)^{1/3}.
Eliminating T_reh with the relation above and defining
    B = ln(T_0/T_max) + (1/3) ln(g_s0/g_reh) + ln(H_k/H_0)
        - ln(k/(a_0 H_0)) - N_k ,
one finds   N_reh (1 - 3w)/4 = B,   i.e.

    N_reh = 4 B / (1 - 3 w),      T_reh = T_max exp[-3(1+w) N_reh / 4].

(w = 1/3 is degenerate: reheating is then unobservable and B = 0 is forced.)
Instant reheating (N_reh = 0) is the zero of B and fixes one n_s for ALL w.

--------------------------------------------------------------------------
LIMITATIONS
--------------------------------------------------------------------------
- First-order slow roll; V_end taken at eps_V = 1.
- rho_end = c * V_end: set RHO_END_FACTOR per model. For non-canonical
  (k-inflation) models c differs; supply the correct value.
- Chain bands are weighted quantiles of the supplied chain; verify its datasets
  from its input YAML. Published Gaussian values are only used when no chain is
  found, and the band label says so.
"""

from pathlib import Path
from collections import Counter
import re
from fractions import Fraction

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

try:
    import sympy as sp
except ImportError:  # only needed if V is given as a string
    sp = None


# ============================================================
# 1. CONFIGURATION  (edit this section only)
# ============================================================

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "reheating_results"

ACTIVE_MODEL = "large_field"          # key of MODELS below

# ---------- cosmology / constants ----------
COSMO = dict(
    h=0.6736,            # H_0 = 100 h km/s/Mpc
    A_s=2.1e-9,          # scalar amplitude at the pivot
    k_pivot_Mpc=0.05,    # pivot scale [Mpc^-1]
    T0_K=2.7255,         # CMB temperature today [K]
    g_s0=3.91,           # entropy d.o.f. today
    g_reh=106.75,        # relativistic d.o.f. at reheating (SM)
)
M_PL_GEV = 2.435e18      # reduced Planck mass [GeV]
T_BBN_GEV = 4.0e-3       # BBN lower bound on T_reh [GeV] (4 MeV)

# ---------- models ----------
# V        : string in `phi` and parameter names (sympy syntax), OR a tuple of
#            three callables (V, dV, d2V), each f(phi, **params).
# param_sets: list of dicts; one curve (Fig 2) / one file (Figs 1, 3) each.
# side     : +1 if phi_* > phi_end (field rolls to smaller phi), -1 otherwise.
# phi_far  : scan start deep in slow roll (number or function of params).
# phi_near : a point on the "end" side where eps > 1 (number or function).
# rho_end_factor: rho_end = factor * V_end  (number or function of params).
MODELS = {
    "tmodel": dict(
        name="T-model alpha-attractor",
        V="tanh(phi/sqrt(6*alpha))**2",
        tex=r"$V=V_0\tanh^2\!\left(\frac{\phi}{\sqrt{6\alpha}\,M_{\rm Pl}}\right)$",
        param_sets=[{"alpha": 0.1}, {"alpha": 1.0}, {"alpha": 10.0}],
        side=+1,
        phi_near=1e-3,
        phi_far=lambda p: 6.0 * np.sqrt(6.0 * p["alpha"]),
        rho_end_factor=1.5,
    ),
    "chaotic": dict(
        name="Large-field (chaotic) inflation",
        V="phi**n",
        tex=r"$V=V_0\,\phi^{n}$",
        param_sets=[{"n": 2.0}],
        side=+1,
        phi_near=1e-3,
        phi_far=lambda p: 25.0 * np.sqrt(p["n"]),
        rho_end_factor=1.5,
    ),
    "large_field": dict(
    name="Quadratic large-field inflation",
    V="0.5*phi**2",
    tex=r"$V=\frac{1}{2}m^2\phi^2$",
    param_sets=[{}],
    side=+1,
    phi_near=1e-3,
    phi_far=20.0,
    rho_end_factor=1.5,
    ),
    "starobinsky": dict(
        name="Starobinsky inflation",
        V="(1-exp(-sqrt(2/3)*phi))**2",
        tex=r"$V=V_0\left(1-e^{-\sqrt{2/3}\,\phi/M_{\rm Pl}}\right)^2$",
        param_sets=[{}],
        side=+1,
        phi_near=1e-3,
        phi_far=15.0,
        rho_end_factor=1.5,
    ),
    "natural": dict(
        name="Natural inflation",
        V="1+cos(phi/f)",
        tex=r"$V=V_0\left[1+\cos(\phi/f)\right]$",
        param_sets=[{"f": 5.0}, {"f": 10.0}],
        side=-1,                                  # inflation near the hilltop
        phi_near=lambda p: 3.14 * p["f"],
        phi_far=lambda p: 1e-3 * p["f"],
        rho_end_factor=1.5,
    ),
}

# ---------- observables (n_s bands) ----------
# Either gaussian=(mean, sigma) or chain=<path> (weighted quantiles are used;
# `gaussian` is then only the fallback when the chain file does not exist).
OBSERVABLES = [
    dict(label="Planck 2018", gaussian=(0.9649, 0.0042), color="0.30"),
    dict(
        label="ACT DR6 + DESI",
        chain=ROOT / "chains" / "pactlb.1.txt",
        paramnames_files=[
            ROOT / "chains" / "pactlb.paramnames",
            ROOT / "chains" / "pactlb.1.paramnames",
            ROOT / "pactlb.paramnames",
        ],
        ns_column=3,                       # fallback; 0=weight 1=-logpost 2=logA 3=ns
        gaussian=(0.9743, 0.0034),         # published P-ACT-LB, used only if no chain
        color="tab:orange",
    ),
]

# ---------- plot settings ----------
W_FIG1 = [-1/3, -1/6, 0.0, 1/6, 2/3, 1.0]            # Fig 1 (viridis)
W_FIG3 = [-1/3, -1/6, 0.0, 1/6, 2/3, 1.0]            # Fig 3 left
T_FIG3 = [1e15, 1e12, 1e9, 1e6, 1e2, 1e-2]           # Fig 3 right [GeV]
NS_XLIM_FIG1 = (0.93, 0.99)
NS_XLIM_FIG2 = (0.90, 0.99)
FIG2_N_YLIM = (10, 120)
FIG2_PHI_YLIM = None                                  # None = automatic
N_GRID = 40000                                        # points along phi_*
RUN_SELFTESTS = True


# ============================================================
# 2. POTENTIAL AND SLOW-ROLL BACKGROUND
# ============================================================

def _resolve(value, params):
    return value(params) if callable(value) else value


def _vectorised(f):
    def g(x):
        x = np.asarray(x, dtype=float)
        return np.asarray(f(x), dtype=float) * np.ones_like(x)
    return g


def make_potential(model, params):
    """Return callables V, V', V'' for the given parameter values."""
    spec = model["V"]

    if isinstance(spec, str):
        if sp is None:
            raise ImportError("sympy is required when V is given as a string "
                              "(pip install sympy), or pass three callables.")
        phi = sp.Symbol("phi", real=True)
        expr = sp.sympify(spec, locals={"phi": phi})
        free = [s for s in expr.free_symbols if s != phi]
        missing = [s.name for s in free if s.name not in params]
        if missing:
            raise ValueError(f"Parameters {missing} appear in V but not in "
                             f"the parameter set {params}.")
        expr = expr.subs({s: float(params[s.name]) for s in free})
        funcs = [sp.lambdify(phi, e, "numpy")
                 for e in (expr, sp.diff(expr, phi), sp.diff(expr, phi, 2))]
        return tuple(_vectorised(f) for f in funcs)

    V, dV, d2V = spec
    return tuple(_vectorised(lambda x, f=f: f(x, **params)) for f in (V, dV, d2V))


def find_phi_end(V, dV, phi_near, phi_far, n=400001):
    """First point where eps = 1 when moving from phi_far (slow roll) inward."""
    grid = np.linspace(phi_far, phi_near, n)
    eps = 0.5 * (dV(grid) / V(grid)) ** 2
    g = eps - 1.0
    if g[0] >= 0:
        raise ValueError("eps >= 1 at phi_far: move phi_far deeper into the "
                         "slow-roll region.")
    idx = np.flatnonzero(g >= 0)
    if idx.size == 0:
        raise ValueError("eps never reaches 1 between phi_far and phi_near: "
                         "adjust phi_near (and check `side`).")
    i = idx[0]
    f = lambda x: 0.5 * (dV(x) / V(x)) ** 2 - 1.0
    return brentq(f, grid[i - 1], grid[i], xtol=1e-14, rtol=1e-13)


def build_background(model, params):
    """Slow-roll quantities along phi_* from phi_end to phi_far."""
    V, dV, d2V = make_potential(model, params)
    side = model["side"]
    phi_near = _resolve(model["phi_near"], params)
    phi_far = _resolve(model["phi_far"], params)

    phi_end = find_phi_end(V, dV, phi_near, phi_far)
    span = abs(phi_far - phi_end)
    dist = np.concatenate(([0.0], np.geomspace(1e-7 * span, span, N_GRID)))
    phi = phi_end + side * dist

    v, dv, d2v = V(phi), dV(phi), d2V(phi)
    N = cumulative_trapezoid(v / dv, phi, initial=0.0)
    if N[-1] <= 0:
        raise ValueError("N_k came out negative: the `side` setting is wrong.")

    eps = 0.5 * (dv / v) ** 2
    eta = d2v / v
    sl = slice(1, None)                     # drop phi_* = phi_end (N = 0)
    return dict(
        phi=phi[sl], N=N[sl], eps=eps[sl], eta=eta[sl],
        ns=1 - 6 * eps[sl] + 2 * eta[sl], r=16 * eps[sl],
        Vend_over_Vstar=v[0] / v[sl],
        phi_end=phi_end,
    )


# ============================================================
# 3. REHEATING
# ============================================================

def cosmo_scales():
    h = COSMO["h"]
    return dict(
        H0=2.1332e-42 * h,                                   # GeV
        T0=COSMO["T0_K"] * 8.617333262e-14,                  # GeV
        lnk=np.log(COSMO["k_pivot_Mpc"] * 2997.92458 / h),   # ln[k/(a0 H0)]
    )


def reheating_background(bg, rho_factor):
    """T_max and the bracket B (see module docstring) along phi_*."""
    c = cosmo_scales()
    A_s, g_reh, g_s0 = COSMO["A_s"], COSMO["g_reh"], COSMO["g_s0"]

    Hk = np.pi * M_PL_GEV * np.sqrt(8.0 * bg["eps"] * A_s)             # GeV
    Vstar = 24.0 * np.pi**2 * A_s * bg["eps"] * M_PL_GEV**4            # GeV^4
    rho_end = rho_factor * Vstar * bg["Vend_over_Vstar"]
    Tmax = (30.0 * rho_end / (np.pi**2 * g_reh)) ** 0.25

    B = (np.log(c["T0"] / Tmax) + np.log(g_s0 / g_reh) / 3.0
         + np.log(Hk / c["H0"]) - c["lnk"] - bg["N"])
    return dict(Tmax=Tmax, B=B)


def reheating_vs_w(rb, w):
    """Raw (unmasked) N_reh and T_reh along phi_* for a given w."""
    if abs(w - 1.0 / 3.0) < 1e-9:
        raise ValueError("w = 1/3 is degenerate (N_reh undetermined).")
    N_reh = 4.0 * rb["B"] / (1.0 - 3.0 * w)
    lnT = np.log(rb["Tmax"]) - 0.75 * (1.0 + w) * N_reh
    T_reh = np.exp(np.clip(lnT, -700, 700))
    return N_reh, T_reh


def instant_reheating_point(bg, rb):
    """Interpolated point where B = 0 (N_reh = 0, independent of w)."""
    B = rb["B"]
    idx = np.flatnonzero(B[:-1] * B[1:] < 0)
    if idx.size == 0:
        i = int(np.argmin(np.abs(B)))
        print("  WARNING: B has no zero on the grid; using the closest point.")
        t, j = 0.0, i
    else:
        j = idx[0]
        t = B[j] / (B[j] - B[j + 1])
    lerp = lambda a: a[j] + t * (a[min(j + 1, len(a) - 1)] - a[j])
    return dict(ns=lerp(bg["ns"]), N=lerp(bg["N"]), phi=lerp(bg["phi"]),
                r=lerp(bg["r"]), Tmax=lerp(rb["Tmax"]))


def at_ns(ns_arr, y, ns_val):
    """Interpolate y along the (monotonic) n_s branch."""
    order = np.argsort(ns_arr)
    return float(np.interp(ns_val, ns_arr[order], y[order],
                           left=np.nan, right=np.nan))


# ============================================================
# 4. OBSERVABLES (n_s bands)
# ============================================================

def gaussian_summary(mean, sigma):
    return dict(lower95=mean - 2 * sigma, lower68=mean - sigma, median=mean,
                upper68=mean + sigma, upper95=mean + 2 * sigma)


def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    cum /= cum[-1]
    return np.interp(q, cum, v)


def read_numeric_rows(path):
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                vals = [float(x) for x in s.split()]
            except ValueError:
                continue
            if len(vals) >= 4 and np.all(np.isfinite(vals)):
                rows.append(vals)
    if not rows:
        raise ValueError(f"No usable numeric rows in {path}")
    ncol = Counter(len(r) for r in rows).most_common(1)[0][0]
    good = [r for r in rows if len(r) == ncol]
    if len(good) < 10:
        raise ValueError("Too few consistently formatted rows in the chain.")
    print(f"  chain rows kept: {len(good)} ({len(rows) - len(good)} skipped), "
          f"{ncol} columns")
    return np.asarray(good, float)


def ns_column_index(data, obs):
    """paramnames file first (exactly one 'ns'), else the configured column."""
    ncol = data.shape[1]
    col = None
    for p in obs.get("paramnames_files", []):
        if not Path(p).is_file():
            continue
        names = []
        for line in Path(p).read_text(errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line.split()[0].rstrip("*").lower())
        if len(names) == ncol - 2 and names.count("ns") == 1:
            col = 2 + names.index("ns")
            print(f"  n_s column from {p} -> index {col}")
        break
    if col is None:
        col = obs.get("ns_column", 3)
        print(f"  n_s column from configured fallback -> index {col}")
    if not 0 <= col < ncol:
        raise ValueError(f"n_s column {col} invalid for {ncol} columns.")
    med = np.median(data[:, col])
    if not 0.8 < med < 1.2:
        raise ValueError(f"Selected n_s column has median {med:.4g}; wrong "
                         "column mapping.")
    return col


def observable_summary(obs):
    """Return (summary dict, label). Chain if available, else Gaussian."""
    chain = obs.get("chain")
    if chain is not None and Path(chain).is_file():
        print(f"\n{obs['label']}: reading chain {chain}")
        data = read_numeric_rows(Path(chain))
        col = ns_column_index(data, obs)
        w, ns = data[:, 0], data[:, col]
        ok = np.isfinite(w) & (w > 0) & np.isfinite(ns)
        w, ns = w[ok] / w[ok].sum(), ns[ok]
        if np.any((ns < 0.5) | (ns > 1.5)):
            raise ValueError("n_s values outside [0.5, 1.5]: bad column/chain.")
        q = weighted_quantile(ns, w, [0.02275, 0.15865, 0.5, 0.84135, 0.97725])
        s = dict(lower95=q[0], lower68=q[1], median=q[2],
                 upper68=q[3], upper95=q[4])
        return s, f"{obs['label']} (chain)"

    if "gaussian" not in obs:
        raise FileNotFoundError(f"{obs['label']}: chain missing and no "
                                "gaussian fallback given.")
    if chain is not None:
        print(f"\nWARNING: chain {chain} not found -> using the published "
              f"Gaussian summary for '{obs['label']}'.")
    return gaussian_summary(*obs["gaussian"]), f"{obs['label']} (published)"


# ============================================================
# 5. PLOTTING HELPERS
# ============================================================

def style_axes(ax):
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=7)
    ax.tick_params(which="minor", length=3.5)


def draw_bands(ax, observables):
    """2 sigma (light) and 1 sigma (dark) n_s bands, one colour per dataset."""
    for o in observables:
        s = o["summary"]
        ax.axvspan(s["lower95"], s["upper95"], color=o["color"], alpha=0.18,
                   lw=0, zorder=0)
        ax.axvspan(s["lower68"], s["upper68"], color=o["color"], alpha=0.40,
                   lw=0, zorder=0)


def band_handles(observables):
    return [Patch(facecolor=o["color"], alpha=0.40,
                  label=o["full_label"] + r"  $1\sigma$ (dark) / $2\sigma$ (light)")
            for o in observables]


def w_tex(w, bar=False):
    """LaTeX label for w_reh, using simple fractions where possible."""
    f = Fraction(w).limit_denominator(6)
    if abs(float(f) - w) < 1e-9 and f.denominator > 1:
        val = (r"-" if f < 0 else "") + rf"{abs(f.numerator)}/{f.denominator}"
    else:
        val = f"{w:.3g}"
    name = r"\overline{w}_{\rm reh}" if bar else r"w_{\rm reh}"
    return rf"${name}={val}$"


def pset_label(params):
    return ", ".join(f"{k}={v:g}" for k, v in params.items())


def pset_tag(params):
    if not params:
        return "default"
    return re.sub(r"[^A-Za-z0-9_.=-]", "", "_".join(f"{k}{v:g}" for k, v in params.items()))


def annotate(ax, model, params, y=0.97, va="top"):
    txt = model["tex"]
    if params:
        txt += "\n" + pset_label(params)
    ax.text(0.03, y, txt, transform=ax.transAxes, va=va, fontsize=11)


# ---------------------------------------------------------------
# Figure 2:  Delta N_*(n_s) and phi_*/M_Pl (n_s)
# ---------------------------------------------------------------
def figure2(model, backgrounds, observables, path):
    fig, (axN, axP) = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax in (axN, axP):
        draw_bands(ax, observables)

    single = len(backgrounds) == 1
    colors = ["red", "limegreen", "royalblue", "darkorange", "purple", "brown"]
    styles = ["-", "--", "-.", ":"]
    phis = []

    for i, (params, bg) in enumerate(backgrounds):
        vis = (bg["ns"] >= NS_XLIM_FIG2[0]) & (bg["ns"] <= NS_XLIM_FIG2[1])
        if not vis.any():
            print(f"  (Fig 2) no n_s in window for {pset_label(params)}")
            continue
        lab = pset_label(params) or model["name"]
        if single:      # same look as the single-model reference figure
            axN.plot(bg["ns"][vis], bg["N"][vis], color="red", lw=2.2, label=lab)
            axP.plot(bg["ns"][vis], np.abs(bg["phi"][vis]), color="lime", lw=2.8,
                     ls=":", label=lab)
        else:
            c, ls = colors[i % len(colors)], styles[i % len(styles)]
            axN.plot(bg["ns"][vis], bg["N"][vis], color=c, ls=ls, lw=2.2, label=lab)
            axP.plot(bg["ns"][vis], np.abs(bg["phi"][vis]), color=c, ls=ls,
                     lw=2.2, label=lab)
        phis.append(np.abs(bg["phi"][vis]))

    axN.set_ylim(*FIG2_N_YLIM)
    if FIG2_PHI_YLIM:
        axP.set_ylim(*FIG2_PHI_YLIM)
    elif phis:
        allp = np.concatenate(phis)
        pad = 0.05 * (allp.max() - allp.min() + 1e-12)
        axP.set_ylim(max(0.0, allp.min() - pad), allp.max() + pad)

    axN.set_ylabel(r"$\Delta N_*$", fontsize=14)
    axP.set_ylabel(r"$\phi_*/M_{\rm Pl}$", fontsize=14)
    for ax in (axN, axP):
        ax.set_xlim(*NS_XLIM_FIG2)
        ax.set_xlabel(r"$n_s$", fontsize=14)
        style_axes(ax)
        ax.text(0.03, 0.97, model["tex"], transform=ax.transAxes, va="top",
                fontsize=11)

    h, _ = axN.get_legend_handles_labels()
    axN.legend(handles=h + band_handles(observables), loc="upper left",
               bbox_to_anchor=(0.02, 0.88), fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------
# Figure 1:  T_reh(n_s) and N_reh(n_s)
# ---------------------------------------------------------------
def figure1(model, params, bg, rb, observables, inst, path):
    fig, (axT, axN) = plt.subplots(1, 2, figsize=(14, 5.4))
    for ax in (axT, axN):
        draw_bands(ax, observables)

    cmap = plt.cm.viridis(np.linspace(0, 1, len(W_FIG1)))
    ns = bg["ns"]
    win = (ns >= NS_XLIM_FIG1[0] - 0.02) & (ns <= NS_XLIM_FIG1[1] + 0.02)

    for w, col in zip(W_FIG1, cmap):
        N_reh, T_reh = reheating_vs_w(rb, w)
        phys = win & (N_reh >= 0)
        lab = w_tex(w)
        axT.semilogy(ns[phys], T_reh[phys], color=col, lw=2.4, label=lab)
        axN.plot(ns[phys], N_reh[phys], color=col, lw=2.4, label=lab)

    axT.axhline(T_BBN_GEV, color="k", ls="--", lw=1.4)
    axT.text(NS_XLIM_FIG1[0] + 0.0015, T_BBN_GEV * 1.6, "BBN", fontsize=11)

    ytop = 10 ** np.ceil(np.log10(inst["Tmax"] * 3))
    axT.set_ylim(1e-3, ytop)
    axN.set_ylim(0, 60)
    axT.set_ylabel(r"$T_{\rm reh}$ [GeV]", fontsize=14)
    axN.set_ylabel(r"$N_{\rm reh}$", fontsize=14)
    for ax in (axT, axN):
        ax.set_xlim(*NS_XLIM_FIG1)
        ax.set_xlabel(r"$n_s$", fontsize=14)
        style_axes(ax)
        annotate(ax, model, params)

    axT.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86), ncol=2,
               fontsize=9, framealpha=0.9)
    axN.legend(handles=band_handles(observables), loc="lower right",
               fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------
# Figure 3:  T_reh vs N_reh  and  N_reh vs w_reh
# ---------------------------------------------------------------
def figure3(model, params, Tmax, path):
    colors = ["red", "cyan", "blue", "saddlebrown", "lime", "darkorange"]
    styles = ["-", (0, (10, 4)), (0, (4, 2)), "-.", ":", (0, (6, 3))]
    lw = 3.0

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.4))

    # left: T_reh(N_reh) = T_max exp[-3(1+w) N/4]
    N = np.linspace(0, 90, 2000)
    for w, c, ls in zip(W_FIG3, colors, styles):
        T = Tmax * np.exp(-0.75 * (1 + w) * N)
        keep = T >= 1e-2
        axL.semilogy(N[keep], T[keep], color=c, ls=ls, lw=lw,
                     label=w_tex(w, bar=True))
    axL.set_xlim(0, 90)
    axL.set_ylim(1e-2, 10 ** np.ceil(np.log10(Tmax * 2)))
    axL.set_xlabel(r"Reheating e-folds ($N_{\rm reh}$)", fontsize=14)
    axL.set_ylabel(r"$T_{\rm reh}$[GeV]", fontsize=14)
    axL.legend(loc="upper right", fontsize=10, framealpha=0.95, edgecolor="0.2")

    # right: N_reh(w) = 4 ln(T_max/T) / [3(1+w)]
    w = np.linspace(-1 / 3, 1.0, 400)
    for T0, c, ls in zip(T_FIG3, colors, styles):
        Nw = 4.0 * np.log(Tmax / T0) / (3.0 * (1.0 + w))
        Nw = np.where(Nw >= 0, Nw, np.nan)           # T_reh > T_max is unphysical
        exp10 = int(np.round(np.log10(T0)))
        if T0 >= 1e3 or T0 < 1e-1:
            lab = rf"$T_{{\rm reh}}\!\sim\!10^{{{exp10}}}$ GeV"
        else:
            lab = rf"$T_{{\rm reh}}\!\sim\!{T0:g}$ GeV"
        if abs(T0 - 1e-2) < 1e-12:
            lab = r"$T_{\rm reh}\!\sim\!10$ MeV"
        axR.plot(w, Nw, color=c, ls=ls, lw=lw, label=lab)
    axR.set_xlim(-1 / 3 - 0.02, 1.0)
    axR.set_ylim(0, 100)
    axR.set_xlabel(r"$\overline{w}_{\rm reh}$", fontsize=14)
    axR.set_ylabel(r"Reheating e-folds ($N_{\rm reh}$)", fontsize=14)
    axR.legend(loc="upper right", fontsize=10, framealpha=0.95, edgecolor="0.2")

    for ax in (axL, axR):
        style_axes(ax)
    annotate(axL, model, params, y=0.05, va="bottom")   # empty lower-left corner
    annotate(axR, model, params, y=0.93)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================
# 6. RESULTS FILE
# ============================================================

def write_results(model, runs, observables, path):
    L = [f"REHEATING ANALYSIS: {model['name']}", "=" * 60, "",
         "Inputs", "------",
         f"h={COSMO['h']}, A_s={COSMO['A_s']:.3e}, k_pivot={COSMO['k_pivot_Mpc']} /Mpc, "
         f"g_reh={COSMO['g_reh']}, g_s0={COSMO['g_s0']}, T_BBN={T_BBN_GEV:g} GeV", ""]

    L += ["Observables (n_s)", "-----------------"]
    for o in observables:
        s = o["summary"]
        L.append(f"{o['full_label']}: median {s['median']:.5f}, 68% "
                 f"[{s['lower68']:.5f}, {s['upper68']:.5f}], 95% "
                 f"[{s['lower95']:.5f}, {s['upper95']:.5f}]")
    L.append("")

    for params, bg, rb, inst in runs:
        L += [f"Parameter set: {pset_label(params) or 'default'}",
              "-" * 60,
              f"phi_end/M_Pl = {bg['phi_end']:.6f}",
              "Instant reheating (N_reh = 0, same for every w):",
              f"  n_s = {inst['ns']:.6f}   N_k = {inst['N']:.3f}   "
              f"phi_*/M_Pl = {abs(inst['phi']):.4f}   r = {inst['r']:.4e}   "
              f"T_max = {inst['Tmax']:.4e} GeV", ""]
        for o in observables:
            s = o["summary"]
            L.append(f"  T_reh / N_reh at the {o['full_label']} n_s values "
                     "(N_reh < 0 = excluded; * = T_reh below BBN)")
            hdr = f"  {'w_reh':>7} | " + " | ".join(
                f"{k:^23}" for k in ("2s low", "1s low", "median", "1s up", "2s up"))
            L.append(hdr)
            for w in W_FIG1:
                N_reh, T_reh = reheating_vs_w(rb, w)
                cells = []
                for key in ("lower95", "lower68", "median", "upper68", "upper95"):
                    n = at_ns(bg["ns"], N_reh, s[key])
                    T = at_ns(bg["ns"], T_reh, s[key])
                    if np.isnan(n):
                        cells.append(f"{'outside grid':^23}")
                    elif n < 0:
                        cells.append(f"{'N_reh<0':^23}")
                    else:
                        flag = "*" if T < T_BBN_GEV else " "
                        cells.append(f"{n:6.1f} {T:9.2e}{flag:1}".center(23))
                L.append(f"  {w:7.3f} | " + " | ".join(cells))
            L.append("  (each cell: N_reh, T_reh [GeV])")
            L.append("")
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"Saved: {path}")


# ============================================================
# 7. SELF-TESTS (numerics vs analytic slow-roll results)
# ============================================================

def run_selftests():
    # (a) V = phi^2: N = phi^2/4 - 1/2, n_s = 1 - 2/(N + 1/2), r = 8/(N + 1/2)
    m = dict(V="phi**n", side=+1, phi_near=1e-3, phi_far=40.0)
    bg = build_background(m, {"n": 2.0})
    ns60 = at_ns(bg["N"], bg["ns"], 60.0)
    r60 = at_ns(bg["N"], bg["r"], 60.0)
    assert abs(ns60 - (1 - 2 / 60.5)) < 2e-5, ("phi^2 n_s", ns60)
    assert abs(r60 - 8 / 60.5) < 2e-4, ("phi^2 r", r60)

    # (b) T-model: cosh(2x*) = cosh(2x_end) + 4N/(3 alpha), eps = 4/(3 alpha sinh^2 2x*)
    for a in (0.1, 1.0, 10.0):
        m = dict(V="tanh(phi/sqrt(6*alpha))**2", side=+1, phi_near=1e-3,
                 phi_far=6 * np.sqrt(6 * a))
        bg = build_background(m, {"alpha": a})
        c_end = np.sqrt(1 + 4 / (3 * a))
        c_star = c_end + 4 * 60.0 / (3 * a)
        eps = 4 / (3 * a * (c_star**2 - 1))
        x = 0.5 * np.arccosh(c_star)
        eta = (1 / np.cosh(x) ** 2) * (1 - 3 * np.tanh(x) ** 2) / (3 * a * np.tanh(x) ** 2)
        ns_an = 1 - 6 * eps + 2 * eta
        ns_num = at_ns(bg["N"], bg["ns"], 60.0)
        assert abs(ns_num - ns_an) < 5e-5, ("T-model n_s", a, ns_num, ns_an)
    print("Self-tests passed (phi^2 and T-model match analytic slow roll).")


# ============================================================
# 8. MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if RUN_SELFTESTS:
        run_selftests()

    model = MODELS[ACTIVE_MODEL]
    print("=" * 60)
    print(f"Model: {model['name']}")
    print("=" * 60)

    observables = []
    for o in OBSERVABLES:
        summary, full_label = observable_summary(o)
        observables.append(dict(color=o["color"], summary=summary,
                                full_label=full_label))
        print(f"  {full_label}: median {summary['median']:.5f}, "
              f"68% [{summary['lower68']:.5f}, {summary['upper68']:.5f}]")

    runs, backgrounds = [], []
    for params in model["param_sets"]:
        print(f"\nParameter set: {pset_label(params) or 'default'}")
        bg = build_background(model, params)
        rho_factor = _resolve(model["rho_end_factor"], params)
        rb = reheating_background(bg, rho_factor)
        inst = instant_reheating_point(bg, rb)
        print(f"  phi_end = {bg['phi_end']:.5f} M_Pl;  instant reheating: "
              f"n_s = {inst['ns']:.5f}, N_k = {inst['N']:.2f}, "
              f"T_max = {inst['Tmax']:.3e} GeV")
        runs.append((params, bg, rb, inst))
        backgrounds.append((params, bg))

        tag = pset_tag(params)
        figure1(model, params, bg, rb, observables, inst,
                OUTPUT_DIR / f"fig1_reheating_vs_ns_{tag}.png")
        figure3(model, params, inst["Tmax"],
                OUTPUT_DIR / f"fig3_Treh_Nreh_w_{tag}.png")

    figure2(model, backgrounds, observables, OUTPUT_DIR / "fig2_dN_phi_vs_ns.png")
    write_results(model, runs, observables, OUTPUT_DIR / "results.txt")
    print(f"\nDone. Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()