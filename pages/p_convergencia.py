"""
Página: Gráficas de Convergencia
Visualización dedicada y detallada de la convergencia de los tres
algoritmos sobre las cuatro funciones benchmark.

Modos:
  - Demo rápida: ejecuta pocas corridas al vuelo para mostrar gráficas de
    inmediato sin necesidad de haber pasado por el Experimento Completo.
  - Datos de sesión: usa los resultados ya calculados en p2_experimento.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from harmony_search import (
    harmony_search_adaptive,
    genetic_algorithm,
    de_best_1_bin,
    PROBLEMS,
)

COLOR_HS = "#534AB7"
COLOR_GA = "#D85A30"
COLOR_DE = "#1D9E75"
ALGOS   = ["HS_adapt", "GA", "DE_best"]
LABELS  = {"HS_adapt": "HS Adaptativo", "GA": "Alg. Genético", "DE_best": "DE/best/1/bin"}
COLORS  = {"HS_adapt": COLOR_HS, "GA": COLOR_GA, "DE_best": COLOR_DE}
LS      = {"HS_adapt": "-", "GA": "--", "DE_best": ":"}
LW      = {"HS_adapt": 2.2, "GA": 1.8, "DE_best": 1.8}


# ──────────────────────────────────────────────────────────────────────────────
def render():
    st.header(" Gráficas de Convergencia")
    st.markdown("""
    <div class="info-box">
    Visualización detallada de la <b>evolución del mejor fitness</b> a lo largo
    de las evaluaciones de la función objetivo para los tres algoritmos sobre
    las cuatro funciones benchmark.<br>
    Puedes usar los resultados del <b>Experimento Completo</b> (si ya los generaste)
    o lanzar una <b>Demo rápida</b> directamente desde esta página.
    </div>
    """, unsafe_allow_html=True)

    # ── Fuente de datos ───────────────────────────────────────────────────────
    tiene_sesion = "historiales" in st.session_state and "resultados" in st.session_state

    col_src, col_cfg = st.columns([1, 2])
    with col_src:
        fuente = st.radio(
            "Fuente de datos",
            [" Datos de sesión (p2)", " Demo rápida (aquí)"],
            index=0 if tiene_sesion else 1,
            help="'Datos de sesión' requiere haber ejecutado el Experimento Completo.",
        )
    with col_cfg:
        if fuente == " Demo rápida (aquí)":
            c1, c2, c3 = st.columns(3)
            with c1:
                demo_runs = st.slider("Ejecuciones demo", 3, 15, 5, 1)
                dim       = st.slider("Dimensiones", 2, 20, 10)
            with c2:
                max_iter  = st.slider("Iter. HS", 200, 5000, 1000, 200)
                hms       = st.slider("HMS", 10, 100, 50, 10)
            with c3:
                pop_size  = st.slider("Población AG/DE", 20, 200, 100, 10)
                funcs_sel = st.multiselect(
                    "Funciones", list(PROBLEMS.keys()),
                    default=list(PROBLEMS.keys()),
                )
            lanzar = st.button(" Generar gráficas", type="primary",
                               use_container_width=True)
        else:
            lanzar = False
            funcs_sel = list(st.session_state.get("resultados", {}).keys())
            max_iter  = st.session_state.get("max_iter", 5000)
            pop_size  = st.session_state.get("pop_size", 100)
            dim       = 10

    # ── Obtención / generación de datos ──────────────────────────────────────
    if fuente == " Datos de sesión (p2)":
        if not tiene_sesion:
            st.markdown("""
            <div class="warn-box">
            ⚠️ No hay datos en sesión. Ve a <b>Experimento Completo</b> y ejecútalo,
            o cambia la fuente a <b>Demo rápida</b>.
            </div>
            """, unsafe_allow_html=True)
            return
        historiales = st.session_state["historiales"]
        resultados  = st.session_state["resultados"]
        n_runs      = st.session_state.get("n_runs", 30)

    else:  # Demo rápida
        if not lanzar:
            st.info("Configura los parámetros y pulsa **⚡ Generar gráficas**.")
            return
        if not funcs_sel:
            st.warning("Selecciona al menos una función benchmark.")
            return

        historiales, resultados, n_runs = _run_demo(
            funcs_sel, dim, hms, max_iter, pop_size, demo_runs
        )

    # ── Tabs de visualización ─────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        " Vista global (4 funciones)",
        " Vista individual",
        " Ejecuciones individuales",
        " Mapa de calor de convergencia",
    ])

    with tab1:
        _tab_global(historiales, funcs_sel, n_runs, max_iter, pop_size)

    with tab2:
        _tab_individual(historiales, funcs_sel, n_runs, max_iter, pop_size)

    with tab3:
        _tab_runs(historiales, funcs_sel, max_iter, pop_size)

    with tab4:
        _tab_heatmap(historiales, resultados, funcs_sel, max_iter, pop_size)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Vista global: 4 subplots en una sola figura
# ──────────────────────────────────────────────────────────────────────────────
def _tab_global(historiales, funcs_sel, n_runs, max_iter, pop_size):
    st.markdown(f"**Curvas de convergencia mediana + banda IQR — {n_runs} ejecuciones**")

    escala  = st.radio("Escala Y", ["Logarítmica", "Lineal"],
                       horizontal=True, key="g_escala")
    banda   = st.checkbox("Mostrar banda IQR (p25–p75)", value=True, key="g_banda")
    algos_v = st.multiselect("Algoritmos", ALGOS, default=ALGOS,
                             format_func=lambda x: LABELS[x], key="g_algos")

    n = len(funcs_sel)
    cols_n = min(n, 2)
    rows_n = (n + 1) // 2

    fig, axes = plt.subplots(rows_n, cols_n,
                             figsize=(7.5 * cols_n, 4.5 * rows_n),
                             squeeze=False)
    axes_flat = axes.flatten()

    for i, fname in enumerate(funcs_sel):
        ax = axes_flat[i]
        for a in algos_v:
            arr = np.array(historiales[fname][a])
            med = np.median(arr, axis=0)
            p25 = np.percentile(arr, 25, axis=0)
            p75 = np.percentile(arr, 75, axis=0)
            xs  = _xs(a, len(med), pop_size)

            if escala == "Logarítmica":
                ax.semilogy(xs, np.maximum(med, 1e-14),
                            color=COLORS[a], lw=LW[a], ls=LS[a], label=LABELS[a])
                if banda:
                    ax.fill_between(xs, np.maximum(p25, 1e-14),
                                    np.maximum(p75, 1e-14),
                                    alpha=0.12, color=COLORS[a])
            else:
                ax.plot(xs, med, color=COLORS[a], lw=LW[a], ls=LS[a], label=LABELS[a])
                if banda:
                    ax.fill_between(xs, p25, p75, alpha=0.12, color=COLORS[a])

        ax.set_title(fname, fontsize=12, fontweight="bold")
        ax.set_xlabel("Evaluaciones", fontsize=9)
        ax.set_ylabel("Mejor fitness", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.22, ls="--")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    tipo_banda = " + banda IQR" if banda else ""
    fig.suptitle(f"Convergencia mediana{tipo_banda} — {n_runs} ejecuciones",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Botón de descarga
    _download_fig(fig, "convergencia_global.png")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Vista individual: una función, más detalles
# ──────────────────────────────────────────────────────────────────────────────
def _tab_individual(historiales, funcs_sel, n_runs, max_iter, pop_size):
    fname = st.selectbox("Función benchmark", funcs_sel, key="ind_func")
    c1, c2 = st.columns(2)
    with c1:
        escala = st.radio("Escala Y", ["Logarítmica", "Lineal"],
                          horizontal=True, key="ind_escala")
    with c2:
        mostrar = st.multiselect(
            "Elementos adicionales",
            ["Banda IQR", "Marcadores de cambio HMCR/PAR", "Línea óptimo (f=0)"],
            default=["Banda IQR"],
            key="ind_extra",
        )

    fig, ax = plt.subplots(figsize=(11, 5))

    for a in ALGOS:
        arr = np.array(historiales[fname][a])
        med = np.median(arr, axis=0)
        p25 = np.percentile(arr, 25, axis=0)
        p75 = np.percentile(arr, 75, axis=0)
        xs  = _xs(a, len(med), pop_size)

        plot_fn = ax.semilogy if escala == "Logarítmica" else ax.plot
        clip = (lambda v: np.maximum(v, 1e-14)) if escala == "Logarítmica" else (lambda v: v)

        plot_fn(xs, clip(med), color=COLORS[a], lw=LW[a], ls=LS[a], label=LABELS[a])

        if "Banda IQR" in mostrar:
            ax.fill_between(xs, clip(p25), clip(p75), alpha=0.13, color=COLORS[a])

    if "Línea óptimo (f=0)" in mostrar:
        ax.axhline(1e-10 if escala == "Logarítmica" else 0,
                   color="gray", ls="-.", lw=1, alpha=0.6, label="Óptimo (f≈0)")

    if "Marcadores de cambio HMCR/PAR" in mostrar:
        # Marcar cada 100 evaluaciones (ventana adaptativa)
        window = 100
        for x_mark in range(window, max_iter, window):
            ax.axvline(x_mark, color="#aaaaaa", lw=0.6, alpha=0.4)
        ax.text(0.02, 0.96, "│ = ventana adaptativa (100 iter.)",
                transform=ax.transAxes, fontsize=8, color="#888888",
                va="top")

    ax.set_title(f"Convergencia mediana con IQR — {fname} ({n_runs} ejecuciones)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Evaluaciones de la función objetivo", fontsize=11)
    ax.set_ylabel("Mejor fitness", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.22, ls="--")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — Ejecuciones individuales de HS: mejor / mediana / peor
# ──────────────────────────────────────────────────────────────────────────────
def _tab_runs(historiales, funcs_sel, max_iter, pop_size):
    st.markdown("""
    <div class="info-box">
    Muestra las curvas de convergencia de <b>ejecuciones individuales</b> de
    HS Adaptativo: la mejor, la mediana y la peor corrida (según fitness final),
    útil para analizar la variabilidad del algoritmo.
    </div>
    """, unsafe_allow_html=True)

    fname = st.selectbox("Función benchmark", funcs_sel, key="runs_func")
    algo  = st.selectbox("Algoritmo", ALGOS, format_func=lambda x: LABELS[x],
                         key="runs_algo")
    escala = st.radio("Escala Y", ["Logarítmica", "Lineal"],
                      horizontal=True, key="runs_escala")

    arr    = np.array(historiales[fname][algo])   # shape (n_runs, iter+1)
    finals = arr[:, -1]
    orden  = np.argsort(finals)

    idx_best   = orden[0]
    idx_median = orden[len(orden) // 2]
    idx_worst  = orden[-1]

    fig, ax = plt.subplots(figsize=(11, 5))
    xs_all = _xs(algo, arr.shape[1], pop_size)

    # Fondo: todas las ejecuciones en gris tenue
    for i in range(arr.shape[0]):
        ys = np.maximum(arr[i], 1e-14) if escala == "Logarítmica" else arr[i]
        if escala == "Logarítmica":
            ax.semilogy(xs_all, ys, color="#cccccc", lw=0.6, alpha=0.4)
        else:
            ax.plot(xs_all, ys, color="#cccccc", lw=0.6, alpha=0.4)

    # Destacadas
    clip = (lambda v: np.maximum(v, 1e-14)) if escala == "Logarítmica" else (lambda v: v)
    plot_fn = ax.semilogy if escala == "Logarítmica" else ax.plot

    plot_fn(xs_all, clip(arr[idx_best]),   color="#1D9E75", lw=2.5,
            label=f"Mejor  (f={finals[idx_best]:.3e})")
    plot_fn(xs_all, clip(arr[idx_median]), color="#534AB7", lw=2.5, ls="--",
            label=f"Mediana (f={finals[idx_median]:.3e})")
    plot_fn(xs_all, clip(arr[idx_worst]),  color="#D85A30", lw=2.5, ls=":",
            label=f"Peor   (f={finals[idx_worst]:.3e})")

    # Leyenda de fondo
    ax.plot([], [], color="#cccccc", lw=1, label="Resto de ejecuciones")

    ax.set_title(f"Ejecuciones individuales — {LABELS[algo]} sobre {fname}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Evaluaciones", fontsize=11)
    ax.set_ylabel("Mejor fitness", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.22, ls="--")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Estadística rápida
    st.markdown("**Estadística de fitness final en esta función**")
    import pandas as pd
    df = pd.DataFrame([{
        "Algoritmo": LABELS[algo],
        "Mejor":   f"{finals.min():.4e}",
        "Mediana": f"{np.median(finals):.4e}",
        "Peor":    f"{finals.max():.4e}",
        "Media":   f"{finals.mean():.4e}",
        "Std Dev": f"{finals.std():.4e}",
    }])
    st.dataframe(df, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — Mapa de calor: tasa de mejora por iteración
# ──────────────────────────────────────────────────────────────────────────────
def _tab_heatmap(historiales, resultados, funcs_sel, max_iter, pop_size):
    st.markdown("""
    <div class="info-box">
    El mapa de calor muestra la <b>tasa de mejora normalizada</b> del mejor fitness
    a lo largo de las iteraciones para cada algoritmo. Colores más oscuros indican
    mayor descenso relativo del fitness (convergencia más intensa).
    </div>
    """, unsafe_allow_html=True)

    fname  = st.selectbox("Función benchmark", funcs_sel, key="hm_func")
    n_bins = st.slider("Resolución (número de ventanas)", 20, 100, 50, 5, key="hm_bins")

    fig, axes = plt.subplots(1, len(ALGOS), figsize=(5 * len(ALGOS), 4),
                             sharey=True)

    for ax, a in zip(axes, ALGOS):
        arr = np.array(historiales[fname][a])   # (n_runs, steps)
        n_runs_local, steps = arr.shape

        # Remuestrear a n_bins columnas
        bin_edges = np.linspace(0, steps, n_bins + 1, dtype=int)
        matrix = np.zeros((n_runs_local, n_bins))
        for b in range(n_bins):
            s, e = bin_edges[b], bin_edges[b + 1]
            if e > s:
                # mejora relativa en la ventana
                window_vals = arr[:, s:e]
                col_min = window_vals.min(axis=1)
                col_max = np.maximum(arr[:, s], 1e-14)
                matrix[:, b] = np.log10(np.maximum(col_min, 1e-14) / col_max)

        # Ordenar filas por fitness final
        orden = np.argsort(arr[:, -1])
        matrix = matrix[orden]

        im = ax.imshow(matrix, aspect="auto", origin="upper",
                       cmap="RdPu_r", interpolation="nearest",
                       vmin=matrix.min(), vmax=0)
        ax.set_title(LABELS[a], fontsize=10, fontweight="bold", color=COLORS[a])
        ax.set_xlabel("Ventana de iteraciones", fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel("Ejecución (ordenada por fitness final)", fontsize=9)
        ax.set_xticks([0, n_bins // 2, n_bins - 1])
        xticklabels = [
            "0",
            str(_xs(a, steps, pop_size)[steps // 2]),
            str(_xs(a, steps, pop_size)[-1]),
        ]
        ax.set_xticklabels(xticklabels, fontsize=8)

    fig.colorbar(im, ax=axes[-1], label="log₁₀(mejora relativa)", shrink=0.8)
    fig.suptitle(f"Mapa de calor de convergencia — {fname}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Comparativa rápida de velocidad
    st.markdown("**Velocidad de convergencia — iteración en que cada algoritmo alcanza el 90% de su mejora total**")
    import pandas as pd
    rows = []
    for a in ALGOS:
        arr   = np.array(historiales[fname][a])
        med   = np.median(arr, axis=0)
        total = med[0] - med[-1]
        if total <= 0:
            iter_90 = 0
        else:
            threshold = med[0] - 0.9 * total
            reach = np.where(med <= threshold)[0]
            iter_90 = int(_xs(a, len(med), pop_size)[reach[0]]) if len(reach) else int(_xs(a, len(med), pop_size)[-1])
        rows.append({
            "Algoritmo": LABELS[a],
            "Fitness inicial (mediana)": f"{med[0]:.4e}",
            "Fitness final (mediana)":   f"{med[-1]:.4e}",
            "Eval. al 90% de mejora":    iter_90,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _xs(algo, n_steps, pop_size):
    """Convierte pasos de historia a evaluaciones totales."""
    if algo == "HS_adapt":
        return np.arange(1, n_steps + 1)
    else:
        return np.arange(1, n_steps + 1) * pop_size


def _download_fig(fig, filename):
    """Botón de descarga PNG para una figura matplotlib."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        "⬇ Descargar gráfica (PNG)",
        buf,
        file_name=filename,
        mime="image/png",
    )


def _run_demo(funcs_sel, dim, hms, max_iter, pop_size, n_runs):
    """Ejecuta n_runs corridas rápidas para la demo y devuelve historiales."""
    gens = max(1, max_iter // pop_size)
    historiales = {f: {a: [] for a in ALGOS} for f in funcs_sel}
    resultados  = {f: {a: [] for a in ALGOS} for f in funcs_sel}

    total = len(funcs_sel) * n_runs * 3
    prog  = st.progress(0, text="Calculando demo...")
    paso  = 0

    for fname in funcs_sel:
        func, lb, ub = PROBLEMS[fname]
        for run in range(n_runs):
            seed = run * 100 + 42

            f, h = harmony_search_adaptive(func, lb, ub, dim=dim, HMS=hms,
                                           max_iter=max_iter, seed=seed)
            historiales[fname]["HS_adapt"].append(h)
            resultados[fname]["HS_adapt"].append(f)
            paso += 1; prog.progress(paso / total, text=f"[{fname}] HS run {run+1}")

            f, h = genetic_algorithm(func, lb, ub, dim=dim, pop_size=pop_size,
                                     gens=gens, seed=seed)
            historiales[fname]["GA"].append(h)
            resultados[fname]["GA"].append(f)
            paso += 1; prog.progress(paso / total, text=f"[{fname}] GA run {run+1}")

            f, h = de_best_1_bin(func, lb, ub, dim=dim, N=pop_size, Gmax=gens,
                                 F=0.6, Cr=0.9, seed=seed)
            historiales[fname]["DE_best"].append(h)
            resultados[fname]["DE_best"].append(f)
            paso += 1; prog.progress(paso / total, text=f"[{fname}] DE run {run+1}")

    prog.empty()
    st.success(f"✅ Demo completada: {n_runs} corridas × {len(funcs_sel)} funciones × 3 algoritmos")
    return historiales, resultados, n_runs
