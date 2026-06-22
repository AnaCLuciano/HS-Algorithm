"""
Página 2: Experimento Completo
Ejecuta N corridas independientes (por defecto 30) y calcula
media, desviación estándar y prueba de Wilcoxon.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from harmony_search import (
    harmony_search_adaptive,
    genetic_algorithm,
    de_best_1_bin,
    PROBLEMS,
)
import scipy.stats as stats

COLOR_HS = "#534AB7"
COLOR_GA = "#D85A30"
COLOR_DE = "#1D9E75"
ALGOS    = ["HS_adapt", "GA", "DE_best"]
LABELS   = {"HS_adapt": "HS Adaptativo", "GA": "Alg. Genético", "DE_best": "DE/best/1/bin"}


def render():
    st.header("Experimento Completo")
    st.markdown("""
    <div class="info-box">
    Ejecuta <b>N corridas independientes</b> (mínimo requerido por el proyecto: <b>30</b>) con semillas
    distintas para cada combinación de algoritmo × función benchmark.
    Genera tabla estadística completa y prueba de Wilcoxon (α = 0.05).
    </div>
    """, unsafe_allow_html=True)

    # ── Configuración ─────────────────────────────────────────────────────────
    with st.expander("Configuración del experimento", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            n_runs   = st.slider("Número de ejecuciones", 5, 30, 30, 5,
                                 help="El proyecto final requiere mínimo 30.")
            dim      = st.slider("Dimensiones (n)", 2, 20, 10)
        with col2:
            max_iter = st.slider("Iteraciones HS", 500, 5000, 5000, 500)
            hms      = st.slider("HMS", 10, 100, 50, 10)
        with col3:
            pop_size = st.slider("Población AG/DE", 20, 200, 100, 10)
            funciones_sel = st.multiselect(
                "Funciones a evaluar",
                list(PROBLEMS.keys()),
                default=list(PROBLEMS.keys()),
            )

    if not funciones_sel:
        st.warning("Selecciona al menos una función benchmark.")
        return

    if st.button("▶ Ejecutar experimento completo", type="primary", use_container_width=True):

        gens = max(1, max_iter // pop_size)
        resultados = {p: {a: [] for a in ALGOS} for p in funciones_sel}
        historiales = {p: {a: [] for a in ALGOS} for p in funciones_sel}

        total_steps = len(funciones_sel) * n_runs * 3
        paso = 0
        prog = st.progress(0, text="Iniciando experimento...")

        for p_name in funciones_sel:
            func, lb, ub = PROBLEMS[p_name]
            for run in range(n_runs):
                seed = run * 100 + 42

                # HS
                f, h = harmony_search_adaptive(
                    func, lb, ub, dim=dim, HMS=hms, max_iter=max_iter, seed=seed
                )
                resultados[p_name]["HS_adapt"].append(f)
                historiales[p_name]["HS_adapt"].append(h)
                paso += 1
                prog.progress(paso / total_steps,
                              text=f"[{p_name}] Run {run+1}/{n_runs} — HS ✓")

                # GA
                f, h = genetic_algorithm(
                    func, lb, ub, dim=dim, pop_size=pop_size, gens=gens, seed=seed
                )
                resultados[p_name]["GA"].append(f)
                historiales[p_name]["GA"].append(h)
                paso += 1
                prog.progress(paso / total_steps,
                              text=f"[{p_name}] Run {run+1}/{n_runs} — GA ✓")

                # DE
                f, h = de_best_1_bin(
                    func, lb, ub, dim=dim, N=pop_size, Gmax=gens, F=0.6, Cr=0.9, seed=seed
                )
                resultados[p_name]["DE_best"].append(f)
                historiales[p_name]["DE_best"].append(h)
                paso += 1
                prog.progress(paso / total_steps,
                              text=f"[{p_name}] Run {run+1}/{n_runs} — DE ")

        prog.empty()
        st.success(f" Experimento completado: {n_runs} ejecuciones × {len(funciones_sel)} funciones × 3 algoritmos")

        # Guardar en sesión para la página de análisis
        st.session_state["resultados"]  = resultados
        st.session_state["historiales"] = historiales
        st.session_state["n_runs"]      = n_runs
        st.session_state["max_iter"]    = max_iter
        st.session_state["pop_size"]    = pop_size

        # ── Tabla estadística ─────────────────────────────────────────────────
        st.markdown("### Tabla comparativa (media ± desviación estándar)")
        filas = []
        for p in funciones_sel:
            for a in ALGOS:
                vals = np.array(resultados[p][a])
                filas.append({
                    "Función": p,
                    "Algoritmo": LABELS[a],
                    "Media": f"{vals.mean():.4e}",
                    "Std Dev": f"{vals.std():.4e}",
                    "Mín": f"{vals.min():.4e}",
                    "Máx": f"{vals.max():.4e}",
                })
        df = pd.DataFrame(filas)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Prueba de Wilcoxon ────────────────────────────────────────────────
        st.markdown("### Prueba de Wilcoxon (α = 0.05) — HS_adapt vs mejor referencia")
        filas_w = []
        for p in funciones_sel:
            r_hs = np.array(resultados[p]["HS_adapt"])
            r_ga = np.array(resultados[p]["GA"])
            r_de = np.array(resultados[p]["DE_best"])

            if r_ga.mean() < r_de.mean():
                ref_name, ref_vals = "GA", r_ga
            else:
                ref_name, ref_vals = "DE_best", r_de

            try:
                _, p_val = stats.wilcoxon(r_hs, ref_vals)
            except ValueError:
                p_val = 1.0

            sig = p_val < 0.05
            filas_w.append({
                "Función": p,
                "HS vs": LABELS[ref_name],
                "p-valor": f"{p_val:.4f}",
                "Resultado": "✅ Diferencia significativa" if sig else "⚠️ Sin diferencia significativa",
            })
        df_w = pd.DataFrame(filas_w)
        st.dataframe(df_w, use_container_width=True, hide_index=True)

        # ── Convergencia mediana ──────────────────────────────────────────────
        st.markdown("### Curvas de convergencia (mediana de ejecuciones, escala log)")
        n_funcs = len(funciones_sel)
        cols_fig = min(n_funcs, 2)
        rows_fig = (n_funcs + 1) // 2

        fig, axes = plt.subplots(rows_fig, cols_fig,
                                 figsize=(7 * cols_fig, 4 * rows_fig),
                                 squeeze=False)
        axes_flat = axes.flatten()

        for i, p_name in enumerate(funciones_sel):
            ax = axes_flat[i]
            for a, color, ls in [("HS_adapt", COLOR_HS, "-"),
                                   ("GA",       COLOR_GA, "--"),
                                   ("DE_best",  COLOR_DE, ":")]:
                arr = np.array(historiales[p_name][a])
                med = np.median(arr, axis=0)
                if a == "HS_adapt":
                    xs = np.arange(1, len(med) + 1)
                else:
                    xs = np.arange(1, len(med) + 1) * pop_size

                ax.semilogy(xs, np.maximum(med, 1e-12),
                            color=color, lw=2, ls=ls, label=LABELS[a])

            ax.set_title(p_name, fontsize=11)
            ax.set_xlabel("Evaluaciones", fontsize=9)
            ax.set_ylabel("Mejor fitness", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, which="both", alpha=0.25, ls="--")

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(f"Convergencia mediana — {n_runs} ejecuciones",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ── Boxplots ──────────────────────────────────────────────────────────
        st.markdown("### Distribución de fitness final (boxplot)")
        fig2, axes2 = plt.subplots(1, n_funcs,
                                   figsize=(4.5 * n_funcs, 5),
                                   squeeze=False)
        colors_bp = [COLOR_HS, COLOR_GA, COLOR_DE]
        for i, p_name in enumerate(funciones_sel):
            ax = axes2[0][i]
            data = [np.array(resultados[p_name][a]) for a in ALGOS]
            bp = ax.boxplot(data, patch_artist=True, notch=False,
                            medianprops=dict(color="white", lw=2))
            for patch, color in zip(bp["boxes"], colors_bp):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
            ax.set_yscale("log")
            ax.set_title(p_name, fontsize=11)
            ax.set_xticklabels(["HS", "GA", "DE"], fontsize=10)
            ax.set_ylabel("Fitness final", fontsize=9)
            ax.grid(True, which="both", alpha=0.25, ls="--")

        fig2.suptitle("Distribución de resultados finales", fontsize=13, fontweight="bold")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # Exportar CSV
        st.markdown("### Exportar resultados")
        filas_raw = []
        for p in funciones_sel:
            for a in ALGOS:
                for run_i, val in enumerate(resultados[p][a]):
                    filas_raw.append({"Función": p, "Algoritmo": LABELS[a],
                                      "Run": run_i + 1, "Fitness": val})
        df_raw = pd.DataFrame(filas_raw)
        csv = df_raw.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Descargar resultados CSV", csv,
                           "resultados_hs_lab.csv", "text/csv",
                           use_container_width=True)
