"""
Página 3: Análisis Comparativo
Carga los resultados de la sesión (generados en p2) y permite
explorar comparaciones individualmente por función.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from harmony_search import PROBLEMS

COLOR_HS = "#534AB7"
COLOR_GA = "#D85A30"
COLOR_DE = "#1D9E75"
ALGOS    = ["HS_adapt", "GA", "DE_best"]
LABELS   = {"HS_adapt": "HS Adaptativo", "GA": "Alg. Genético", "DE_best": "DE/best/1/bin"}


def render():
    st.header("Análisis Comparativo")

    if "resultados" not in st.session_state:
        st.markdown("""
        <div class="warn-box">
         No hay resultados cargados. Ve primero a <b>Experimento Completo</b> y ejecuta
        al menos una corrida para poder analizar los resultados aquí.
        </div>
        """, unsafe_allow_html=True)
        return

    resultados  = st.session_state["resultados"]
    historiales = st.session_state["historiales"]
    n_runs      = st.session_state.get("n_runs", 30)
    pop_size    = st.session_state.get("pop_size", 100)
    funciones   = list(resultados.keys())

    # ── Selector de función ───────────────────────────────────────────────────
    funcion = st.selectbox("Seleccionar función benchmark", funciones)

    tab1, tab2, tab3 = st.tabs(["Convergencia", "Distribución", "Estadísticas"])

    # ── TAB 1: Convergencia ───────────────────────────────────────────────────
    with tab1:
        st.markdown(f"**Convergencia mediana — {funcion}** ({n_runs} ejecuciones)")
        opciones = st.multiselect("Algoritmos a mostrar", ALGOS,
                                  default=ALGOS,
                                  format_func=lambda x: LABELS[x])
        escala = st.radio("Escala Y", ["Logarítmica", "Lineal"], horizontal=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        styles = [("-", COLOR_HS), ("--", COLOR_GA), (":", COLOR_DE)]
        for a, (ls, col) in zip(ALGOS, styles):
            if a not in opciones:
                continue
            arr = np.array(historiales[funcion][a])
            med = np.median(arr, axis=0)
            p25 = np.percentile(arr, 25, axis=0)
            p75 = np.percentile(arr, 75, axis=0)

            xs = np.arange(1, len(med) + 1) if a == "HS_adapt" else np.arange(1, len(med) + 1) * pop_size

            if escala == "Logarítmica":
                ax.semilogy(xs, np.maximum(med, 1e-14), color=col, lw=2, ls=ls, label=LABELS[a])
                ax.fill_between(xs, np.maximum(p25, 1e-14),
                                np.maximum(p75, 1e-14), alpha=0.15, color=col)
            else:
                ax.plot(xs, med, color=col, lw=2, ls=ls, label=LABELS[a])
                ax.fill_between(xs, p25, p75, alpha=0.15, color=col)

        ax.set_xlabel("Evaluaciones de la función objetivo", fontsize=11)
        ax.set_ylabel("Mejor fitness", fontsize=11)
        ax.set_title(f"Convergencia mediana con IQR — {funcion}", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, which="both", alpha=0.25, ls="--")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── TAB 2: Distribución ───────────────────────────────────────────────────
    with tab2:
        st.markdown(f"**Distribución de fitness final — {funcion}**")
        col_v = st.radio("Visualización", ["Boxplot", "Histograma", "Violín"], horizontal=True)

        data = {a: np.array(resultados[funcion][a]) for a in ALGOS}
        fig, ax = plt.subplots(figsize=(9, 4))

        if col_v == "Boxplot":
            bp = ax.boxplot(list(data.values()), patch_artist=True,
                            medianprops=dict(color="white", lw=2))
            for patch, color in zip(bp["boxes"], [COLOR_HS, COLOR_GA, COLOR_DE]):
                patch.set_facecolor(color); patch.set_alpha(0.8)
            ax.set_xticklabels([LABELS[a] for a in ALGOS], fontsize=10)
            ax.set_yscale("log")
            ax.set_ylabel("Fitness final (log)", fontsize=10)

        elif col_v == "Histograma":
            for a, color in zip(ALGOS, [COLOR_HS, COLOR_GA, COLOR_DE]):
                vals = np.log10(np.maximum(data[a], 1e-14))
                ax.hist(vals, bins=12, alpha=0.6, color=color, label=LABELS[a])
            ax.set_xlabel("log₁₀(Fitness final)", fontsize=10)
            ax.set_ylabel("Frecuencia", fontsize=10)
            ax.legend(fontsize=9)

        else:  # Violín
            parts = ax.violinplot(list(data.values()), showmedians=True)
            for pc, color in zip(parts["bodies"], [COLOR_HS, COLOR_GA, COLOR_DE]):
                pc.set_facecolor(color); pc.set_alpha(0.7)
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels([LABELS[a] for a in ALGOS], fontsize=10)
            ax.set_yscale("log")
            ax.set_ylabel("Fitness final (log)", fontsize=10)

        ax.set_title(f"Distribución — {funcion}", fontsize=11)
        ax.grid(True, which="both", alpha=0.2, ls="--")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── TAB 3: Estadísticas ───────────────────────────────────────────────────
    with tab3:
        st.markdown(f"**Estadísticas detalladas — {funcion}**")

        filas = []
        for a in ALGOS:
            vals = np.array(resultados[funcion][a])
            filas.append({
                "Algoritmo": LABELS[a],
                "Media":     f"{vals.mean():.4e}",
                "Mediana":   f"{np.median(vals):.4e}",
                "Std Dev":   f"{vals.std():.4e}",
                "Mín":       f"{vals.min():.4e}",
                "Máx":       f"{vals.max():.4e}",
                "Q1":        f"{np.percentile(vals, 25):.4e}",
                "Q3":        f"{np.percentile(vals, 75):.4e}",
            })
        st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)

        # Todos los pares de Wilcoxon
        st.markdown("**Prueba de Wilcoxon entre todos los pares**")
        pares = [("HS_adapt","GA"), ("HS_adapt","DE_best"), ("GA","DE_best")]
        filas_w = []
        for a1, a2 in pares:
            v1 = np.array(resultados[funcion][a1])
            v2 = np.array(resultados[funcion][a2])
            try:
                _, p_val = stats.wilcoxon(v1, v2)
            except ValueError:
                p_val = 1.0
            sig = p_val < 0.05
            mejor = LABELS[a1] if v1.mean() < v2.mean() else LABELS[a2]
            filas_w.append({
                "Comparación": f"{LABELS[a1]} vs {LABELS[a2]}",
                "p-valor": f"{p_val:.4f}",
                "Significativo": " Sí" if sig else " No",
                "Mejor (media)": mejor,
            })
        st.dataframe(pd.DataFrame(filas_w), use_container_width=True, hide_index=True)

        # Ranking global
        st.markdown("**Ranking por función (todos los resultados cargados)**")
        rank_rows = []
        for fn in funciones:
            medias = {a: np.array(resultados[fn][a]).mean() for a in ALGOS}
            orden = sorted(medias, key=lambda x: medias[x])
            rank_rows.append({
                "Función": fn,
                "1° lugar": LABELS[orden[0]],
                "Media 1°": f"{medias[orden[0]]:.4e}",
                "2° lugar": LABELS[orden[1]],
                "Media 2°": f"{medias[orden[1]]:.4e}",
                "3° lugar": LABELS[orden[2]],
                "Media 3°": f"{medias[orden[2]]:.4e}",
            })
        st.dataframe(pd.DataFrame(rank_rows), use_container_width=True, hide_index=True)
