"""
Página 1: Ejecución Rápida
Permite al usuario configurar parámetros y ejecutar una sola corrida
de los tres algoritmos para comparar en tiempo real.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from harmony_search import (
    harmony_search_adaptive,
    genetic_algorithm,
    de_best_1_bin,
    PROBLEMS,
)

# Paleta de colores del proyecto
COLOR_HS = "#534AB7"
COLOR_GA = "#D85A30"
COLOR_DE = "#1D9E75"


def render():
    st.header(" Ejecución Rápida")
    st.markdown("""
    <div class="info-box">
    Configura los parámetros y ejecuta <b>una sola corrida</b> de los tres algoritmos.
    Observa la convergencia en tiempo real y compara los resultados finales.
    </div>
    """, unsafe_allow_html=True)

    # ── Panel de configuración ────────────────────────────────────────────────
    with st.expander(" Configuración de parámetros", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Problema**")
            funcion = st.selectbox(
                "Función benchmark",
                list(PROBLEMS.keys()),
                index=0,
                help="Función de optimización sobre la que se ejecutarán los algoritmos.",
            )
            dim = st.slider("Dimensiones (n)", min_value=2, max_value=20, value=10, step=1)

        with col2:
            st.markdown("**Búsqueda Armónica**")
            hms = st.slider("HMS — Tamaño de memoria", 10, 100, 50, 10,
                            help="Número de soluciones en la Memoria Armónica.")
            max_iter = st.slider("Iteraciones máximas", 500, 5000, 2000, 500)

        with col3:
            st.markdown("**General**")
            seed = st.number_input("Semilla aleatoria", min_value=0, max_value=9999,
                                   value=42, step=1)
            pop_size = st.slider("Tamaño de población (AG/DE)", 20, 200, 100, 10)

    # ── Botón de ejecución ────────────────────────────────────────────────────
    if st.button("▶ Ejecutar experimento", type="primary", use_container_width=True):

        func, lb, ub = PROBLEMS[funcion]
        gens = max(1, max_iter // pop_size)

        prog = st.progress(0, text="Ejecutando Búsqueda Armónica Adaptativa...")

        # --- HS ---
        f_hs, h_hs, hmcr_hist, par_hist = _run_hs(
            func, lb, ub, dim, hms, max_iter, int(seed)
        )
        prog.progress(33, text="Ejecutando Algoritmo Genético...")

        # --- GA ---
        f_ga, h_ga = genetic_algorithm(
            func, lb, ub, dim=dim, pop_size=pop_size, gens=gens, seed=int(seed)
        )
        prog.progress(66, text="Ejecutando DE/best/1/bin...")

        # --- DE ---
        f_de, h_de = de_best_1_bin(
            func, lb, ub, dim=dim, N=pop_size, Gmax=gens, F=0.6, Cr=0.9, seed=int(seed)
        )
        prog.progress(100, text="¡Listo!")
        prog.empty()

        # ── Métricas ──────────────────────────────────────────────────────────
        st.markdown("### Resultados")
        c1, c2, c3, c4 = st.columns(4)
        mejor = min(f_hs, f_ga, f_de)

        with c1:
            _metrica("HS Adaptativo", f_hs, mejor == f_hs, "HMCR/PAR adaptativos")
        with c2:
            _metrica("Alg. Genético", f_ga, mejor == f_ga, f"SBX + PM | {gens} gen.")
        with c3:
            _metrica("DE/best/1/bin", f_de, mejor == f_de, f"F=0.6, Cr=0.9 | {gens} gen.")
        with c4:
            _metrica_params("Parámetros HS\nfinal", hmcr_hist[-1], par_hist[-1])

        # ── Gráfica convergencia ──────────────────────────────────────────────
        st.markdown("### Convergencia (curva individual, escala log)")
        fig, ax = plt.subplots(figsize=(10, 4))

        # Normalizar ejes a evaluaciones totales
        eval_hs = np.arange(1, len(h_hs) + 1)
        eval_ga = np.arange(1, len(h_ga) + 1) * pop_size
        eval_de = np.arange(1, len(h_de) + 1) * pop_size

        ax.semilogy(eval_hs, np.maximum(h_hs, 1e-12), color=COLOR_HS,
                    lw=2, label="HS Adaptativo")
        ax.semilogy(eval_ga, np.maximum(h_ga, 1e-12), color=COLOR_GA,
                    lw=2, ls="--", label="Alg. Genético")
        ax.semilogy(eval_de, np.maximum(h_de, 1e-12), color=COLOR_DE,
                    lw=2, ls=":", label="DE/best/1/bin")

        ax.set_xlabel("Evaluaciones de la función objetivo", fontsize=11)
        ax.set_ylabel("Mejor fitness", fontsize=11)
        ax.set_title(f"Convergencia — {funcion} (n={dim}, semilla={seed})", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, which="both", alpha=0.3, ls="--")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ── Evolución de parámetros adaptativos ───────────────────────────────
        st.markdown("### Adaptación de HMCR y PAR a lo largo del tiempo")
        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharex=True)

        ventanas = np.arange(1, len(hmcr_hist) + 1) * 100
        ax1.step(ventanas, hmcr_hist, where="post", color=COLOR_HS, lw=2)
        ax1.set_ylabel("HMCR", fontsize=10)
        ax1.set_yticks([0.6, 0.8, 0.9])
        ax1.grid(True, alpha=0.3, ls="--")
        ax1.set_title("Evolución de los parámetros adaptativos", fontsize=11)

        ax2.step(ventanas, par_hist, where="post", color=COLOR_DE, lw=2)
        ax2.set_ylabel("PAR", fontsize=10)
        ax2.set_yticks([0.01, 0.1, 0.2])
        ax2.set_xlabel("Iteración", fontsize=10)
        ax2.grid(True, alpha=0.3, ls="--")

        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # ── Tabla resumen ─────────────────────────────────────────────────────
        st.markdown("### Tabla de resultados")
        import pandas as pd
        df = pd.DataFrame({
            "Algoritmo": ["HS Adaptativo", "Alg. Genético", "DE/best/1/bin"],
            "Fitness final": [f_hs, f_ga, f_de],
            "Evaluaciones totales": [max_iter, gens * pop_size, gens * pop_size],
            "HMCR final": [hmcr_hist[-1], "—", "—"],
            "PAR final": [par_hist[-1], "—", "—"],
        })
        df["Fitness final"] = df["Fitness final"].apply(lambda v: f"{v:.6e}")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.success(f" Experimento completado. Mejor algoritmo en {funcion}: "
                   f"{'HS Adaptativo' if mejor == f_hs else ('Alg. Genético' if mejor == f_ga else 'DE/best/1/bin')}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_hs(func, lb, ub, dim, hms, max_iter, seed):
    """Wrapper de HS que además extrae el historial de HMCR/PAR."""
    import numpy as np
    rng = np.random.RandomState(seed)
    lb_arr = np.full(dim, lb)
    ub_arr = np.full(dim, ub)

    hmcr_values = [0.6, 0.8, 0.9]
    par_values  = [0.01, 0.1, 0.2]
    hmcr_idx, par_idx = 1, 1
    HMCR = hmcr_values[hmcr_idx]
    PAR  = par_values[par_idx]
    BW   = 0.05 * (ub - lb)

    HM     = lb_arr + (ub_arr - lb_arr) * rng.rand(hms, dim)
    HM_fit = np.array([func(h) for h in HM])
    worst_idx = int(np.argmax(HM_fit))

    history      = [float(HM_fit.min())]
    success_buf  = []
    hmcr_hist    = []
    par_hist     = []
    window       = 100

    for t in range(max_iter):
        new_h = np.zeros(dim)
        for j in range(dim):
            if rng.rand() < HMCR:
                new_h[j] = HM[rng.randint(0, hms), j]
                if rng.rand() < PAR:
                    new_h[j] = np.clip(new_h[j] + BW * rng.uniform(-1, 1),
                                       lb_arr[j], ub_arr[j])
            else:
                new_h[j] = rng.uniform(lb_arr[j], ub_arr[j])

        nf = func(new_h)
        ok = False
        if nf < HM_fit[worst_idx]:
            HM[worst_idx]     = new_h
            HM_fit[worst_idx] = nf
            worst_idx         = int(np.argmax(HM_fit))
            ok = True

        history.append(float(HM_fit.min()))
        success_buf.append(1 if ok else 0)
        if len(success_buf) > window:
            success_buf.pop(0)

        if (t + 1) % window == 0 and len(success_buf) == window:
            ps = sum(success_buf) / window
            if ps > 0.3:
                hmcr_idx = min(hmcr_idx + 1, 2)
                par_idx  = max(par_idx  - 1, 0)
            elif ps < 0.15:
                hmcr_idx = max(hmcr_idx - 1, 0)
                par_idx  = min(par_idx  + 1, 2)
            HMCR = hmcr_values[hmcr_idx]
            PAR  = par_values[par_idx]
            hmcr_hist.append(HMCR)
            par_hist.append(PAR)

    if not hmcr_hist:
        hmcr_hist = [HMCR]
        par_hist  = [PAR]

    return float(HM_fit.min()), history, hmcr_hist, par_hist


def _metrica(titulo, valor, es_mejor, subtitulo):
    badge = '<span class="badge-best">⭐ Mejor</span>' if es_mejor else ""
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{titulo}</div>
      <div class="metric-value">{valor:.4e}</div>
      <div class="metric-sub">{subtitulo} {badge}</div>
    </div>
    """, unsafe_allow_html=True)


def _metrica_params(titulo, hmcr, par):
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Estado final HS</div>
      <div class="metric-value">HMCR {hmcr}</div>
      <div class="metric-sub">PAR = {par}</div>
    </div>
    """, unsafe_allow_html=True)
