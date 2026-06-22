"""
Página 4: Descripción del Algoritmo
Muestra el pseudocódigo, el mecanismo adaptativo y permite
simular la evolución de HMCR/PAR con controles interactivos.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def render():
    st.header("🧠 Algoritmo — Búsqueda Armónica Adaptativa")

    tab1, tab2, tab3 = st.tabs(["Pseudocódigo", "Mecanismo adaptativo", "Simulador"])

    # ── TAB 1: Pseudocódigo ───────────────────────────────────────────────────
    with tab1:
        st.markdown("""
<div class="info-box">
El algoritmo de <b>Búsqueda Armónica (Harmony Search)</b> fue propuesto por Geem et al. (2001).
Esta implementación incorpora <b>adaptación dinámica</b> de HMCR y PAR basada en tasa de
éxitos, cumpliendo el requerimiento de la práctica P3.
</div>
""", unsafe_allow_html=True)

        st.markdown("#### Pseudocódigo")
        st.code("""
Algorithm: Adaptive Harmony Search (AHS)
Requiere: func, lb, ub, HMS, max_iter, dim

╔══════════════════════════════════════════════════════════════╗
║  INICIALIZACIÓN                                              ║
╠══════════════════════════════════════════════════════════════╣
║  HM ← GenerarPoblacionAleatoria(HMS, dim, lb, ub)            ║
║  HM_fit ← [func(HM[i]) para i en 0..HMS-1]                   ║
║  HMCR ← 0.8    // ∈ {0.6, 0.8, 0.9}                          ║
║  PAR  ← 0.1    // ∈ {0.01, 0.1, 0.2}                         ║
║  BW   ← 0.05 × (ub - lb)                                     ║
║  buffer ← []   // ventana deslizante de 100 éxitos           ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║  BUCLE PRINCIPAL  (t = 1 .. max_iter)                        ║
╠══════════════════════════════════════════════════════════════╣
║  IMPROVISACIÓN:                                              ║
║    para j = 1 .. dim:                                        ║
║      si rand() < HMCR:                                       ║
║        x_new[j] ← HM[random_idx, j]   // memoria armónica    ║
║        si rand() < PAR:                                      ║
║          x_new[j] += BW × U(-1, 1)    // ajuste de tono      ║
║          x_new[j]  = clip(x_new[j], lb, ub)                  ║
║      sino:                                                   ║
║        x_new[j] ← U(lb, ub)           // aleatorio           ║
║                                                              ║
║  ACTUALIZACIÓN:                                              ║
║    si func(x_new) < max(HM_fit):                             ║
║      HM[worst] ← x_new                                       ║
║      éxito ← True                                            ║
║                                                              ║
║  ADAPTACIÓN (cada 100 iteraciones):                          ║
║    ps ← sum(buffer) / 100                                    ║
║    si ps > 0.30: HMCR↑  PAR↓   // explotar                   ║
║    si ps < 0.15: HMCR↓  PAR↑   // explorar                   ║
╚══════════════════════════════════════════════════════════════╝

Retorna: min(HM_fit)
""", language="text")

        st.markdown("#### Valores discretos de parámetros")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            **HMCR** (Harmony Memory Considering Rate)
            | Índice | Valor | Modo |
            |--------|-------|------|
            | 0 | 0.6 | Exploración |
            | 1 | 0.8 | Balance |
            | 2 | 0.9 | Explotación |
            """)
        with c2:
            st.markdown("""
            **PAR** (Pitch Adjusting Rate)
            | Índice | Valor | Modo |
            |--------|-------|------|
            | 0 | 0.01 | Explotación |
            | 1 | 0.10 | Balance |
            | 2 | 0.20 | Exploración |
            """)
        with c3:
            st.markdown("""
            **Regla de adaptación**
            | ps (tasa éxito) | Acción |
            |-----------------|--------|
            | ps > 0.30 | HMCR↑, PAR↓ (explotar) |
            | ps < 0.15 | HMCR↓, PAR↑ (explorar) |
            | 0.15 ≤ ps ≤ 0.30 | Sin cambio |
            """)

        st.markdown("#### Parámetros de los algoritmos de referencia")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            **Algoritmo Genético (AG)**
            - Selección: Torneo binario (k=2)
            - Cruza: SBX, η=15, Pc=0.9
            - Mutación: Polinomial, η=20, Pm=0.3
            - Elitismo: 2 mejores individuos
            - Generaciones: evaluaciones_totales / pop_size
            """)
        with col_b:
            st.markdown("""
            **Evolución Diferencial DE/best/1/bin**
            - Vector base: mejor individuo
            - Factor de escala: F = 0.6
            - Tasa de cruza: Cr = 0.9
            - Cruza: binaria (j_rand garantiza ≥1 dim)
            - Selección: greedy (reemplaza si mejora)
            """)

    # ── TAB 2: Mecanismo adaptativo ───────────────────────────────────────────
    with tab2:
        st.markdown("""
<div class="info-box">
El mecanismo de adaptación responde a la <b>tasa de éxito (ps)</b> calculada sobre
una ventana deslizante de 100 iteraciones. Cuando el algoritmo tiene muchos éxitos
(alta ps), se explota intensificando la memoria; cuando tiene pocos (baja ps),
se aumenta la exploración mediante ajuste de tono.
</div>
""", unsafe_allow_html=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Diagrama de flujo simplificado como scatter/anotación
        ax = axes[0]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_title("Lógica de adaptación de HMCR y PAR", fontsize=11)

        boxes = [
            (5, 9.0, "Calcular ps = éxitos/100", "#534AB7", "white"),
            (2, 7.0, "ps > 0.30\n(éxito alto)", "#1D9E75", "white"),
            (8, 7.0, "ps < 0.15\n(éxito bajo)", "#D85A30", "white"),
            (5, 5.2, "Sin cambio", "#888780", "white"),
            (2, 5.2, "HMCR ↑\nPAR ↓\n(Explotar)", "#1D9E75", "white"),
            (8, 5.2, "HMCR ↓\nPAR ↑\n(Explorar)", "#D85A30", "white"),
        ]
        for (x, y, txt, fc, tc) in boxes:
            ax.text(x, y, txt, ha="center", va="center", fontsize=9,
                    color=tc, fontweight="bold",
                    bbox=dict(facecolor=fc, edgecolor="none",
                              boxstyle="round,pad=0.5", alpha=0.85))

        # flechas
        for arrow in [(5, 8.6, 2, 7.4), (5, 8.6, 8, 7.4), (5, 8.6, 5, 5.6),
                      (2, 6.6, 2, 5.6), (8, 6.6, 8, 5.6)]:
            ax.annotate("", xy=(arrow[2], arrow[3]), xytext=(arrow[0], arrow[1]),
                        arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))

        # Gráfica de ejemplo de adaptación
        ax2 = axes[1]
        np.random.seed(99)
        iters = np.arange(1, 51) * 100
        hmcr_sim = []
        par_sim  = []
        hi, pi = 1, 1
        hv = [0.6, 0.8, 0.9]; pv = [0.01, 0.1, 0.2]
        buf = []
        rng = np.random.default_rng(42)
        for i in range(50):
            ps = max(0, min(1, 0.2 + 0.25 * np.sin(i * 0.4) + 0.1 * rng.standard_normal()))
            if ps > 0.3: hi = min(hi+1,2); pi = max(pi-1,0)
            elif ps < 0.15: hi = max(hi-1,0); pi = min(pi+1,2)
            hmcr_sim.append(hv[hi])
            par_sim.append(pv[pi])

        ax2.step(iters, hmcr_sim, where="post", color="#534AB7", lw=2, label="HMCR")
        ax2b = ax2.twinx()
        ax2b.step(iters, par_sim, where="post", color="#D85A30", lw=2, ls="--", label="PAR")
        ax2.set_xlabel("Iteración", fontsize=10)
        ax2.set_ylabel("HMCR", color="#534AB7", fontsize=10)
        ax2b.set_ylabel("PAR", color="#D85A30", fontsize=10)
        ax2.set_title("Ejemplo de evolución de parámetros", fontsize=11)
        ax2.set_yticks([0.6, 0.8, 0.9])
        ax2b.set_yticks([0.01, 0.1, 0.2])
        ax2.grid(True, alpha=0.25, ls="--")

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── TAB 3: Simulador ──────────────────────────────────────────────────────
    with tab3:
        st.markdown("""
<div class="info-box">
Simula manualmente cómo reacciona el mecanismo adaptativo ante distintas tasas
de éxito. Ajusta la tasa de éxito en cada ventana y observa cómo cambian HMCR y PAR.
</div>
""", unsafe_allow_html=True)

        st.markdown("**Configura la tasa de éxito por ventana (cada 100 iteraciones)**")
        n_ventanas = st.slider("Número de ventanas", 5, 30, 15)
        patron = st.selectbox("Patrón de tasa de éxito", [
            "Decreciente (explorar → explotar)",
            "Creciente (explotar → explorar)",
            "Oscilante",
            "Manual (deslizadores)",
        ])

        if patron == "Manual (deslizadores)":
            ps_vals = []
            cols = st.columns(min(n_ventanas, 5))
            for i in range(n_ventanas):
                with cols[i % 5]:
                    ps_vals.append(st.slider(f"Vent. {i+1}", 0.0, 0.5, 0.2, 0.05, key=f"ps_{i}"))
        else:
            if patron == "Decreciente (explorar → explotar)":
                ps_vals = np.linspace(0.05, 0.45, n_ventanas).tolist()
            elif patron == "Creciente (explotat → explorar)":
                ps_vals = np.linspace(0.45, 0.05, n_ventanas).tolist()
            else:
                ps_vals = (0.25 + 0.2 * np.sin(np.linspace(0, 4 * np.pi, n_ventanas))).tolist()

        # Simular
        hv = [0.6, 0.8, 0.9]; pv = [0.01, 0.1, 0.2]
        hi, pi = 1, 1
        hmcr_out, par_out, accion = [], [], []
        for ps in ps_vals:
            if ps > 0.3:
                hi = min(hi+1,2); pi = max(pi-1,0); act = "Explotar ↑"
            elif ps < 0.15:
                hi = max(hi-1,0); pi = min(pi+1,2); act = "Explorar ↑"
            else:
                act = "Sin cambio"
            hmcr_out.append(hv[hi])
            par_out.append(pv[pi])
            accion.append(act)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        iters = np.arange(1, n_ventanas + 1) * 100

        ax1.bar(iters, ps_vals, width=80, color="#4e79a7", alpha=0.7)
        ax1.axhline(0.3, color="#1D9E75", ls="--", lw=1.5, label="Umbral explotar (0.30)")
        ax1.axhline(0.15, color="#D85A30", ls="--", lw=1.5, label="Umbral explorar (0.15)")
        ax1.set_ylabel("Tasa de éxito (ps)", fontsize=10)
        ax1.set_title("Simulación de adaptación de parámetros", fontsize=11)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.25, ls="--")

        ax2.step(iters, hmcr_out, where="post", color="#534AB7", lw=2.5)
        ax2.set_yticks([0.6, 0.8, 0.9])
        ax2.set_ylabel("HMCR", fontsize=10)
        ax2.grid(True, alpha=0.25, ls="--")

        ax3.step(iters, par_out, where="post", color="#D85A30", lw=2.5)
        ax3.set_yticks([0.01, 0.1, 0.2])
        ax3.set_ylabel("PAR", fontsize=10)
        ax3.set_xlabel("Iteración", fontsize=10)
        ax3.grid(True, alpha=0.25, ls="--")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Tabla de acciones
        import pandas as pd
        df = pd.DataFrame({
            "Ventana": range(1, n_ventanas+1),
            "ps": [f"{p:.2f}" for p in ps_vals],
            "HMCR": hmcr_out,
            "PAR": par_out,
            "Acción": accion,
        })
        st.dataframe(df, use_container_width=True, hide_index=True)
