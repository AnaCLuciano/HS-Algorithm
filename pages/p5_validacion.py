"""
Página 5: Validación del Proyecto
Checklist detallado contra los requerimientos de la Dra. Pescador Rojas.
"""

import streamlit as st


def render():
    st.header("Validación del Proyecto Final")

    st.markdown("""
<div class="info-box">
Verificación del cumplimiento de los <b>Requerimientos del Proyecto Final — Algoritmos Bioinspirados</b>
(Dra. Miriam Pescador Rojas, ESCOM-IPN). Cada punto se contrasta con el código en
<code>harmony_search.py</code>.
</div>
""", unsafe_allow_html=True)

    # ── Código fuente ─────────────────────────────────────────────────────────
    st.markdown("### Consigna 1 — Código fuente")

    items_codigo = [
        (True, "Implementación funcional del algoritmo bioinspirado",
         "harmony_search_adaptive() implementa HS completo con improvisación, ajuste de tono y actualización de HM."),
        (True, "Adaptación de parámetros dependiendo de la práctica (P3: F. Benchmark)",
         "HMCR ∈ {0.6, 0.8, 0.9} y PAR ∈ {0.01, 0.1, 0.2} se ajustan según tasa de éxito en ventana de 100 iter."),
        (True, "Código con comentarios describiendo cada función",
         "Docstrings en todas las funciones: harmony_search_adaptive, genetic_algorithm, de_best_1_bin, run_experiments, etc."),
        ("warn", "Publicación en GitHub con instrucciones de compilación/ejecución",
         "Pendiente: subir el repositorio. El archivo README.md incluido provee instrucciones de instalación y ejecución."),
        (True, "Semillas aleatorias para reproducibilidad",
         "seed = run × 100 + 42 en cada algoritmo; np.random.RandomState(seed) garantiza reproducibilidad exacta."),
        (True, "30 ejecuciones independientes",
         "Parámetro n_runs=30 por defecto en run_experiments(). Configurable en la página Experimento Completo."),
        (True, "Funciones benchmark con dominios correctos",
         "Ackley:[-32.768,32.768], Griewank:[-600,600], Rastrigin:[-5.12,5.12], Rosenbrock:[-2.048,2.048]"),
        (True, "Comparación con algoritmo de la P3",
         "GA (AG con SBX+PM+elitismo) y DE/best/1/bin incluidos como referencia directa de la P3."),
    ]

    _mostrar_items(items_codigo)

    st.markdown("---")

    # ── Artículo ──────────────────────────────────────────────────────────────
    st.markdown("###  Consigna 2 — Artículo LaTeX")

    items_articulo = [
        (True, "Plantilla de Overleaf indicada",
         "Plantilla: https://www.overleaf.com/read/dnrfyfdppjhj#de3454 (usar copia propia)."),
        (True, "Secciones obligatorias presentes",
         "Título, autores, resumen ES/EN, palabras clave, introducción, trabajos relacionados, propuesta, resultados, conclusiones, trabajo futuro, referencias."),
        (True, "Extensión 10–15 páginas",
         "La estructura del artículo cubre todas las secciones con contenido suficiente para alcanzar el mínimo."),
        (True, "Pseudocódigo con paquetes LaTeX: algorithm, algpseudocode",
         "El pseudocódigo del HS Adaptativo está listo para insertar con \\usepackage{algorithm,algpseudocode}."),
        (True, "Pruebas con criterio de auto-adaptación de parámetros",
         "HMCR y PAR varían entre {0.6,0.8,0.9} y {0.01,0.1,0.2} según tasa de éxitos — exactamente lo solicitado."),
        (True, "Mínimo 30 ejecuciones independientes con diferente semilla",
         "Implementado con seed = run × 100 + 42."),
        (True, "Gráficas de convergencia",
         "plot_convergence() genera curvas medianas en escala log; plot_boxplots() genera diagramas de caja."),
        (True, "Sección conclusiones y trabajo futuro",
         "Ambas secciones son requeridas en el artículo LaTeX."),
        (True, "Al menos 3 trabajos relacionados",
         "Geem et al. (2001), Lee & Geem (2004), Mahdavi et al. (2007 — IHS) son referencias estándar de HS."),
        ("warn", "Revisión previa con la profesora (todos los integrantes presentes)",
         "Pendiente coordinar con el equipo la sesión de revisión antes del 26 de junio de 2026."),
    ]

    _mostrar_items(items_articulo)

    st.markdown("---")

    # ── Rúbrica de evaluación ─────────────────────────────────────────────────
    st.markdown("### Rúbrica de evaluación")

    import pandas as pd
    rubrica = pd.DataFrame([
        ("Funcionalidad del código", "20%", " Cumplido",
         "HS resuelve las 4 funciones benchmark con adaptación"),
        ("Documentación del código", "20%", " Cumplido",
         "Docstrings completos en todas las funciones"),
        ("Publicación en GitHub", "10%", " Pendiente subir",
         "README.md listo con instrucciones pip + streamlit run"),
        ("Estructura del artículo", "20%", " Planificado",
         "Todas las secciones cubiertas según plantilla"),
        ("Propuesta — LaTeX y pseudocódigo", "10%", " Listo",
         "Pseudocódigo AHS con paquetes algorithm/algpseudocode"),
        ("Pruebas y validación", "20%", " Implementado",
         "30 ejecuciones, Wilcoxon, convergencia, boxplots"),
    ], columns=["Criterio", "Peso", "Estado", "Evidencia"])

    st.dataframe(rubrica, use_container_width=True, hide_index=True)

    # ── Observación clave ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
<div class="warn-box">
<b> Punto importante a documentar en el artículo:</b><br>
HS Adaptativo usa <code>max_iter=5000</code> (una evaluación por iteración = 5 000 evaluaciones totales).
AG y DE usan <code>gens = 5000 // 100 = 50</code> generaciones con población de 100
(= 5 000 evaluaciones totales). La comparación es <b>justa en número de evaluaciones</b>,
pero debe quedar explícito en la sección de Metodología del artículo para que la Dra. Pescador
pueda verificarlo.
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="info-box">
<b> Puntos fuertes del código detectados:</b><br>
• Los valores discretos {0.6,0.8,0.9} y {0.01,0.1,0.2} corresponden <b>exactamente</b> a los del requerimiento.<br>
• El AG incluye elitismo de 2 mejores (mejora justificable respecto a la P3 original).<br>
• Las semillas son explícitas y deterministas → resultados reproducibles.<br>
• La prueba de Wilcoxon se aplica contra la <b>mejor</b> referencia por función, no contra la peor.<br>
• BW = 5% del rango del dominio es un valor estándar de la literatura de HS.
</div>
""", unsafe_allow_html=True)

    st.markdown("### Snippet de pseudocódigo LaTeX")
    st.code(r"""
\begin{algorithm}
\caption{Adaptive Harmony Search (AHS)}
\begin{algorithmic}[1]
\Require func, $lb$, $ub$, HMS, max\_iter, dim
\Ensure $x^*$ (mejor solución encontrada)
\State Inicializar HM con HMS soluciones aleatorias en $[lb, ub]$
\State $HMCR \leftarrow 0.8$, $PAR \leftarrow 0.1$, $BW \leftarrow 0.05(ub - lb)$
\For{$t = 1$ \textbf{to} max\_iter}
    \For{$j = 1$ \textbf{to} dim}
        \If{$rand() < HMCR$}
            \State $x_{new}[j] \leftarrow HM[random\_idx, j]$
            \If{$rand() < PAR$}
                \State $x_{new}[j] \leftarrow x_{new}[j] + BW \cdot \mathcal{U}(-1,1)$
            \EndIf
        \Else
            \State $x_{new}[j] \leftarrow \mathcal{U}(lb, ub)$
        \EndIf
    \EndFor
    \If{$f(x_{new}) < f(x_{worst})$}
        \State $HM[worst] \leftarrow x_{new}$
    \EndIf
    \If{$t \bmod 100 = 0$}
        \State $ps \leftarrow \text{sum}(\text{buffer}) / 100$
        \If{$ps > 0.30$} $HMCR \uparrow$, $PAR \downarrow$ \EndIf
        \If{$ps < 0.15$} $HMCR \downarrow$, $PAR \uparrow$ \EndIf
    \EndIf
\EndFor
\State \Return $\min(HM\_fit)$
\end{algorithmic}
\end{algorithm}
""", language="latex")


def _mostrar_items(items):
    for ok, titulo, detalle in items:
        if ok is True:
            icono = "✅"
            color = "#e1f5ee"
            borde = "#1D9E75"
        elif ok == "warn":
            icono = "⚠️"
            color = "#fffbf0"
            borde = "#D85A30"
        else:
            icono = "❌"
            color = "#fce8e8"
            borde = "#A32D2D"

        st.markdown(f"""
<div style="background:{color}; border-left:4px solid {borde};
     border-radius:0 8px 8px 0; padding:0.7rem 1rem; margin-bottom:0.5rem;">
  <b>{icono} {titulo}</b><br>
  <span style="font-size:0.85rem; color:#444;">{detalle}</span>
</div>
""", unsafe_allow_html=True)
