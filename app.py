"""
=============================================================================
Laboratorio Interactivo — Búsqueda Armónica Adaptativa
Proyecto Final · Algoritmos Bioinspirados · ESCOM-IPN · Grupo 5BM2
Profesora: Dra. Miriam Pescador Rojas
=============================================================================
Ejecutar con:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="HS Lab — Algoritmos Bioinspirados",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos globales ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Barra lateral */
    [data-testid="stSidebar"] { background: #1a1240; }
    [data-testid="stSidebar"] * { color: #d4cff7 !important; }
    [data-testid="stSidebar"] .stRadio > label { color: #a89ee8 !important; }

    /* Encabezado principal */
    .main-header {
        background: linear-gradient(135deg, #2d1b8e 0%, #534AB7 60%, #7F77DD 100%);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: white !important; margin: 0; font-size: 1.7rem; }
    .main-header p  { color: #d4cff7; margin: 0.25rem 0 0; font-size: 0.9rem; }

    /* Tarjetas métricas */
    .metric-card {
        background: #f8f7ff;
        border: 1px solid #e0deff;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        text-align: center;
    }
    .metric-label { font-size: 0.78rem; color: #7a72b8; font-weight: 500; margin-bottom: 4px; }
    .metric-value { font-size: 1.4rem; font-weight: 700; color: #2d1b8e; }
    .metric-sub   { font-size: 0.72rem; color: #a89ee8; margin-top: 2px; }

    /* Badges de estado */
    .badge-best { background:#eeedfe; color:#3C3489; padding:2px 10px;
                  border-radius:20px; font-size:0.78rem; font-weight:600; }
    .badge-ok   { background:#e1f5ee; color:#0F6E56; padding:2px 10px;
                  border-radius:20px; font-size:0.78rem; font-weight:600; }
    .badge-warn { background:#faeeda; color:#633806; padding:2px 10px;
                  border-radius:20px; font-size:0.78rem; font-weight:600; }

    /* Cajas de info */
    .info-box {
        background: #f8f7ff;
        border-left: 4px solid #534AB7;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        font-size: 0.88rem;
        color: #4a4070;
        margin-bottom: 1rem;
        line-height: 1.6;
    }
    .warn-box {
        background: #fffbf0;
        border-left: 4px solid #D85A30;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        font-size: 0.88rem;
        color: #633806;
        margin-bottom: 1rem;
    }

    /* Ocultar footer de streamlit */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Encabezado ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🎵 — Búsqueda Armónica Adaptativa</h1>
  <p>Proyecto Final · Algoritmos Bioinspirados · Grupo 5BM2 · ESCOM-IPN · Dra. Miriam Pescador Rojas</p>
</div>
""", unsafe_allow_html=True)

# ── Navegación lateral ────────────────────────────────────────────────────────
st.sidebar.markdown("## 🎵 HS Lab")
st.sidebar.markdown("**Proyecto Final — ESCOM IPN**")
st.sidebar.markdown("---")

pagina = st.sidebar.radio(
    "Navegar a:",
    [
        " Ejecución rápida",
        " Experimento completo",
        " Gráficas de convergencia",
        " Análisis comparativo",
        " Algoritmo",
        " Validación del proyecto",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Integrantes**")
st.sidebar.markdown("- Luciano Alvarado Ana Cristina")
st.sidebar.markdown("- Peña Gálvez Iokebed")
st.sidebar.markdown("- Salinas González Diego Nahum")
st.sidebar.markdown("---")
st.sidebar.caption("Grupo 5BM2 · IIA · ESCOM · IPN")

# ── Enrutador de páginas ──────────────────────────────────────────────────────
if pagina == " Ejecución rápida":
    from pages.p1_ejecucion import render
    render()
elif pagina == " Experimento completo":
    from pages.p2_experimento import render
    render()
elif pagina == " Gráficas de convergencia":
    from pages.p_convergencia import render
    render()
elif pagina == " Análisis comparativo":
    from pages.p3_analisis import render
    render()
elif pagina == " Algoritmo":
    from pages.p4_algoritmo import render
    render()
elif pagina == " Validación del proyecto":
    from pages.p5_validacion import render
    render()
