# 🎵 Laboratorio — Búsqueda Armónica Adaptativa

**Proyecto Final · Algoritmos Bioinspirados · ESCOM-IPN · Grupo 5BM2**  
Profesora: Dra. Miriam Pescador Rojas

---

## Integrantes

- Luciano Alvarado Ana Cristina
- Peña Gálvez Iokebed
- Salinas González Diego Nahum

---

## Descripción

Implementación de **Harmony Search (HS) con adaptación dinámica de parámetros** para optimización de funciones benchmark (Práctica 3 — F. Benchmark).

Los parámetros HMCR y PAR se ajustan automáticamente en línea:
- **HMCR** ∈ {0.6, 0.8, 0.9}
- **PAR** ∈ {0.01, 0.1, 0.2}

La adaptación se basa en la tasa de éxitos en una ventana deslizante de 100 iteraciones:
- `ps > 0.30` → explotar (HMCR↑, PAR↓)
- `ps < 0.15` → explorar (HMCR↓, PAR↑)

El algoritmo se compara con **Algoritmo Genético** y **DE/best/1/bin** de la P3.

---

## Estructura del proyecto

```
harmony_search_lab/
│
├── app.py                  # Punto de entrada de Streamlit
├── harmony_search.py       # Algoritmo principal + AG + DE (lógica pura)
├── requirements.txt        # Dependencias
├── README.md               # Este archivo
│
└── pages/
    ├── p1_ejecucion.py     # Ejecución rápida con una corrida
    ├── p2_experimento.py   # Experimento completo (N corridas, Wilcoxon)
    ├── p3_analisis.py      # Análisis comparativo interactivo
    ├── p4_algoritmo.py     # Pseudocódigo y simulador adaptativo
    └── p5_validacion.py    # Checklist contra rúbrica del proyecto
```

---

## Requisitos

- Python 3.9 o superior
- pip

---

## Instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/<tu-usuario>/harmony_search_lab.git
cd harmony_search_lab
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación

```bash
streamlit run app.py
```

La interfaz se abrirá automáticamente en tu navegador en `http://localhost:8501`.

---

## Uso de la interfaz

| Página | Descripción |
|--------|-------------|
| 🚀 Ejecución rápida | Configura parámetros y ejecuta una sola corrida. Ver convergencia y adaptación de HMCR/PAR en tiempo real. |
| 📊 Experimento completo | Ejecuta 30 corridas independientes. Genera tabla estadística y prueba de Wilcoxon (α = 0.05). |
| 📈 Análisis comparativo | Explora los resultados del experimento: convergencia, boxplots, violínes, estadísticas detalladas. |
| 🧠 Algoritmo | Pseudocódigo, mecanismo adaptativo y simulador interactivo de HMCR/PAR. |
| ✅ Validación | Checklist contra la rúbrica del proyecto final + snippet LaTeX para el artículo. |

---

## Funciones benchmark implementadas

| Función | Dominio | Óptimo |
|---------|---------|--------|
| Ackley | [-32.768, 32.768]ⁿ | f(0,...,0) = 0 |
| Griewank | [-600, 600]ⁿ | f(0,...,0) = 0 |
| Rastrigin | [-5.12, 5.12]ⁿ | f(0,...,0) = 0 |
| Rosenbrock | [-2.048, 2.048]ⁿ | f(1,...,1) = 0 |

Todas con n = 10 variables de decisión (configurable en la interfaz).

---

## Algoritmos comparados

- **HS Adaptativo** — Búsqueda Armónica con adaptación dinámica de HMCR y PAR
- **AG** — Algoritmo Genético con SBX, Mutación Polinomial y elitismo de 2 mejores
- **DE/best/1/bin** — Evolución Diferencial con vector base = mejor individuo, cruza binaria

---

## Notas metodológicas

- Todas las corridas usan semillas explícitas: `seed = run × 100 + 42`
- La comparación es justa en **evaluaciones totales**: HS usa 5 000 iteraciones (1 eval/iter); AG y DE usan 50 generaciones × 100 individuos = 5 000 evaluaciones.
- La prueba de Wilcoxon se aplica entre HS y la **mejor referencia** por función (no contra la peor).

---

## Fecha de entrega

26 de junio de 2026 — 12:00 pm (hora límite)

---

## Referencias

- Geem, Z. W., Kim, J. H., & Loganathan, G. V. (2001). A new heuristic optimization algorithm: Harmony search. *Simulation*, 76(2), 60–68.
- Lee, K. S., & Geem, Z. W. (2004). A new structural optimization method based on the harmony search algorithm. *Computers & Structures*, 82(9–10), 781–798.
- Mahdavi, M., Fesanghary, M., & Damangir, E. (2007). An improved harmony search algorithm for solving optimization problems. *Applied Mathematics and Computation*, 188(2), 1567–1579.
