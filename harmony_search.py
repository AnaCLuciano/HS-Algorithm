"""
=============================================================================
Proyecto Final - Algoritmos Bioinspirados
Búsqueda Armónica con Adaptación de Parámetros en Línea
para Optimización de Funciones Benchmark (P3)

Equipo:
  - Luciano Alvarado Ana Cristina
  - Peña Gálvez Iokebed
  - Salinas González Diego Nahum

Grupo: 5BM2  |  IIA - ESCOM - IPN
Profesora: Dra. Miriam Pescador Rojas
=============================================================================

Descripción del algoritmo:
  Implementación de Harmony Search (HS) con adaptación dinámica de los
  parámetros HMCR (Harmony Memory Considering Rate) y PAR (Pitch Adjusting
  Rate) basada en el conteo de éxitos durante la búsqueda, tal como lo
  requiere la práctica P3.

Uso:
  python harmony_search.py

Dependencias:
  pip install numpy scipy matplotlib
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# =============================================================================
# 1. FUNCIONES BENCHMARK
# =============================================================================

def ackley(x):
    """
    Función Ackley multimodal.
    Óptimo global: f(0,...,0) = 0. Dominio: [-32.768, 32.768]^n.
    """
    n = len(x)
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
            - np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
            + 20 + np.e)


def griewank(x):
    """
    Función Griewank no separable.
    Óptimo global: f(0,...,0) = 0. Dominio: [-600, 600]^n.
    """
    part1 = np.sum(x**2) / 4000
    part2 = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return part1 - part2 + 1


def rastrigin(x):
    """
    Función Rastrigin altamente multimodal (~10^n mínimos locales).
    Óptimo global: f(0,...,0) = 0. Dominio: [-5.12, 5.12]^n.
    """
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def rosenbrock(x):
    """
    Función Rosenbrock unimodal (valle curvo y estrecho).
    Óptimo global: f(1,...,1) = 0. Dominio: [-2.048, 2.048]^n.
    """
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


# Diccionario de problemas: nombre -> (función, lb, ub)
PROBLEMS = {
    "Ackley":     (ackley,     -32.768,  32.768),
    "Griewank":   (griewank,   -600.0,   600.0),
    "Rastrigin":  (rastrigin,  -5.12,    5.12),
    "Rosenbrock": (rosenbrock, -2.048,   2.048),
}


# =============================================================================
# 2. BÚSQUEDA ARMÓNICA CON ADAPTACIÓN DE PARÁMETROS
# =============================================================================

def harmony_search_adaptive(func, lb, ub, dim=10, HMS=50,
                            max_iter=5000, seed=None):
    """
    Implementación de Harmony Search con adaptación dinámica de parámetros.

    Parámetros de entrada:
    ----------------------
    func     : función objetivo (minimización).
    lb, ub   : límites inferior y superior del dominio (escalares).
    dim      : dimensionalidad del problema (número de variables).
    HMS      : tamaño de la memoria armónica (Harmony Memory Size).
    max_iter : número máximo de iteraciones.
    seed     : semilla para reproducibilidad.

    Retorna:
    --------
    best_fit  : mejor valor de fitness encontrado.
    history   : lista con el mejor fitness por iteración.

    Descripción del mecanismo adaptativo:
    --------------------------------------
    Los parámetros HMCR y PAR se adaptan según el porcentaje de éxitos
    en una ventana deslizante de 'window' iteraciones:
      - Si ps > 0.3  → se aumenta HMCR y se reduce PAR (explotar).
      - Si ps < 0.15 → se reduce HMCR y se aumenta PAR (explorar).
      - En otro caso → sin cambio.

    Valores discretos permitidos:
      HMCR ∈ {0.6, 0.8, 0.9}
      PAR  ∈ {0.01, 0.1, 0.2}
    """
    rng = np.random.RandomState(seed)
    lb_arr = np.full(dim, lb)
    ub_arr = np.full(dim, ub)

    # --- Valores discretos para la adaptación ---
    hmcr_values = [0.6, 0.8, 0.9]
    par_values  = [0.01, 0.1, 0.2]

    # Índices actuales (comenzamos en valores medios)
    hmcr_idx = 1   # HMCR = 0.8
    par_idx  = 1   # PAR  = 0.1

    HMCR = hmcr_values[hmcr_idx]
    PAR  = par_values[par_idx]
    BW   = 0.05 * (ub - lb)   # Ancho de banda de ajuste de tono

    # --- Inicialización de la Memoria Armónica (HM) ---
    HM      = lb_arr + (ub_arr - lb_arr) * rng.rand(HMS, dim)
    HM_fit  = np.array([func(h) for h in HM])

    # Índice del peor y mejor en la HM
    best_idx = np.argmin(HM_fit)
    worst_idx = np.argmax(HM_fit)

    history = [HM_fit[best_idx]]

    # Ventana para contar éxitos
    window = 100
    success_buffer = []

    for iteration in range(max_iter):
        # --- Improvisación de nueva armonía ---
        new_harmony = np.zeros(dim)
        for j in range(dim):
            r1 = rng.rand()
            if r1 < HMCR:
                # Considerar memoria armónica
                idx = rng.randint(0, HMS)
                new_harmony[j] = HM[idx, j]
                # Ajuste de tono (Pitch Adjustment)
                r2 = rng.rand()
                if r2 < PAR:
                    adjustment = BW * rng.uniform(-1, 1)
                    new_harmony[j] += adjustment
                    new_harmony[j] = np.clip(new_harmony[j],
                                             lb_arr[j], ub_arr[j])
            else:
                # Generación aleatoria
                new_harmony[j] = rng.uniform(lb_arr[j], ub_arr[j])

        # --- Evaluación ---
        new_fit = func(new_harmony)

        # --- Actualización de la HM ---
        success = False
        if new_fit < HM_fit[worst_idx]:
            HM[worst_idx]     = new_harmony
            HM_fit[worst_idx] = new_fit
            worst_idx         = np.argmax(HM_fit)
            success           = True

        # Actualizar mejor global
        current_best_idx = np.argmin(HM_fit)
        history.append(HM_fit[current_best_idx])

        # --- Adaptación de parámetros cada 'window' iteraciones ---
        success_buffer.append(1 if success else 0)
        if len(success_buffer) > window:
            success_buffer.pop(0)

        if (iteration + 1) % window == 0 and len(success_buffer) == window:
            ps = sum(success_buffer) / window   # tasa de éxito

            if ps > 0.3:
                # Explotar: aumentar HMCR, reducir PAR
                hmcr_idx = min(hmcr_idx + 1, len(hmcr_values) - 1)
                par_idx  = max(par_idx  - 1, 0)
            elif ps < 0.15:
                # Explorar: reducir HMCR, aumentar PAR
                hmcr_idx = max(hmcr_idx - 1, 0)
                par_idx  = min(par_idx  + 1, len(par_values) - 1)
            # else: sin cambio

            HMCR = hmcr_values[hmcr_idx]
            PAR  = par_values[par_idx]

    best_fit = np.min(HM_fit)
    return best_fit, history


# =============================================================================
# 3. ALGORITMOS DE REFERENCIA DE LA PRÁCTICA P3
#    (AG y ED/best/1/bin tal como se implementaron en la práctica original)
# =============================================================================

def binary_tournament(pop, fitness, k=2, rng=None):
    """Selección por torneo binario. Retorna el individuo con menor fitness."""
    if rng is None:
        rng = np.random
    idx = rng.randint(0, len(pop), k)
    return pop[idx[np.argmin(fitness[idx])]]


def sbx_crossover(p1, p2, eta=15, rng=None):
    """Cruza SBX (Simulated Binary Crossover). Pc = 0.9."""
    if rng is None:
        rng = np.random
    if rng.rand() > 0.9:
        return p1.copy(), p2.copy()
    u = rng.rand(len(p1))
    beta = np.where(u <= 0.5,
                    (2 * u) ** (1 / (eta + 1)),
                    (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    return c1, c2


def polynomial_mutation(x, lb, ub, eta=20, pm=0.3, rng=None):
    """Mutación Polinomial. Pm = 0.3."""
    if rng is None:
        rng = np.random
    x_mut = x.copy()
    for i in range(len(x)):
        if rng.rand() < pm:
            u = rng.rand()
            delta = ((2 * u) ** (1 / (eta + 1)) - 1 if u < 0.5
                     else 1 - (2 * (1 - u)) ** (1 / (eta + 1)))
            x_mut[i] = np.clip(x_mut[i] + delta * (ub[i] - lb[i]),
                                lb[i], ub[i])
    return x_mut


def genetic_algorithm(func, lb_s, ub_s, dim=10, pop_size=100,
                      gens=5000, seed=None):
    """
    Algoritmo Genético (AG) con SBX + Mutación Polinomial.
    Referencia de comparación de la práctica P3.
    """
    rng = np.random.RandomState(seed)
    lb = np.full(dim, lb_s)
    ub = np.full(dim, ub_s)
    pop = rng.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([func(ind) for ind in pop])
    history = [np.min(fitness)]
    for _ in range(gens):
        new_pop = []
        mejores_indices = np.argsort(fitness)[:2]
        new_pop.extend([pop[mejores_indices[0]].copy(), pop[mejores_indices[1]].copy()])

        for _ in range((pop_size - 2) // 2):
            p1 = binary_tournament(pop, fitness, rng=rng)
            p2 = binary_tournament(pop, fitness, rng=rng)
            c1, c2 = sbx_crossover(p1, p2, rng=rng)
            new_pop.extend([polynomial_mutation(c1, lb, ub, rng=rng),
                            polynomial_mutation(c2, lb, ub, rng=rng)])
        pop = np.array(new_pop)
        fitness = np.array([func(ind) for ind in pop])
        history.append(np.min(fitness))
    return np.min(fitness), history


def de_best_1_bin(func, lb_s, ub_s, dim=10, N=100, Gmax=5000,
                  F=0.6, Cr=0.9, seed=None):
    """
    Evolución Diferencial DE/best/1/bin.
    Referencia de comparación de la práctica P3 (segundo mejor algoritmo).
    """
    rng = np.random.RandomState(seed)
    lb = np.full(dim, lb_s)
    ub = np.full(dim, ub_s)
    pop = lb + (ub - lb) * rng.rand(N, dim)
    fitness = np.array([func(ind) for ind in pop])
    history = [np.min(fitness)]
    for _ in range(Gmax):
        best_idx = np.argmin(fitness)
        x_best = pop[best_idx]
        for i in range(N):
            idxs = [j for j in range(N) if j != i]
            r1, r2 = rng.choice(idxs, 2, replace=False)
            v = x_best + F * (pop[r1] - pop[r2])
            v = np.clip(v, lb, ub)
            # Cruza binaria
            j_rand = rng.randint(0, dim)
            mask = rng.rand(dim) < Cr
            mask[j_rand] = True
            trial = np.where(mask, v, pop[i])
            f_trial = func(trial)
            if f_trial <= fitness[i]:
                pop[i]     = trial
                fitness[i] = f_trial
        history.append(np.min(fitness))
    return np.min(fitness), history


# =============================================================================
# 4. EXPERIMENTO PRINCIPAL
# =============================================================================

def run_experiments(n_runs=30, dim=10, max_iter=5000, HMS=50,
                    results_dir="resultados"):
    """
    Ejecuta 30 ejecuciones independientes para cada combinación de
    algoritmo × función benchmark.

    Algoritmos comparados:
      - HS_adapt  : Búsqueda Armónica con adaptación de parámetros (propuesta)
      - GA        : Algoritmo Genético (referencia P3)
      - DE_best   : DE/best/1/bin (referencia P3, mejor variante)

    Retorna diccionarios con resultados finales e historiales.
    """
    os.makedirs(results_dir, exist_ok=True)

    algos = ["HS_adapt", "GA", "DE_best"]
    results  = {p: {a: [] for a in algos} for p in PROBLEMS}
    histories = {p: {a: [] for a in algos} for p in PROBLEMS}

    for p_name, (p_func, lb, ub) in PROBLEMS.items():
        print(f"\n{'='*60}")
        print(f"  Problema: {p_name}  |  dominio: [{lb}, {ub}]")
        print(f"{'='*60}")

        for run in range(n_runs):
            seed = run * 100 + 42

            evaluaciones_totales = 5000
            tam_poblacion = 100
            generaciones_justas = evaluaciones_totales // tam_poblacion
            # --- Búsqueda Armónica adaptativa ---
            f_hs, h_hs = harmony_search_adaptive(
                p_func, lb, ub, dim=dim, HMS=HMS,
                max_iter=max_iter, seed=seed
            )
            results[p_name]["HS_adapt"].append(f_hs)
            histories[p_name]["HS_adapt"].append(h_hs)

            # --- Algoritmo Genético ---
            f_ga, h_ga = genetic_algorithm(
                p_func, lb, ub, dim=dim, 
                pop_size=tam_poblacion, gens=generaciones_justas, seed=seed
            )
            results[p_name]["GA"].append(f_ga)
            histories[p_name]["GA"].append(h_ga)

            # --- DE/best/1/bin ---
            f_de, h_de = de_best_1_bin(
                p_func, lb, ub, dim=dim, 
                N=tam_poblacion, Gmax=generaciones_justas, F=0.6, Cr=0.9, seed=seed
            )
            results[p_name]["DE_best"].append(f_de)
            histories[p_name]["DE_best"].append(h_de)

            if (run + 1) % 10 == 0:
                print(f"  [{p_name}] Ejecución {run+1}/{n_runs} completada")

    return results, histories


# =============================================================================
# 5. REPORTE ESTADÍSTICO Y PRUEBA DE WILCOXON
# =============================================================================

def print_statistics(results):
    """
    Imprime la tabla comparativa con media y desviación estándar
    para cada combinación algoritmo × función.
    """
    algos = ["HS_adapt", "GA", "DE_best"]
    header = f"{'Problema':<12} | {'Algoritmo':<12} | {'Media':<14} | {'Std Dev':<14}"
    print("\n" + "=" * len(header))
    print("TABLA COMPARATIVA DE RESULTADOS (30 ejecuciones)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for p in PROBLEMS:
        for a in algos:
            vals = results[p][a]
            print(f"{p:<12} | {a:<12} | {np.mean(vals):.4e}     | {np.std(vals):.4e}")
        print("-" * len(header))


def wilcoxon_test(results):
    """
    Aplica la prueba de Wilcoxon entre HS_adapt y el mejor algoritmo
    de referencia por función.
    """
    print("\n" + "=" * 60)
    print("PRUEBA DE WILCOXON (α = 0.05) — HS_adapt vs mejor referencia")
    print("=" * 60)
    for p in PROBLEMS:
        r_hs = np.array(results[p]["HS_adapt"])
        r_ga = np.array(results[p]["GA"])
        r_de = np.array(results[p]["DE_best"])

        # Seleccionar la mejor referencia (menor media)
        if np.mean(r_ga) < np.mean(r_de):
            ref_name, ref_vals = "GA", r_ga
        else:
            ref_name, ref_vals = "DE_best", r_de

        try:
            stat, p_val = stats.wilcoxon(r_hs, ref_vals)
            sig = "Diferencia SIGNIFICATIVA" if p_val < 0.05 else "Sin diferencia significativa"
        except ValueError:
            p_val, sig = 1.0, "Muestras idénticas"

        print(f"  {p:<12} | HS_adapt vs {ref_name:<8} | p = {p_val:.4f} | {sig}")


# =============================================================================
# 6. GRÁFICAS DE CONVERGENCIA
# =============================================================================

def plot_convergence(histories, results_dir="resultados"):
    """
    Genera gráficas de convergencia mediana (escala logarítmica)
    para cada función benchmark y las guarda como PNG.
    """
    algos      = ["HS_adapt", "GA", "DE_best"]
    colors     = {"HS_adapt": "#1f77b4", "GA": "#ff7f0e", "DE_best": "#2ca02c"}
    labels     = {"HS_adapt": "HS Adaptativo", "GA": "Alg. Genético",
                  "DE_best": "DE/best/1/bin"}

    for p_name in PROBLEMS:
        plt.figure(figsize=(8, 5))
        for a in algos:
            h_array = np.array(histories[p_name][a])
            median_curve = np.median(h_array, axis=0)
            if a == "HS_adapt":
                # HS hace 1 evaluación por iteración
                x_vals = np.arange(1, len(median_curve) + 1)
            else:
                # AG y DE hacen 100 evaluaciones por generación
                x_vals = np.arange(1, len(median_curve) + 1) * 100

            plt.plot(x_vals, median_curve, label=labels[a], color=colors[a], linewidth=1.8)
        plt.yscale("log")
        plt.xlabel("Iteración / Generación", fontsize=11)
        plt.ylabel("Mejor Fitness (escala log)", fontsize=11)
        plt.title(f"Convergencia Mediana — {p_name}", fontsize=13)
        plt.legend(fontsize=10)
        plt.tight_layout()
        fname = os.path.join(results_dir, f"convergencia_{p_name.lower()}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Gráfica guardada: {fname}")


def plot_boxplots(results, results_dir="resultados"):
    """
    Genera diagramas de caja para comparar la distribución de
    resultados finales por función y algoritmo.
    """
    algos  = ["HS_adapt", "GA", "DE_best"]
    labels = ["HS Adapt.", "Alg. Gen.", "DE/best"]
    colors = ["#4e79a7", "#f28e2b", "#59a14f"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for i, p_name in enumerate(PROBLEMS):
        data = [results[p_name][a] for a in algos]
        bp = axes[i].boxplot(data, patch_artist=True, notch=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[i].set_yscale("log")
        axes[i].set_title(f"{p_name}", fontsize=12)
        axes[i].set_xticklabels(labels, fontsize=10)
        axes[i].set_ylabel("Fitness final", fontsize=10)

    fig.suptitle("Distribución de Resultados Finales (30 ejecuciones)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fname = os.path.join(results_dir, "boxplots_comparacion.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Gráfica guardada: {fname}")


# =============================================================================
# 7. PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    print("Búsqueda Armónica Adaptativa — Proyecto Final Algoritmos Bioinspirados")
    print("ESCOM-IPN  |  Grupo 5BM2  |  Dra. Miriam Pescador Rojas\n")

    N_RUNS   = 30
    DIM      = 10
    MAX_ITER = 5000
    HMS      = 50
    OUT_DIR  = "resultados"

    print(f"Configuración: {N_RUNS} ejecuciones, dim={DIM}, max_iter={MAX_ITER}, HMS={HMS}")

    results, histories = run_experiments(
        n_runs=N_RUNS, dim=DIM, max_iter=MAX_ITER,
        HMS=HMS, results_dir=OUT_DIR
    )

    print_statistics(results)
    wilcoxon_test(results)

    print("\nGenerando gráficas...")
    plot_convergence(histories, results_dir=OUT_DIR)
    plot_boxplots(results, results_dir=OUT_DIR)

    print("\n¡Experimento completado! Revisa la carpeta 'resultados/'.")
