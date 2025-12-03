import time
import numpy as np
from numba import njit, prange
import os
import sys

# --- CONFIGURATION / CONFIGURACIÓN ---
# Valores por defecto (se sobrescriben si existe current_data.py)
JACKPOT = 0.0        
ESTIMATION = 100000.0     
BET_PRICE = 1.0           
PRIZE_DISTRIBUTION = np.array([0.10, 0.09, 0.08, 0.08, 0.20]) 

# --- HIGH PRECISION SETTINGS / AJUSTES DE ALTA PRECISIÓN ---
PORTFOLIO_SIZE = 10       
N_SIMULATIONS = 5000000   

# --- DYNAMIC CANDIDATE SELECTION / SELECCIÓN DINÁMICA ---
MIN_EV_THRESHOLD = 1.60   
MAX_CANDIDATES_SAFETY = 1000 

# --- 1. DATA LOADING (AUTOMATED SOURCE) / CARGA AUTOMÁTICA ---
LAE_PROBS_MATRIX = None # Inicializamos vacío

# Intentamos cargar desde el archivo generado por el scraper
try:
    # Añadimos el directorio raíz al path por si current_data.py está arriba
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import quinigol.current_data as current_data
    
    print("✅ 'current_data.py' found & loaded.")
    JACKPOT = current_data.JACKPOT
    ESTIMATION = current_data.ESTIMATION
    LAE_PROBS_MATRIX = current_data.LAE_PROBS_MATRIX
    print(f"   Jackpot: {JACKPOT} | Estimation: {ESTIMATION}")
    
except ImportError:
    print("⚠️ 'current_data.py' not found.")

# --- 2. DATA LOADING (MANUAL REAL ODDS) / CARGA CUOTAS REALES ---
# Esto lo mantienes manual aquí para tener control total
real_probs_1 = [19,9.5,8,4.75,21,10,9,5.5,41,19,17,11,91,51,36,19]
real_probs_2 = [13,7.5,7.5,6,15,8,8.5,7,34,19,17,15,96,56,46,34]
real_probs_3 = [8,15,41,161,5.5,8,23,81,5.5,9,26,76,5.5,9,26,67]
real_probs_4 = [8,11,23,71,5.5,7,17,51,7,9,21,51,9,11,23,56]
real_probs_5 = [13,9,10,15,11,7,9,10,17,11,13,15,23,17,21,29]
real_probs_6 = [29,41,51,201,13,15,29,61,10,9.5,19,46,4,3.75,7.5,15]

raw_real_probs = [real_probs_1, real_probs_2, real_probs_3, real_probs_4, real_probs_5, real_probs_6]
REAL_PROBS_MATRIX = np.zeros((6, 16), dtype=np.float64)

for i in range(6):
    row = np.array(raw_real_probs[i], dtype=np.float64)
    inv_row = 1.0 / row
    REAL_PROBS_MATRIX[i, :] = inv_row / np.sum(inv_row)
# --- COMBINATIONS LOADING / CARGA DE COMBINACIONES ---
npy_path = "combinations/kinigmotza.npy"
try:
    COMBINATIONS = np.load(npy_path).astype(np.int32)
    print(f"✅ Loaded {len(COMBINATIONS)} combinations from {npy_path}")
except:
    print(f"⚠️ Warning: '{npy_path}' not found. Generating random data for testing...")
    COMBINATIONS = np.random.randint(0, 16, (50000, 6)).astype(np.int32)

# --- ANALYTICAL FILTERS (JIT) / FILTROS ANALÍTICOS ---

@njit(fastmath=True)
def calc_taxed_prize(gross):
    """Aplica impuesto del 20% a premios > 40.000"""
    if gross > 40000.0: return (gross - 40000.0) * 0.8 + 40000.0
    return gross

@njit(fastmath=True)
def poisson_prize_share(pot, lambda_val):
    """Calcula dilución del premio por múltiples acertantes"""
    if lambda_val < 1e-9: return calc_taxed_prize(pot)
    share = (1.0 - np.exp(-lambda_val)) / lambda_val
    return calc_taxed_prize(pot * share)

@njit(parallel=True, fastmath=True)
def get_top_candidates(combs, real_probs, lae_probs, est, jack, dist):
    """
    Calcula el EV Total sumando las categorías 6, 5, 4, 3 y 2.
    """
    n = combs.shape[0]
    evs = np.zeros(n, dtype=np.float64)
    
    # Pre-calcular botes por categoría (Prize Pools)
    cat_pots = est * dist 
    
    # Bote puro para la 1ª categoría
    pot6 = jack 
    
    pot5 = cat_pots[1]
    pot4 = cat_pots[2]
    pot3 = cat_pots[3]
    pot2 = cat_pots[4]
    
    for i in prange(n):
        # 1. Extraer probabilidades de la matriz para esta columna
        p_real = np.zeros(6, dtype=np.float64)
        p_lae = np.zeros(6, dtype=np.float64)
        
        for m in range(6):
            s = combs[i, m]
            p_real[m] = real_probs[m, s]
            p_lae[m] = lae_probs[m, s]
            
        # -------------------------------------------------------
        # CÁLCULO DE PROBABILIDADES (REALES Y LAE)
        # -------------------------------------------------------

        # --- 6 ACIERTOS (PLENO) ---
        prob_real_6 = 1.0
        prob_lae_6 = 1.0
        for m in range(6):
            prob_real_6 *= p_real[m]
            prob_lae_6 *= p_lae[m]
            
        # --- 5 ACIERTOS (FALLA 1) ---
        prob_real_5 = 0.0
        prob_lae_5 = 0.0
        for m in range(6):
            term_real = (1.0 - p_real[m])
            term_lae = (1.0 - p_lae[m])
            for x in range(6):
                if x != m:
                    term_real *= p_real[x]
                    term_lae *= p_lae[x]
            prob_real_5 += term_real
            prob_lae_5 += term_lae

        # --- 4 ACIERTOS (FALLAN 2) ---
        prob_real_4 = 0.0
        prob_lae_4 = 0.0
        for m in range(6):
            for k in range(m + 1, 6):
                term_real = (1.0 - p_real[m]) * (1.0 - p_real[k])
                term_lae = (1.0 - p_lae[m]) * (1.0 - p_lae[k])
                for x in range(6):
                    if x != m and x != k:
                        term_real *= p_real[x]
                        term_lae *= p_lae[x]
                prob_real_4 += term_real
                prob_lae_4 += term_lae

        # --- 3 ACIERTOS (FALLAN 3) ---
        prob_real_3 = 0.0
        prob_lae_3 = 0.0
        for m in range(6):
            for k in range(m + 1, 6):
                for j in range(k + 1, 6):
                    term_real = (1.0 - p_real[m]) * (1.0 - p_real[k]) * (1.0 - p_real[j])
                    term_lae = (1.0 - p_lae[m]) * (1.0 - p_lae[k]) * (1.0 - p_lae[j])
                    for x in range(6):
                        if x != m and x != k and x != j:
                            term_real *= p_real[x]
                            term_lae *= p_lae[x]
                    prob_real_3 += term_real
                    prob_lae_3 += term_lae

        # --- 2 ACIERTOS (ACIERTAN 2) ---
        prob_real_2 = 0.0
        prob_lae_2 = 0.0
        for m in range(6):
            for k in range(m + 1, 6):
                term_real = p_real[m] * p_real[k]
                term_lae = p_lae[m] * p_lae[k]
                for x in range(6):
                    if x != m and x != k:
                        term_real *= (1.0 - p_real[x])
                        term_lae *= (1.0 - p_lae[x])
                prob_real_2 += term_real
                prob_lae_2 += term_lae

        # -------------------------------------------------------
        # CÁLCULO DE EV
        # -------------------------------------------------------
        
        l6 = est * prob_lae_6
        ev6 = prob_real_6 * poisson_prize_share(pot6, l6)
        
        l5 = est * prob_lae_5
        ev5 = prob_real_5 * poisson_prize_share(pot5, l5)
        
        l4 = est * prob_lae_4
        ev4 = prob_real_4 * poisson_prize_share(pot4, l4)
        
        l3 = est * prob_lae_3
        ev3 = prob_real_3 * poisson_prize_share(pot3, l3)
        
        l2 = est * prob_lae_2
        ev2 = prob_real_2 * poisson_prize_share(pot2, l2)
        
        # EV TOTAL
        evs[i] = ev6 + ev5 + ev4 + ev3 + ev2
        
    return evs

# --- MONTE CARLO CORE (HEAVY DUTY) / NÚCLEO MONTECARLO ---

@njit(parallel=True)
def generate_scenarios(probs_matrix, n_sims):
    """Genera millones de jornadas ganadoras simuladas"""
    scenarios = np.zeros((n_sims, 6), dtype=np.int8) 
    
    for m in range(6):
        cdf = np.cumsum(probs_matrix[m])
        for i in prange(n_sims):
            rand_val = np.random.random()
            idx = 0
            while idx < 15 and rand_val > cdf[idx]:
                idx += 1
            scenarios[i, m] = idx
    return scenarios

@njit(parallel=True)
def precompute_hits_matrix(candidates, scenarios):
    """
    Calcula matriz de aciertos usando int8.
    """
    n_sims = scenarios.shape[0]
    n_cands = candidates.shape[0]
    hits_matrix = np.zeros((n_sims, n_cands), dtype=np.int8)
    
    for i in prange(n_sims):
        for c in range(n_cands):
            hits = 0
            for m in range(6):
                if candidates[c, m] == scenarios[i, m]:
                    hits += 1
            hits_matrix[i, c] = hits
            
    return hits_matrix


@njit(parallel=True, fastmath=True)
def precompute_scenario_prizes(scenarios, lae_probs, estimation, jackpot, dist):
    """
    Calcula cuánto pagaría cada categoría de premio en cada uno de los 5M de escenarios.
    Basado en la probabilidad LAE de que ese escenario ocurra.
    """
    n_sims = scenarios.shape[0]
    # Matriz: [Simulacion, Categoría] -> Valor del premio en Euros
    # Indices: 6=Pleno, 5=5ac, ... 
    dynamic_prizes = np.zeros((n_sims, 7), dtype=np.float32) # float32 ahorra RAM
    
    # Botes Totales (Pots)
    pot6 = jackpot + (estimation * dist[0])
    pot5 = estimation * dist[1]
    pot4 = estimation * dist[2]
    pot3 = estimation * dist[3]
    pot2 = estimation * dist[4]
    
    for i in prange(n_sims):
        # 1. Recuperar probabilidades LAE de LOS RESULTADOS QUE SALIERON en este escenario
        p_lae = np.zeros(6, dtype=np.float32)
        for m in range(6):
            res = scenarios[i, m]
            p_lae[m] = lae_probs[m, res]
            
        # 2. Calcular cuánta gente tendría X aciertos contra este escenario
        # (Es la misma lógica que get_top_candidates pero usando p_lae sobre sí mismo)
        
        # Probabilidad de que un apostante random tenga 6 aciertos (Coincida exacto)
        prob_6 = 1.0
        for m in range(6): prob_6 *= p_lae[m]
            
        # Probabilidad de que un apostante random tenga 5 aciertos
        prob_5 = 0.0
        for m in range(6):
            term = (1.0 - p_lae[m])
            for x in range(6):
                if x != m: term *= p_lae[x]
            prob_5 += term
            
        # Probabilidad de que un apostante random tenga 4 aciertos
        prob_4 = 0.0
        for m in range(6):
            for k in range(m + 1, 6):
                term = (1.0 - p_lae[m]) * (1.0 - p_lae[k])
                for x in range(6):
                    if x != m and x != k: term *= p_lae[x]
                prob_4 += term
        
        # Probabilidad de que un apostante random tenga 3 aciertos (Aprox rápida)
        # Para 3 y 2 usamos aproximación si queremos velocidad extrema, 
        # pero Numba aguanta el cálculo completo.
        prob_3 = 0.0
        for m in range(6):
            for k in range(m + 1, 6):
                for j in range(k + 1, 6):
                    term = (1.0 - p_lae[m]) * (1.0 - p_lae[k]) * (1.0 - p_lae[j])
                    for x in range(6):
                        if x != m and x != k and x != j: term *= p_lae[x]
                    prob_3 += term

        # Probabilidad de que un apostante random tenga 2 aciertos
        prob_2 = 0.0
        for m in range(6):
            for k in range(m + 1, 6):
                # Acierta m y k, falla el resto
                term = p_lae[m] * p_lae[k]
                for x in range(6):
                    if x != m and x != k: term *= (1.0 - p_lae[x])
                prob_2 += term

        # 3. Calcular Premios Dinámicos (Euros) usando Poisson Share
        # Lambda = Cuántos ganadores se esperan
        
        # Cat 6
        l6 = estimation * prob_6
        dynamic_prizes[i, 6] = poisson_prize_share(pot6, l6)
        
        # Cat 5
        l5 = estimation * prob_5
        dynamic_prizes[i, 5] = poisson_prize_share(pot5, l5)
        
        # Cat 4
        l4 = estimation * prob_4
        dynamic_prizes[i, 4] = poisson_prize_share(pot4, l4)
        
        # Cat 3
        l3 = estimation * prob_3
        dynamic_prizes[i, 3] = poisson_prize_share(pot3, l3)
        
        # Cat 2
        l2 = estimation * prob_2
        dynamic_prizes[i, 2] = poisson_prize_share(pot2, l2)
        
    return dynamic_prizes

# Elige tu objetivo:
# 1 = "PROB_PROFIT": Maximiza la probabilidad de recuperar la inversión (muchos premios pequeños).
# 2 = "SORTINO": Busca rentabilidad penalizando solo las pérdidas (Equilibrado).
OPTIMIZATION_MODE = 2

# --- OPTIMIZED METRIC FUNCTIONS (JIT) ---

@njit(fastmath=True)
def calculate_metric_numba(current_earnings, new_hits, dynamic_prizes, mode, cost_so_far):
    """
    Calcula la métrica de calidad de la cartera según el modo elegido.
    """
    n = len(current_earnings)
 
    # Variables acumuladoras
    sum_val = 0.0      # Para media (EV)
    sum_sq_down = 0.0  # Para Sortino (Riesgo de bajada)
    wins_count = 0.0   # Para Probabilidad de Beneficio
    
    # El coste aumenta al añadir una apuesta (Coste actual + 1)
    new_cost = cost_so_far + 1.0
    
    for i in range(n):
        val = current_earnings[i] # Ganancia actual en simulación i
        h = new_hits[i]           # Aciertos del nuevo candidato en simulación i

        # Sumamos premio DINÁMICO de esta simulación específica 'i'
        if h >= 2: # Solo si hay premio (2 a 6)
            val += dynamic_prizes[i, h]
        
        # --- CÁLCULOS SEGÚN MODO ---
        
        if mode == 1: # PROB_PROFIT
            # Contamos si en este escenario ganamos más de lo que gastamos
            if val > new_cost:
                wins_count += 1.0
                
        elif mode == 2: # SORTINO RATIO
            sum_val += val
            # Solo sumamos al riesgo si perdemos dinero (val < new_cost)
            # El riesgo es la desviación de las pérdidas al cuadrado
            if val < new_cost:
                diff = new_cost - val
                sum_sq_down += diff * diff
                

    # --- RETORNO DE RESULTADOS ---
    
    if mode == 1: # Probabilidad de Rentabilizar
        return wins_count / n
        
    elif mode == 2: # Sortino Ratio
        mean_profit = (sum_val / n) - new_cost # Beneficio neto medio
        if sum_sq_down < 1e-9: 
            return 999999.0 
        downside_deviation = np.sqrt(sum_sq_down / n)
        return mean_profit / downside_deviation

    return 0.0

@njit
def update_earnings_numba(current_earnings, new_hits, dynamic_prizes):
    """Actualiza el array de ganancias acumuladas (Incluye 2 aciertos)"""
    n = len(current_earnings)
    for i in range(n):
        h = new_hits[i]
        if h >= 2:
            current_earnings[i] += dynamic_prizes[i, h]

def greedy_portfolio_selection(hits_matrix, dynamic_prizes, candidate_indices, combinations, estimation, jackpot, prize_dist, target_size, mode):
    n_sims, n_cands = hits_matrix.shape
    selected_local_indices = []
    
    current_portfolio_earnings = np.zeros(n_sims, dtype=np.float64)
    
    mode_names = {1: "PROBABILITY OF PROFIT", 2: "SORTINO RATIO (Organic Diversification)"}
    print(f"   > Strategy: {mode_names.get(mode, 'Unknown')}")
    print(f"   > Starting Greedy Selection Loop ({target_size} steps)...")
    
    for step in range(target_size):
        best_candidate = -1
        best_metric = -float('inf')
        
        # Coste acumulado actual (para calcular el downside risk correctamente)
        current_cost = float(step)
        
        # Iterar sobre candidatos
        for c in range(n_cands):
            if c in selected_local_indices: continue
            
            # --- OPTIMIZED METRIC CALCULATION ---
            metric = calculate_metric_numba(current_portfolio_earnings, hits_matrix[:, c], dynamic_prizes, mode, current_cost)
            
            if metric > best_metric:
                best_metric = metric
                best_candidate = c
        
        if (step % 1) == 0: 
            sys.stdout.write(f"\r     [Step {step+1}/{target_size}] Best Candidate found (Metric: {best_metric:.4f})")
            sys.stdout.flush()

        if best_candidate != -1:
            curr_comb = combinations[candidate_indices[best_candidate]]
            max_matches = 0
            most_similar_idx = -1
            
            if len(selected_local_indices) > 0:
                for i, saved_c_idx in enumerate(selected_local_indices):
                    saved_comb = combinations[candidate_indices[saved_c_idx]]
                    # Contamos coincidencias (0 a 6)
                    matches = np.sum(curr_comb == saved_comb)
                    if matches > max_matches:
                        max_matches = matches
                        most_similar_idx = i + 1 # +1 para que coincida con "Bet #1"
                
                diff_msg = f"(Max overlap: {max_matches}/6 with Bet #{most_similar_idx})"
            else:
                diff_msg = "(Initial Anchor Bet)"

            selected_local_indices.append(best_candidate)
            update_earnings_numba(current_portfolio_earnings, hits_matrix[:, best_candidate], dynamic_prizes)
            
            # Imprimimos la selección confirmada
            print(f" -> Selected #{best_candidate} {diff_msg}")
            
        else:
            print("\n     No more valid candidates.")
            break
            
    print("\n")        
    return selected_local_indices, current_portfolio_earnings

# --- EXECUTION / EJECUCIÓN ---

if __name__ == "__main__":
    start_total = time.time()
    
    print(f"--- QUINIGOL MONTE CARLO (CPU MANUAL - DYNAMIC POOL) ---")
    print(f"Simulations: {N_SIMULATIONS:,}")
    print(f"Min EV Threshold: {MIN_EV_THRESHOLD}")
    
    # 1. Pre-filtro
    print("1. Filtering top candidates via Analytical EV...")
    evs = get_top_candidates(COMBINATIONS, REAL_PROBS_MATRIX, LAE_PROBS_MATRIX, ESTIMATION, JACKPOT, PRIZE_DISTRIBUTION)
    
    # --- DYNAMIC SELECTION LOGIC ---
    # Seleccionamos todo lo que supere el umbral
    mask_good = evs > MIN_EV_THRESHOLD
    count_good = np.sum(mask_good)
    
    print(f"   Found {count_good} columns with EV > {MIN_EV_THRESHOLD}")
    
    if count_good == 0:
        print("⚠️ No columns meet the EV threshold! Lowering threshold or using Top 100 as fallback.")
        top_indices = evs.argsort()[::-1][:100]
    else:
        # Indices de los buenos
        good_indices = np.where(mask_good)[0]
        
        # Si hay demasiados, ordenamos y cogemos los mejores hasta el tope de seguridad
        if len(good_indices) > MAX_CANDIDATES_SAFETY:
            print(f" Limiting candidate pool to {MAX_CANDIDATES_SAFETY} for RAM safety (Top {MAX_CANDIDATES_SAFETY} of {len(good_indices)})")
            # Ordenamos los buenos por EV descendente
            sorted_good = good_indices[np.argsort(evs[good_indices])[::-1]]
            top_indices = sorted_good[:MAX_CANDIDATES_SAFETY]
        else:
            # Si caben todos, usamos todos (¡Diversificación máxima!)
            top_indices = good_indices
            
    print(f"   Candidate Pool Size: {len(top_indices)}")
    
    top_combinations = COMBINATIONS[top_indices]
    
    # --- DEBUGGING: PRINT TOP 10 CANDIDATES ---
    print("\n--- TOP 10 ANALYTICAL CANDIDATES (DEBUG) ---")
    # Para el debug, usamos los 10 mejores absolutos, no solo del pool
    top_10_debug = evs.argsort()[::-1][:10]
    result_labels_debug = ["0-0","0-1","0-2","0-M","1-0","1-1","1-2","1-M","2-0","2-1","2-2","2-M","M-0","M-1","M-2","M-M"]
    for i in range(10):
        idx = top_10_debug[i]
        ev = evs[idx]
        comb = COMBINATIONS[idx]
        txt = " | ".join([result_labels_debug[c] for c in comb])
        print(f"#{i+1}: EV={ev:.4f} | {txt}")
    print("--------------------------------------------\n")
    
    # 2. Generación Escenarios
    print(f"2. Generating {N_SIMULATIONS:,} scenarios...")
    t0 = time.time()
    scenarios = generate_scenarios(REAL_PROBS_MATRIX, N_SIMULATIONS)
    print(f"   Done in {time.time()-t0:.2f}s")
    
    # --- PRE-CÁLCULO DE PREMIOS DINÁMICOS ---
    print(f"2b. Calculating Dynamic Prizes for {N_SIMULATIONS:,} scenarios (Using LAE data)...")
    t0 = time.time()
    # Esta matriz ocupará: 5M * 7 * 4bytes = 140MB (Muy ligero)
    dynamic_prizes = precompute_scenario_prizes(scenarios, LAE_PROBS_MATRIX, ESTIMATION, JACKPOT, PRIZE_DISTRIBUTION)
    print(f"   Done in {time.time()-t0:.2f}s. (Prizes adjusted by difficulty)")

    # 3. Matriz Aciertos
    mem_gb = (N_SIMULATIONS * len(top_indices)) / (1024**3)
    print(f"3. Building Hits Matrix (~{mem_gb:.2f} GB RAM required)...")
    t0 = time.time()
    hits_matrix = precompute_hits_matrix(top_combinations, scenarios)
    print(f"   Done in {time.time()-t0:.2f}s")
    
    # 4. Optimización
    print("4. Optimizing Portfolio (Greedy Sharpe Selection)...")
    sel_local_indices, final_earnings = greedy_portfolio_selection(
        hits_matrix, dynamic_prizes, top_indices, COMBINATIONS, ESTIMATION, JACKPOT, PRIZE_DISTRIBUTION, PORTFOLIO_SIZE, OPTIMIZATION_MODE
    )
    
    final_global_indices = top_indices[sel_local_indices]
    
    # 5. Resultados
    print("\n" + "="*50)
    print(f"   PORTFOLIO REPORT")
    print("="*50)
    
    # --- CÁLCULO DE EV SIMULADO (MONTECARLO) ---
    # Media de ganancias totales por jornada en la simulación
    simulated_ev_per_round = np.mean(final_earnings)
    # ROI Simulado
    simulated_roi = ((simulated_ev_per_round - len(final_global_indices)) / len(final_global_indices)) * 100
    
    # --- CÁLCULO DE EV TEÓRICO (ANALÍTICO) ---
    # Recuperamos los EVs individuales que calculó 'get_top_candidates' al principio
    # Como son aditivos, simplemente los sumamos.
    theoretical_ev_individual = evs[final_global_indices]
    theoretical_total_ev = np.sum(theoretical_ev_individual)
    # ROI Teórico
    theoretical_roi = ((theoretical_total_ev - len(final_global_indices)) / len(final_global_indices)) * 100
    
    # --- METRICAS DE RIESGO (Solo posibles vía Montecarlo) ---
    prob_profit = np.mean(final_earnings > len(final_global_indices)) * 100
    prob_any_prize = np.mean(final_earnings > 0) * 100
    
    # Desviación típica de tus ganancias (Volatilidad)
    std_dev_earnings = np.std(final_earnings)
    
    # --- IMPRESIÓN DEL INFORME COMPARATIVO ---
    print(f"Coste por Jornada:      {len(final_global_indices)} €")
    print("-" * 30)
    print(f"EV TEÓRICO (Suma):      {theoretical_total_ev:.4f} €  (ROI: {theoretical_roi:+.2f}%)")
    print(f"EV SIMULADO (Media):    {simulated_ev_per_round:.4f} €  (ROI: {simulated_roi:+.2f}%)")
    print("-" * 30)
    
    # Interpretación automática
    diff = abs(theoretical_total_ev - simulated_ev_per_round)
    if diff < (theoretical_total_ev * 0.05):
        print("✅ La simulación valida la teoría (Diferencia < 5%).")
    else:
        print("⚠️ Divergencia detectada. La simulación ha encontrado escenarios complejos.")
        
    print("-" * 30)
    print(f"RIESGO / ESTABILIDAD:")
    print(f"Prob. de Rentabilizar:  {prob_profit:.2f}% (Ganar > {len(final_global_indices)}€)")
    print(f"Prob. de Cobrar Algo:   {prob_any_prize:.2f}%")
    print(f"Volatilidad (StdDev):   {std_dev_earnings:.2f} €")
    
    # Recalcular aciertos exactos para la curiosidad
    final_hits = hits_matrix[:, sel_local_indices]
    best_hit = np.max(final_hits, axis=1) # El mejor acierto de la jornada
    count_6 = np.sum(best_hit == 6)
    
    print(f"Plenos (6) estimados:   {count_6} en {N_SIMULATIONS:,} simulaciones.")
    
    print("\n--- SELECTED BETS ---")
    labels = ["0-0","0-1","0-2","0-M","1-0","1-1","1-2","1-M","2-0","2-1","2-2","2-M","M-0","M-1","M-2","M-M"]
    
    os.makedirs("results", exist_ok=True)
    
    with open("results/resulQuinigolOptimizer.txt", "w") as f:
        for i, idx in enumerate(final_global_indices):
            comb = COMBINATIONS[idx]
            txt = " | ".join([labels[c] for c in comb])
            line = "".join([labels[c].replace("-", "") for c in comb])
            
            print(f"Bet #{i+1}: {txt}")
            f.write(line + "\n")
            
    print(f"\nTotal Execution Time: {time.time() - start_total:.2f}s")