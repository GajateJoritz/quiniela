import time
import numpy as np
from numba import njit, prange
import os
import sys

# --- CONFIGURATION / CONFIGURACIÓN ---
JACKPOT = 255000.0        
ESTIMATION = 100000.0     
BET_PRICE = 1.0           
PRIZE_DISTRIBUTION = np.array([0.10, 0.09, 0.08, 0.08, 0.20]) 

# --- HIGH PRECISION SETTINGS / AJUSTES DE ALTA PRECISIÓN ---
PORTFOLIO_SIZE = 10       
N_SIMULATIONS = 5000000   

# --- DYNAMIC CANDIDATE SELECTION / SELECCIÓN DINÁMICA ---
# En lugar de fijo, cogemos todo lo que supere este EV.
MIN_EV_THRESHOLD = 1.50   # Solo analizamos columnas que sean teóricamente rentables (>1.0)
MAX_CANDIDATES_SAFETY = 1000 # Tope de seguridad para no desbordar la RAM 

# --- DATA LOADING (MANUAL REAL ODDS) / CARGA DE CUOTAS REALES ---
# Pega aquí tus cuotas reales (formato decimal)
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

# --- DATA LOADING (LAE TEXT FILE) / CARGA DE ARCHIVO LAE ---
LAE_PROBS_MATRIX = np.zeros((6, 16), dtype=np.float64)
lae_file_path = "quinigol/laequinig.txt"

print(f"Loading LAE data from: {lae_file_path}")

try:
    with open(lae_file_path, mode="r") as datos:
        valores = []
        for linea in datos:
            linea = linea.replace(",", ".")
            valores.append([float(x)/100 for x in linea.strip().split(";")])
            
    if len(valores) < 12:
        print("❌ Error: 'laequinig.txt' debe tener al menos 12 líneas.")
        LAE_PROBS_MATRIX[:] = REAL_PROBS_MATRIX[:]
    else:
        for i in range(0, 12, 2):
            idx_partido = i // 2
            local = valores[i]   
            visit = valores[i+1] 
            k = 0
            for l in range(4): 
                for v in range(4): 
                    LAE_PROBS_MATRIX[idx_partido, k] = local[l] * visit[v]
                    k += 1
        print("✅ LAE Data loaded successfully.")

except FileNotFoundError:
    print(f"⚠️ Warning: '{lae_file_path}' not found. Using Real Probs as fallback.")
    LAE_PROBS_MATRIX[:] = REAL_PROBS_MATRIX[:]
except Exception as e:
    print(f"❌ Error reading LAE file: {e}")
    LAE_PROBS_MATRIX[:] = REAL_PROBS_MATRIX[:]

# --- COMBINATIONS LOADING / CARGA DE COMBINACIONES ---
npy_path = "combinations/prueba.npy"
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

@njit(fastmath=True)
def calculate_sharpe_numba(current_earnings, new_hits, prizes):
    """
    Calcula la media y desviación típica combinando el portfolio actual con el nuevo candidato.
    Todo en una sola pasada para no usar RAM extra.
    """
    n = len(current_earnings)
    sum_val = 0.0
    sum_sq = 0.0
    
    # Premios descomprimidos para acceso rápido
    p6, p5, p4, p3 = prizes[6], prizes[5], prizes[4], prizes[3]
    
    for i in range(n):
        val = current_earnings[i]
        h = new_hits[i]
        
        # Sumamos premio si corresponde
        if h == 6: val += p6
        elif h == 5: val += p5
        elif h == 4: val += p4
        elif h == 3: val += p3
        
        sum_val += val
        sum_sq += val * val
        
    mean = sum_val / n
    variance = (sum_sq / n) - (mean * mean)
    
    if variance < 1e-9: return 0.0
    return mean / np.sqrt(variance)

@njit
def update_earnings_numba(current_earnings, new_hits, prizes):
    """Actualiza el array de ganancias acumuladas"""
    n = len(current_earnings)
    p6, p5, p4, p3 = prizes[6], prizes[5], prizes[4], prizes[3]
    for i in range(n):
        h = new_hits[i]
        if h == 6: current_earnings[i] += p6
        elif h == 5: current_earnings[i] += p5
        elif h == 4: current_earnings[i] += p4
        elif h == 3: current_earnings[i] += p3

def greedy_portfolio_selection(hits_matrix, candidate_indices, combinations, estimation, jackpot, prize_dist, target_size):
    n_sims, n_cands = hits_matrix.shape
    selected_local_indices = []
    
    prizes_value = np.zeros(7)
    prizes_value[6] = calc_taxed_prize(jackpot) 
    prizes_value[5] = calc_taxed_prize(estimation * prize_dist[1] / 5.0)    
    prizes_value[4] = calc_taxed_prize(estimation * prize_dist[2] / 50.0)   
    prizes_value[3] = calc_taxed_prize(estimation * prize_dist[3] / 500.0)
    
    current_portfolio_earnings = np.zeros(n_sims, dtype=np.float64)
    
    print(f"   > Starting Greedy Selection Loop ({target_size} steps)...")
    
    for step in range(target_size):
        best_candidate = -1
        best_metric = -float('inf')
        
        # Iterar sobre candidatos
        for c in range(n_cands):
            if c in selected_local_indices: continue
            
            # --- DIVERSITY CHECK FIRST (Faster than math) ---
            curr_comb = combinations[candidate_indices[c]]
            is_too_similar = False
            for sel in selected_local_indices:
                sel_comb = combinations[candidate_indices[sel]]
                if np.sum(curr_comb == sel_comb) >= 5: # Max 4 matches equal
                    is_too_similar = True
                    break
            if is_too_similar: continue 

            # --- OPTIMIZED CALCULATION ---
            # Usamos la función compilada para no gastar RAM
            sharpe = calculate_sharpe_numba(current_portfolio_earnings, hits_matrix[:, c], prizes_value)
            
            if sharpe > best_metric:
                best_metric = sharpe
                best_candidate = c
        
        # Feedback de progreso para que sepas que no está colgado
        if (step % 1) == 0: 
            print(f"     [Calculating...] Best candidate for step {step+1} found.")

        if best_candidate != -1:
            selected_local_indices.append(best_candidate)
            # Actualizamos earnings permanentemente
            update_earnings_numba(current_portfolio_earnings, hits_matrix[:, best_candidate], prizes_value)
            
            print(f"     ✅ Selected candidate {best_candidate} (Sharpe: {best_metric:.4f})")
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
            print(f"⚠️ Limiting candidate pool to {MAX_CANDIDATES_SAFETY} for RAM safety (Top {MAX_CANDIDATES_SAFETY} of {len(good_indices)})")
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
    
    # 3. Matriz Aciertos
    mem_gb = (N_SIMULATIONS * len(top_indices)) / (1024**3)
    print(f"3. Building Hits Matrix (~{mem_gb:.2f} GB RAM required)...")
    t0 = time.time()
    hits_matrix = precompute_hits_matrix(top_combinations, scenarios)
    print(f"   Done in {time.time()-t0:.2f}s")
    
    # 4. Optimización
    print("4. Optimizing Portfolio (Greedy Sharpe Selection)...")
    sel_local_indices, final_earnings = greedy_portfolio_selection(
        hits_matrix, top_indices, COMBINATIONS, ESTIMATION, JACKPOT, PRIZE_DISTRIBUTION, PORTFOLIO_SIZE
    )
    
    final_global_indices = top_indices[sel_local_indices]
    
    # 5. Resultados
    print("\n" + "="*50)
    print(f"   PORTFOLIO REPORT")
    print("="*50)
    
    # Recalcular aciertos exactos
    final_hits = hits_matrix[:, sel_local_indices]
    best_hit = np.max(final_hits, axis=1)
    
    count_6 = np.sum(best_hit == 6)
    count_5 = np.sum(best_hit == 5)
    
    prob_profit = np.mean(final_earnings > len(final_global_indices)) * 100
    prob_any_prize = np.mean(final_earnings > 0) * 100
    
    print(f"Investment: {len(final_global_indices)} €")
    print(f"Probability of Profit: {prob_profit:.2f}%")
    print(f"Probability of Any Prize: {prob_any_prize:.2f}%")
    print(f"Jackpot (6) hits: {count_6} in {N_SIMULATIONS:,}")
    
    print("\n--- SELECTED BETS ---")
    labels = ["0-0","0-1","0-2","0-M","1-0","1-1","1-2","1-M","2-0","2-1","2-2","2-M","M-0","M-1","M-2","M-M"]
    
    os.makedirs("results", exist_ok=True)
    
    with open("results/final_portfolio_montecarlo.txt", "w") as f:
        for i, idx in enumerate(final_global_indices):
            comb = COMBINATIONS[idx]
            txt = " | ".join([labels[c] for c in comb])
            line = "".join([labels[c].replace("-", "") for c in comb])
            
            print(f"Bet #{i+1}: {txt}")
            f.write(line + "\n")
            
    print(f"\nTotal Execution Time: {time.time() - start_total:.2f}s")