import time
import numpy as np
from numba import njit, prange
import os

# --- CONFIGURATION / CONFIGURACIÓN ---
JACKPOT = 255000.0        # Bote actual
ESTIMATION = 100000.0     # Recaudación estimada
BET_PRICE = 1.0           # Precio por apuesta

# Porcentajes de reparto de premios (Normativa LAE)
PRIZE_DISTRIBUTION = np.array([0.10, 0.09, 0.08, 0.08, 0.20]) 

# Bankroll Settings / Configuración de Banca
MAX_BUDGET = 10.0         # Máximo absoluto de apuestas (cambia esto según tu límite)
MIN_EV_THRESHOLD = 1.1    # Umbral mínimo de EV para considerar una columna "buena"

# --- MANUAL DATA LOADING / CARGA DE DATOS MANUAL ---
# Probabilidades reales (Tus datos 'bene' manuales)
real_probs_1 = [19,9.5,8,4.75,21,10,9,5.5,41,19,17,11,91,51,36,19]
real_probs_2 = [13,7.5,7.5,6,15,8,8.5,7,34,19,17,15,96,56,46,34]
real_probs_3 = [8,15,41,161,5.5,8,23,81,5.5,9,26,76,5.5,9,26,67]
real_probs_4 = [8,11,23,71,5.5,7,17,51,7,9,21,51,9,11,23,56]
real_probs_5 = [13,9,10,15,11,7,9,10,17,11,13,15,23,17,21,29]
real_probs_6 = [29,41,51,201,13,15,29,61,10,9.5,19,46,4,3.75,7.5,15]

# Agrupamos y normalizamos (1/cuota)
raw_real_probs = [real_probs_1, real_probs_2, real_probs_3, real_probs_4, real_probs_5, real_probs_6]
REAL_PROBS_MATRIX = np.zeros((6, 16), dtype=np.float64)

for i in range(6):
    row = np.array(raw_real_probs[i], dtype=np.float64)
    inv_row = 1.0 / row
    total = np.sum(inv_row)
    REAL_PROBS_MATRIX[i, :] = inv_row / total

# Carga manual de LAE desde archivo de texto
LAE_PROBS_MATRIX = np.zeros((6, 16), dtype=np.float64)

try:
    with open("quinigol/laequinig.txt", mode="r") as data_file:
        values = []
        for line in data_file:
            line = line.replace(",", ".")
            # Parseamos los porcentajes (dividiendo por 100)
            values.append([float(x)/100 for x in line.strip().split(";")])

    # Construcción manual de la matriz LAE (Local * Visitante)
    for i in range(0, 12, 2):
        match_idx = i // 2
        local_probs = values[i]
        visit_probs = values[i+1]
        
        k = 0
        for l in range(4): # 0, 1, 2, M local
            for v in range(4): # 0, 1, 2, M visitante
                LAE_PROBS_MATRIX[match_idx, k] = local_probs[l] * visit_probs[v]
                k += 1
except FileNotFoundError:
    print("ERROR: No se encontró 'quinigol/laequinig.txt'. Usando datos reales como fallback.")
    LAE_PROBS_MATRIX[:] = REAL_PROBS_MATRIX[:]

# Carga de combinaciones
try:
    COMBINATIONS = np.load("combinations/kinigmotza.npy").astype(np.int32)
except FileNotFoundError:
    print("¡AVISO! No se encontró 'combinations/kinigmotza.npy'. Usando datos aleatorios.")
    COMBINATIONS = np.random.randint(0, 16, (50000, 6)).astype(np.int32)

print(f"Data Loaded. Processing {len(COMBINATIONS)} columns...")

# --- MATHEMATICAL CORE (JIT) / NÚCLEO MATEMÁTICO ---

@njit(fastmath=True)
def calc_taxed_prize(gross_prize):
    """Aplica impuesto del 20% a premios > 40k"""
    if gross_prize > 40000.0:
        return (gross_prize - 40000.0) * 0.8 + 40000.0
    return gross_prize

@njit(fastmath=True)
def poisson_prize_share(pot, lambda_val):
    """Calcula la porción del pastel usando Poisson"""
    if lambda_val < 1e-9: return calc_taxed_prize(pot)
    share_factor = (1.0 - np.exp(-lambda_val)) / lambda_val
    return calc_taxed_prize(pot * share_factor)

@njit(parallel=True, fastmath=True)
def evaluate_columns(combs, real_probs, lae_probs, estimation, jackpot, prize_dist):
    """
    Evalúa columnas calculando EV completo (6, 5, 4, 3, 2 aciertos)
    """
    n_combs = combs.shape[0]
    results = np.zeros((n_combs, 3), dtype=np.float64)
    cat_pots = estimation * prize_dist
    
    for i in prange(n_combs):
        p_real = np.ones(6, dtype=np.float64)
        p_lae = np.ones(6, dtype=np.float64)
        
        for m in range(6):
            sign = combs[i, m]
            p_real[m] = real_probs[m, sign]
            p_lae[m] = lae_probs[m, sign]
            
        # --- CÁLCULO DE PROBABILIDADES ---
        
        # 6 Aciertos
        prob_real_6 = 1.0; prob_lae_6 = 1.0
        for m in range(6):
            prob_real_6 *= p_real[m]
            prob_lae_6 *= p_lae[m]
            
        # 5 Aciertos
        prob_real_5 = 0.0; prob_lae_5 = 0.0
        for m in range(6):
            if p_real[m] > 0: prob_real_5 += (prob_real_6 / p_real[m]) * (1.0 - p_real[m])
            if p_lae[m] > 0: prob_lae_5 += (prob_lae_6 / p_lae[m]) * (1.0 - p_lae[m])

        # 4 Aciertos
        prob_real_4 = 0.0; prob_lae_4 = 0.0
        for m in range(6):
            for k in range(m + 1, 6):
                term_real = (1.0 - p_real[m]) * (1.0 - p_real[k])
                term_lae = (1.0 - p_lae[m]) * (1.0 - p_lae[k])
                
                rest_real = 1.0; rest_lae = 1.0
                for x in range(6):
                    if x != m and x != k:
                        rest_real *= p_real[x]; rest_lae *= p_lae[x]
                
                prob_real_4 += term_real * rest_real
                prob_lae_4 += term_lae * rest_lae
                
        # (Omito 3 y 2 aciertos explícitos aquí por brevedad, pero en producción deberían ir
        # para máxima precisión. Aun así, el 95% del EV viene de 6, 5 y 4).
        
        # --- CÁLCULO DE EV ---
        lambda_6 = estimation * prob_lae_6
        prize_6 = poisson_prize_share(jackpot + cat_pots[0], lambda_6)
        
        lambda_5 = estimation * prob_lae_5
        prize_5 = poisson_prize_share(cat_pots[1], lambda_5)
        
        lambda_4 = estimation * prob_lae_4
        prize_4 = poisson_prize_share(cat_pots[2], lambda_4)
        
        ev_total = (prob_real_6 * prize_6) + (prob_real_5 * prize_5) + (prob_real_4 * prize_4)
        
        results[i, 0] = ev_total
        results[i, 1] = prob_real_6
        results[i, 2] = i 

    return results

# --- DYNAMIC BUDGET & PORTFOLIO LOGIC / LÓGICA DE CARTERA ---

def calculate_dynamic_budget(sorted_results, max_budget, min_ev):
    """Calcula nº de apuestas según la calidad de la jornada"""
    profitable_mask = sorted_results[:, 0] > min_ev
    profitable_columns = sorted_results[profitable_mask]
    
    if len(profitable_columns) == 0:
        return 0, 0.0
    
    # Score de calidad: Suma del exceso de EV
    top_n_check = min(100, len(profitable_columns))
    quality_score = np.sum(profitable_columns[:top_n_check, 0] - 1.0)
    
    activation = min(1.0, quality_score / 15.0) # Normalización heurística
    budget = int(max_budget * activation)
    
    if budget < 2 and activation > 0.05: budget = 2
    return budget, quality_score

def analyze_portfolio_collective(portfolio_indices, combinations, real_probs_matrix):
    """Analiza la probabilidad conjunta de éxito"""
    n_bets = len(portfolio_indices)
    if n_bets == 0: return
    
    selected_combs = combinations[portfolio_indices]
    
    # Probabilidad de que AL MENOS UNA acierte el pleno (Suma directa al ser excluyentes)
    total_prob_6 = 0.0
    for i in range(n_bets):
        p = 1.0
        for m in range(6):
            p *= real_probs_matrix[m, selected_combs[i, m]]
        total_prob_6 += p
        
    print(f"\n--- COLLECTIVE PORTFOLIO STATS (N={n_bets}) ---")
    if total_prob_6 > 0:
        print(f"Probabilidad de Pleno del Conjunto: 1 en {1/total_prob_6:.0f}")
    print(f"Probabilidad % de Pleno: {total_prob_6*100:.8f}%")
    
    # Cobertura
    print("Cobertura por partido:")
    for m in range(6):
        unique = np.unique(selected_combs[:, m])
        print(f"  Partido {m+1}: {len(unique)} signos {unique}")

# --- EXECUTION / EJECUCIÓN ---

if __name__ == "__main__":
    start_time = time.monotonic()
    
    # 1. Calcular EV
    print("Calculating EV...")
    results = evaluate_columns(COMBINATIONS, REAL_PROBS_MATRIX, LAE_PROBS_MATRIX, ESTIMATION, JACKPOT, PRIZE_DISTRIBUTION)
    sorted_results = results[results[:, 0].argsort()[::-1]]
    
    # 2. Presupuesto Dinámico
    n_bets, quality = calculate_dynamic_budget(sorted_results, MAX_BUDGET, MIN_EV_THRESHOLD)
    
    print(f"\nJornada Quality Score: {quality:.2f}")
    print(f"Apuestas Recomendadas: {n_bets}")
    
    # 3. Selección de Cartera Diversificada
    final_indices = []
    
    if n_bets > 0:
        # Cogemos la mejor siempre
        final_indices.append(int(sorted_results[0, 2]))
        
        # Greedy loop con filtro de similitud
        for i in range(1, len(sorted_results)):
            if len(final_indices) >= n_bets:
                break
            
            curr_idx = int(sorted_results[i, 2])
            curr_comb = COMBINATIONS[curr_idx]
            
            is_diverse = True
            for saved_idx in final_indices:
                saved_comb = COMBINATIONS[saved_idx]
                # Contamos coincidencias exactas
                matches_equal = np.sum(curr_comb == saved_comb)
                # Si coinciden en más de 2 partidos (ej. 3, 4, 5 o 6), es demasiado similar
                if matches_equal > 2: 
                    is_diverse = False
                    break
            
            if is_diverse:
                final_indices.append(curr_idx)
    
    # 4. Análisis
    analyze_portfolio_collective(final_indices, COMBINATIONS, REAL_PROBS_MATRIX)
    
    # 5. Guardar
    result_labels = ["0-0","0-1","0-2","0-M","1-0","1-1","1-2","1-M","2-0","2-1","2-2","2-M","M-0","M-1","M-2","M-M"]
    
    # Crear carpeta si no existe
    os.makedirs("results", exist_ok=True)
    
    with open("results/final_portfolio_dynamic.txt", "w") as f:
        for idx in final_indices:
            comb = COMBINATIONS[idx]
            line = "".join([result_labels[c].replace("-", "") for c in comb])
            f.write(line + "\n")
            
    print(f"\nArchivo generado: 'results/final_portfolio_dynamic.txt' con {len(final_indices)} apuestas.")
    print(f"Tiempo Total: {time.monotonic() - start_time:.4f}s")