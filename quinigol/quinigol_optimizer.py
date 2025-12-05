import time
import numpy as np
import os
import sys
import src.core_math as engine

# --- CONFIGURATION / CONFIGURACIÓN ---
# Valores por defecto (se sobrescriben si existe current_data.py)
JACKPOT = 0.0        
ESTIMATION = 100000.0     
BET_PRICE = 1.0           
PRIZE_DISTRIBUTION = np.array([0.10, 0.09, 0.08, 0.08, 0.20]) 

# --- HIGH PRECISION SETTINGS / AJUSTES DE ALTA PRECISIÓN ---
PORTFOLIO_SIZE = 5       
N_SIMULATIONS = 1000000   

# --- DYNAMIC CANDIDATE SELECTION / SELECCIÓN DINÁMICA ---
MIN_EV_THRESHOLD = 1.60   
MAX_CANDIDATES_SAFETY = 50000

# --- 1. DATA LOADING (AUTOMATED SOURCE) / CARGA AUTOMÁTICA ---
LAE_PROBS_MATRIX = None # Inicializamos vacío

# Elige tu objetivo:
# 1 = "PROB_PROFIT": Maximiza la probabilidad de recuperar la inversión (muchos premios pequeños).
# 2 = "SORTINO": Busca rentabilidad penalizando solo las pérdidas (Equilibrado).
OPTIMIZATION_MODE = 1

# Intentamos cargar desde el archivo generado por el scraper
try:
    # Añadimos el directorio raíz al path por si current_data.py está arriba
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import quinigol.data.current_data as current_data
    
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


def greedy_portfolio_selection(candidate_indices, combinations, scenarios, dynamic_prizes, estimation, jackpot, prize_dist, target_size, mode):
    n_sims = scenarios.shape[0]
    n_cands = len(candidate_indices)
    selected_local_indices = []
    
    # Configuración de Premios (Referencia para valores fijos si hiciera falta, pero usamos dynamic)
    prizes_value = np.zeros(7) # Dummy array para compatibilidad si hiciera falta
    
    current_portfolio_earnings = np.zeros(n_sims, dtype=np.float64)
    
    mode_names = {1: "PROBABILITY OF PROFIT", 2: "SORTINO RATIO (Organic Diversification)"}
    print(f"   > Strategy: {mode_names.get(mode, 'Unknown')}")
    print(f"   > Starting Greedy Selection Loop ({target_size} steps)...")
    
    try:
        for step in range(target_size):
            t_step = time.time()
            best_candidate = -1
            best_metric = -float('inf')
            
            # Coste acumulado actual (para calcular el downside risk correctamente)
            current_cost = float(step)
            
            # Iterar sobre candidatos
            for i, c_idx in enumerate(candidate_indices):
                # Check si ya está seleccionado
                if c_idx in [candidate_indices[x] for x in selected_local_indices]: 
                    continue
                
                # Obtener combinación
                cand_comb = combinations[c_idx]
                
                # --- CALCULATION ON THE FLY ---
                # Pasamos la combinación y los escenarios. Numba hace el cruce.
                metric = engine.calculate_candidate_metric(cand_comb, scenarios, current_portfolio_earnings, dynamic_prizes, prizes_value, mode, current_cost)
                
                if metric > best_metric:
                    best_metric = metric
                    best_candidate = i # Guardamos el índice local de la lista de candidatos
            
            if best_candidate != -1:
                # Diagnóstico diversidad
                curr_comb = combinations[candidate_indices[best_candidate]]
                max_match = 0
                if len(selected_local_indices) > 0:
                    for sel_local in selected_local_indices:
                        prev = combinations[candidate_indices[sel_local]]
                        m = np.sum(curr_comb == prev)
                        if m > max_match: max_match = m
                    diff_msg = f"(Overlap: {max_match}/6)"
                else: diff_msg = "(Base)"

                selected_local_indices.append(best_candidate)
                
                # Actualizar ganancias permanentemente
                engine.update_earnings_on_the_fly(current_portfolio_earnings, curr_comb, scenarios, dynamic_prizes)
                
                print(f"     [Step {step+1}] Selected #{best_candidate} (Metric: {best_metric:.4f}) {diff_msg} [{time.time()-t_step:.2f}s]")
            else:
                print("\n     No more valid candidates.")
                break
    except KeyboardInterrupt:
        print("\n\n🛑 STOPPED BY USER (CTRL+C). Saving current portfolio...")
            
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
    evs = engine.get_top_candidates(COMBINATIONS, REAL_PROBS_MATRIX, LAE_PROBS_MATRIX, ESTIMATION, JACKPOT, PRIZE_DISTRIBUTION)
    
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
    scenarios = engine.generate_scenarios(REAL_PROBS_MATRIX, N_SIMULATIONS)
    dynamic_prizes = engine.precompute_scenario_prizes(scenarios, LAE_PROBS_MATRIX, ESTIMATION, JACKPOT, PRIZE_DISTRIBUTION)

    # 3. Optimización (Sin matriz intermedia)
    print(f"3. Optimizing Portfolio")
    sel_indices_local, final_earnings = greedy_portfolio_selection(
        top_indices, 
        COMBINATIONS, 
        scenarios, # Pasamos escenarios raw
        dynamic_prizes,
        ESTIMATION, JACKPOT, PRIZE_DISTRIBUTION, 
        PORTFOLIO_SIZE, OPTIMIZATION_MODE
    )
    
    final_global_indices = top_indices[sel_indices_local]
    
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