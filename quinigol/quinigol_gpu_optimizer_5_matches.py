import time
import numpy as np
import os
import sys

# --- GPU SETUP ---
try:
    import src.core_math_gpu_5 as engine_gpu  # IMPORTANTE: Usamos el nuevo motor _5
    import cupy as cp
    
    dev = cp.cuda.Device()
    print(f"🚀 GPU DETECTED: NVIDIA Device #{dev.id} (Ready for 5-MATCH Heavy Lifting)")
    
except ImportError as e:
    print(f"❌ ERROR: Could not import GPU modules: {e}")
    sys.exit(1)

# --- CONFIGURATION (5 PARTIDOS) ---
# Normativa LAE para 1 partido excluido (verificar siempre):
# Cat 1 (5 aciertos): 14%
# Cat 2 (4 aciertos): 9%
# Cat 3 (3 aciertos): 9%
# Cat 4 (2 aciertos): 23%
PRIZE_DISTRIBUTION = np.array([0.14, 0.09, 0.09, 0.23]) 

JACKPOT = 0.0        
ESTIMATION = 0.0     
BET_PRICE = 1.0           

# --- SETTINGS ---
PORTFOLIO_SIZE = 10    
N_SIMULATIONS = 1000000  
MIN_EV_THRESHOLD = 1.40   # Umbral ajustado para 5 partidos
MAX_CANDIDATES_SAFETY = 1500000

# Índice del partido a ignorar (0-5). Por defecto el último (5).
CANCELLED_MATCH_INDEX = 0

# 1 = "PROB_PROFIT", 3 = "RECOVER_50"
OPTIMIZATION_MODE = 1

# --- 1. LOAD DATA ---
LAE_PROBS_MATRIX = None 

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import quinigol.data.current_data as current_data
    
    print("✅ 'current_data.py' found & loaded.")
    JACKPOT = current_data.JACKPOT
    ESTIMATION = current_data.ESTIMATION
    
    # Cargar matriz completa y eliminar la fila del partido cancelado
    full_lae = current_data.LAE_PROBS_MATRIX
    LAE_PROBS_MATRIX = np.delete(full_lae, CANCELLED_MATCH_INDEX, axis=0)
    
    print(f"   Jackpot: {JACKPOT}€ | Estimation: {ESTIMATION}€")
    print(f"   Active Matches: {LAE_PROBS_MATRIX.shape[0]}")
    
except ImportError:
    print("⚠️ 'current_data.py' not found. Using defaults.")

# --- 2. ODDS SETUP (MANUAL O CARGADO) ---
# Si no usas current_data, define aquí tus probs reales (6 filas) y el script borrará una.
real_probs_1 = [29.00, 41.00, 67.00, 226.00, 6.70, 11.30, 7.80, 20.70, 14.20, 9.50, 33.20, 22.90, 28.30, 25.00, 22.90, 61.00]
real_probs_2 = [29.00, 41.00, 67.00, 226.00, 12.00, 17.00, 41.00, 131.00, 8.00, 10.00, 26.00, 76.00, 2.90, 3.75, 9.50, 26.00]
real_probs_3 = [4.75, 7.00, 17.00, 46.00, 4.75, 6.00, 17.00, 46.00, 9.00, 12.000, 26.00, 71.00, 19.00, 26.00, 56.00, 131.00]
real_probs_4 = [15.00, 17.00, 34.00, 96.00, 8.00, 9.00, 19.00, 41.00, 7.50, 8.50, 17.00, 41.00, 6.00, 6.50, 13.00, 29.00]
real_probs_5 = [17.00,9.00,8.50,7.00,17.00,8.50,8.50,7.00,29.00,15.00,15.00,12.00,61.00,36.00,34.00,21.00]
real_probs_6 = [13.00,13.00,23.00,51.00,8.00,8.00,15.00,34.00,8.50,8.50,15.00,29.00,8.00,8.00,13.00,26.00]

raw_real_probs = [real_probs_1, real_probs_2, real_probs_3, real_probs_4, real_probs_5, real_probs_6]
FULL_REAL_MATRIX = np.zeros((6, 16), dtype=np.float64)

for i in range(6):
    row = np.array(raw_real_probs[i], dtype=np.float64)
    inv_row = 1.0 / row
    FULL_REAL_MATRIX[i, :] = inv_row / np.sum(inv_row)

# ELIMINAR PARTIDO CANCELADO
REAL_PROBS_MATRIX = np.delete(FULL_REAL_MATRIX, CANCELLED_MATCH_INDEX, axis=0)

# --- COMBINATIONS ---
npy_path = "combinations/kinigmotza.npy" # Usa el mismo archivo de combinaciones
try:
    FULL_COMBINATIONS = np.load(npy_path).astype(np.int32)
    # Eliminar columna del partido cancelado
    COMBINATIONS = np.delete(FULL_COMBINATIONS, CANCELLED_MATCH_INDEX, axis=1)
    
    # Al eliminar una columna, habrá duplicados (filas idénticas). Hay que unificarlos.
    # Esto es pesado en CPU, si el archivo es gigante, cuidado.
    print(f"⚠️ Reducing combinations from 6 to 5 columns...")
    COMBINATIONS = np.unique(COMBINATIONS, axis=0)
    
    print(f"✅ Loaded {len(COMBINATIONS):,} unique 5-match combinations")
except Exception as e:
    print(f"⚠️ Error loading combinations: {e}")
    COMBINATIONS = np.random.randint(0, 16, (50000, 5)).astype(np.int32)

# --- EXECUTION ---

if __name__ == "__main__":
    start_total = time.time()
    
    print(f"\n--- ⚡ QUINIGOL GPU OPTIMIZER (5 MATCHES) ---")
    print(f"Sims: {N_SIMULATIONS:,} | EV > {MIN_EV_THRESHOLD}")
    
    # 1. PRE-FILTERING (CPU) - Usamos la función del módulo nuevo
    print("\n1. [CPU] Filtering candidates via Analytical EV (5 Matches)...")
    evs = engine_gpu.get_top_candidates_cpu_5(
        COMBINATIONS, REAL_PROBS_MATRIX, LAE_PROBS_MATRIX, 
        ESTIMATION, JACKPOT, PRIZE_DISTRIBUTION
    )
    
    mask_good = evs > MIN_EV_THRESHOLD
    count_good = np.sum(mask_good)
    print(f"   Found {count_good} columns with EV > {MIN_EV_THRESHOLD}")
    
    if count_good == 0:
        print("⚠️ No columns > Threshold. Using Top 50k by EV.")
        profitable_indices = np.argsort(evs)[::-1][:50000]
    else:
        profitable_indices = np.where(mask_good)[0]
        
    # --- FUNNELING ---
    if len(profitable_indices) > MAX_CANDIDATES_SAFETY:
        print(f"   Limiting pool to Top {MAX_CANDIDATES_SAFETY}")
        sorted_local = np.argsort(evs[profitable_indices])[::-1]
        top_indices = profitable_indices[sorted_local[:MAX_CANDIDATES_SAFETY]]
    else:
        top_indices = profitable_indices
            
    # 2. GPU SIMULATION
    print(f"2. [GPU] Generating {N_SIMULATIONS:,} scenarios (5 Matches)...")
    
    scenarios_gpu = engine_gpu.generate_scenarios_gpu_5(REAL_PROBS_MATRIX, N_SIMULATIONS)
    
    dynamic_prizes_gpu = engine_gpu.precompute_scenario_prizes_gpu_5(
        scenarios_gpu, LAE_PROBS_MATRIX, ESTIMATION, JACKPOT, PRIZE_DISTRIBUTION
    )

    # 3. GPU OPTIMIZATION
    print(f"3. [GPU] Optimizing Portfolio...")
    
    selected_local_indices, final_earnings = engine_gpu.greedy_portfolio_selection_gpu_5(
        top_indices, 
        COMBINATIONS, 
        scenarios_gpu, 
        dynamic_prizes_gpu,
        PORTFOLIO_SIZE, 
        OPTIMIZATION_MODE
    )
    
    final_global_indices = top_indices[selected_local_indices]
    
    # 4. REPORTING
    print("\n" + "="*50)
    print(f" 📊 PORTFOLIO REPORT (5 MATCHES)")
    print("="*50)
    
    cost = float(len(final_global_indices))
    simulated_ev = np.mean(final_earnings)
    prob_profit = np.mean(final_earnings > cost) * 100
    prob_any = np.mean(final_earnings > 0) * 100
    
    theoretical_total_ev = np.sum(evs[final_global_indices])
    
    print(f"Investment:      {cost} €")
    print(f"EV Theoretical:  {theoretical_total_ev:.4f} €")
    print(f"EV Simulated:    {simulated_ev:.4f} €")
    print("-" * 30)
    print(f"Prob. Profit:    {prob_profit:.2f}%")
    print(f"Prob. Any Prize: {prob_any:.2f}%")
    
    # Save
    print("\n--- SELECTED BETS ---")
    os.makedirs("results", exist_ok=True)
    result_labels = ["0-0","0-1","0-2","0-M","1-0","1-1","1-2","1-M","2-0","2-1","2-2","2-M","M-0","M-1","M-2","M-M"]
    
    with open("results/resulQuinigolGPU_5.txt", "w") as f:
        for i, idx in enumerate(final_global_indices):
            comb = COMBINATIONS[idx]
            txt = " | ".join([result_labels[c] for c in comb])
            line = "".join([result_labels[c].replace("-", "") for c in comb])
            print(f"Bet #{i+1}: {txt}")
            f.write(line + "\n")
            
    print(f"\nTotal Time: {time.time() - start_total:.2f}s")