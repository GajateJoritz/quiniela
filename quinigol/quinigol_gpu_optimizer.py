import time
import numpy as np
import os
import sys
import importlib.util

# --- GPU SETUP ---
try:
    import src.core_math_gpu as engine_gpu
    import src.core_math as engine_cpu 
    import cupy as cp
    
    dev = cp.cuda.Device()
    print(f"🚀 GPU DETECTED: NVIDIA Device #{dev.id} (Ready for Heavy Lifting)")
    
except ImportError as e:
    print("❌ ERROR: Could not import GPU modules.")
    sys.exit(1)

# --- CONFIGURATION ---
JACKPOT = 0.0        
ESTIMATION = 0.0     
BET_PRICE = 1.0           
PRIZE_DISTRIBUTION = np.array([0.10, 0.09, 0.08, 0.08, 0.20]) 

# --- SETTINGS ---
PORTFOLIO_SIZE = 10     
N_SIMULATIONS = 200000  
MIN_EV_THRESHOLD = 1.30   
MAX_CANDIDATES_SAFETY = 20000000

# 1 = "PROB_PROFIT", 3 = "RECOVER_50", 4 = "PRIZE>PORT_SIZE"
OPTIMIZATION_MODE = 1

# --- 1. LOAD DATA ---
LAE_PROBS_MATRIX = None 

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import quinigol.data.current_data as current_data
    
    print("✅ 'current_data.py' found & loaded.")
    JACKPOT = current_data.JACKPOT
    ESTIMATION = current_data.ESTIMATION
    LAE_PROBS_MATRIX = current_data.LAE_PROBS_MATRIX
    print(f"   Jackpot: {JACKPOT}€ | Estimation: {ESTIMATION}€")
    
except ImportError:
    print("⚠️ 'current_data.py' not found. Using defaults.")

# --- 2. ODDS SETUP ---
real_probs_1 = [5.50,5.50,9.50,19.00,7.00,6.00,11.00,23.00,15.00,15.00,21.00,41.00,41.00,36.00,56.00,86.00]
real_probs_2 = [9.00,6.50,8.50,10.00,11.00,6.50,9.00,10.00,21.00,15.00,15.00,19.00,41.00,34.00,36.00,36.00]
real_probs_3 = [9.50,8.50,13.00,21.00,8.50,6.00,10.00,17.00,13.00,10.00,13.00,23.00,23.00,17.00,21.00,34.00]
real_probs_4 = [8.50,10.00,17.00,34.00,7.00,6.00,13.00,23.00,10.00,9.50,15.00,29.00,15.00,13.00,23.00,36.00]
real_probs_5 = [9.00,8.00,11.00,17.00,9.50,6.00,9.50,15.00,15.00,12.00,13.00,21.00,34.00,23.00,26.00,34.00]
real_probs_6 = [7.00,9.50,21.00,56.00,5.50,6.50,15.00,41.00,8.00,9.00,19.00,46.00,12.00,13.00,26.00,56.00]

raw_real_probs = [real_probs_1, real_probs_2, real_probs_3, real_probs_4, real_probs_5, real_probs_6]
REAL_PROBS_MATRIX = np.zeros((6, 16), dtype=np.float64)

for i in range(6):
    row = np.array(raw_real_probs[i], dtype=np.float64)
    inv_row = 1.0 / row
    REAL_PROBS_MATRIX[i, :] = inv_row / np.sum(inv_row)

# --- COMBINATIONS ---
npy_path = "combinations/kinigmotza.npy"
try:
    COMBINATIONS = np.load(npy_path).astype(np.int32)
    print(f"✅ Loaded {len(COMBINATIONS):,} combinations from {npy_path}")
except:
    COMBINATIONS = np.random.randint(0, 16, (50000, 6)).astype(np.int32)

# --- EXECUTION ---

if __name__ == "__main__":
    start_total = time.time()
    
    print(f"\n--- ⚡ QUINIGOL GPU OPTIMIZER ---")
    print(f"Sims: {N_SIMULATIONS:,} | EV > {MIN_EV_THRESHOLD}")
    
    # 1. PRE-FILTERING (CPU)
    print("\n1. [CPU] Filtering candidates via Analytical EV...")
    evs = engine_cpu.get_top_candidates(
        COMBINATIONS, REAL_PROBS_MATRIX, LAE_PROBS_MATRIX, 
        ESTIMATION, JACKPOT, PRIZE_DISTRIBUTION
    )
    
    mask_good = evs > MIN_EV_THRESHOLD
    count_good = np.sum(mask_good)
    print(f"   Found {count_good} columns with EV > {MIN_EV_THRESHOLD}")
    
    if count_good == 0:
        print("⚠️ No columns > Threshold. Lowering to 1.0.")
        profitable_indices = np.where(evs > 1.0)[0]
    else:
        profitable_indices = np.where(mask_good)[0]
        
    # --- FUNNELING ---
    if len(profitable_indices) > MAX_CANDIDATES_SAFETY:
        print(f"   Limiting pool to Top {MAX_CANDIDATES_SAFETY} (Sort by EV)")
        sorted_local = np.argsort(evs[profitable_indices])[::-1]
        top_indices = profitable_indices[sorted_local[:MAX_CANDIDATES_SAFETY]]
    else:
        top_indices = profitable_indices
            
    # 2. GPU SIMULATION
    print(f"2. [GPU] Generating {N_SIMULATIONS:,} scenarios...")
    
    scenarios_gpu = engine_gpu.generate_scenarios_gpu(REAL_PROBS_MATRIX, N_SIMULATIONS)
    dynamic_prizes_gpu = engine_gpu.precompute_scenario_prizes_gpu(
        scenarios_gpu, LAE_PROBS_MATRIX, ESTIMATION, JACKPOT, PRIZE_DISTRIBUTION
    )

    # 3. GPU OPTIMIZATION
    print(f"3. [GPU] Optimizing Portfolio...")
    
    selected_local_indices, final_earnings = engine_gpu.greedy_portfolio_selection_gpu(
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
    print(f" 📊 PORTFOLIO REPORT")
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
    
    with open("results/resulQuinigolGPU.txt", "w") as f:
        for i, idx in enumerate(final_global_indices):
            comb = COMBINATIONS[idx]
            txt = " | ".join([result_labels[c] for c in comb])
            line = "".join([result_labels[c].replace("-", "") for c in comb])
            print(f"Bet #{i+1}: {txt}")
            f.write(line + "\n")
            
    print(f"\nTotal Time: {time.time() - start_total:.2f}s")