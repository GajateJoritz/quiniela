import numpy as np
import time
import sys
import os
import importlib.util

# --- 1. SETUP & IMPORTS ---
try:
    import src.core_math as engine
    import quinigol_optimizer as optimizer
except ImportError:
    print("❌ ERROR: Could not find 'src.core_math' or 'quinigol_optimizer'.")
    print("Make sure you run this script from the project root.")
    sys.exit(1)

# --- 2. CONFIGURATION / CONFIGURACIÓN ---

# Path to the historical data file / Ruta al archivo de datos históricos
DATA_FILE_PATH = os.path.join("quinigol", "data", "historical", "jornada_01.py")

# Financial & Game Settings
BET_PRICE = 1.0           
PRIZE_DISTRIBUTION = np.array([0.10, 0.09, 0.08, 0.08, 0.20]) 

# --- STRATEGY SETTINGS / AJUSTES DE ESTRATEGIA ---
PORTFOLIO_SIZE = 10       
N_SIMULATIONS = 100000    

# EV Threshold / Umbral de EV
MIN_EV_THRESHOLD = 1.30   
MAX_CANDIDATES_SAFETY = 20000000 

OPTIMIZATION_MODE = 3     # 1 = Prob Profit (Conservative), 2 = Sortino

# --- 3. DYNAMIC DATA LOADING / CARGA DINÁMICA DE DATOS ---

print(f"📂 Loading historical data from: {DATA_FILE_PATH} ...")

if not os.path.exists(DATA_FILE_PATH):
    print(f"❌ ERROR: File not found: {DATA_FILE_PATH}")
    sys.exit(1)

# Import module dynamically / Importar módulo dinámicamente
spec = importlib.util.spec_from_file_location("current_round", DATA_FILE_PATH)
round_data = importlib.util.module_from_spec(spec)
sys.modules["current_round"] = round_data
spec.loader.exec_module(round_data)

print(f"✅ Data loaded for JORNADA {round_data.JORNADA_ID}")
print(f"   Jackpot: {round_data.JACKPOT} € | Estimation: {round_data.ESTIMATION} €")

# Load Combinations / Cargar Combinaciones
try:
    COMBINATIONS = np.load("combinations/kinigmotza.npy").astype(np.int32)
    print(f"✅ Combinations loaded: {len(COMBINATIONS)}")
except:
    print("⚠️  Warning: 'kinigmotza.npy' not found. Using random data for testing.")
    COMBINATIONS = np.random.randint(0, 16, (50000, 6)).astype(np.int32)


# --- 4. EXECUTION LOGIC / LÓGICA DE EJECUCIÓN ---

def run_test():
    start_total = time.time()
    print(f"\n--- OPTIMIZATION TEST (Round {round_data.JORNADA_ID}) ---")
    print(f"Sims: {N_SIMULATIONS} | Threshold: {MIN_EV_THRESHOLD} | Max Pool: {MAX_CANDIDATES_SAFETY}")

    # 1. Pre-filter via Analytical EV / Pre-filtro por EV Analítico
    print("1. Filtering top candidates via Analytical EV...")
    evs = engine.get_top_candidates(
        COMBINATIONS, 
        round_data.REAL_PROBS_MATRIX, 
        round_data.LAE_PROBS_MATRIX, 
        round_data.ESTIMATION, 
        round_data.JACKPOT, 
        PRIZE_DISTRIBUTION
    )

    # --- WINNER DIAGNOSTIC / DIAGNÓSTICO DEL GANADOR REAL ---
    # Check if the real winner would have passed our filter
    # Comprobar si el ganador real habría pasado nuestro filtro
    winning_comb = np.array(round_data.WINNING_COMBINATION)
    matches = np.all(COMBINATIONS == winning_comb, axis=1)
    winner_indices = np.where(matches)[0]
    
    print("\n🔍 WINNER DIAGNOSTIC:")
    LABELS = ["0-0","0-1","0-2","0-M","1-0","1-1","1-2","1-M","2-0","2-1","2-2","2-M","M-0","M-1","M-2","M-M"]
    win_str = " | ".join([LABELS[x] for x in winning_comb])
    print(f"   Real Winner: {win_str}")

    if len(winner_indices) > 0:
        w_idx = winner_indices[0]
        w_ev = evs[w_idx]
        rank = np.sum(evs > w_ev) + 1
        print(f"   ✅ Found in DB at index {w_idx}")
        print(f"   📊 Winner EV: {w_ev:.4f}")
        print(f"   🏆 Rank by EV: #{rank} (out of {len(COMBINATIONS)})")
        
        if w_ev < MIN_EV_THRESHOLD:
            print(f"   ⚠️ FATAL: Winner EV ({w_ev:.4f}) < Threshold ({MIN_EV_THRESHOLD}). Filtered out!")
        else:
            print(f"   ✅ Winner passes EV Threshold.")
    else:
        print("   ❌ Winner NOT FOUND in database.")
    print("-" * 40)

    # --- DYNAMIC SELECTION LOGIC ---
    mask_good = evs > MIN_EV_THRESHOLD
    count_good = np.sum(mask_good)
    
    print(f"   Found {count_good} columns with EV > {MIN_EV_THRESHOLD}")
    
    if count_good == 0:
        print("⚠️ No columns meet the EV threshold! Using Top 100 as fallback.")
        top_indices = evs.argsort()[::-1][:100]
    else:
        good_indices = np.where(mask_good)[0]
        
        # Safety Limit / Límite de Seguridad
        if len(good_indices) > MAX_CANDIDATES_SAFETY:
            print(f"   Limiting candidate pool to {MAX_CANDIDATES_SAFETY} for RAM safety.")
            sorted_good = good_indices[np.argsort(evs[good_indices])[::-1]]
            top_indices = sorted_good[:MAX_CANDIDATES_SAFETY]
        else:
            top_indices = good_indices
            
    print(f"   Candidate Pool Size: {len(top_indices)}")
    
    # 2. Generate Scenarios / Generación de Escenarios
    print(f"2. Generating {N_SIMULATIONS:,} Monte Carlo scenarios...")
    scenarios = engine.generate_scenarios(round_data.REAL_PROBS_MATRIX, N_SIMULATIONS)
    dynamic_prizes = engine.precompute_scenario_prizes(
        scenarios, 
        round_data.LAE_PROBS_MATRIX, 
        round_data.ESTIMATION, 
        round_data.JACKPOT, 
        PRIZE_DISTRIBUTION
    )
    
    # 3. Optimization / Optimización Greedy
    print(f"3. Selecting best {PORTFOLIO_SIZE} bets...")
    selected_local_indices, final_earnings = optimizer.greedy_portfolio_selection(
        top_indices, COMBINATIONS, scenarios, dynamic_prizes,
        round_data.ESTIMATION, round_data.JACKPOT, PRIZE_DISTRIBUTION, 
        PORTFOLIO_SIZE, OPTIMIZATION_MODE
    )
    
    final_indices = top_indices[selected_local_indices]
    final_bets = COMBINATIONS[final_indices]
    
    # --- 5. REAL WORLD CHECK / COMPROBACIÓN REAL ---
    print("\n" + "="*50)
    print(f" 🏁 REAL WORLD RESULTS (JORNADA {round_data.JORNADA_ID})")
    print("="*50)
    
    total_winnings = 0.0
    hits_breakdown = {6:0, 5:0, 4:0, 3:0, 2:0, 1:0, 0:0}
    
    print(f"🏆 Winner: {win_str}\n")
    print(f"{'#':<3} {'BET':<35} {'HITS':<10} {'PRIZE (€)':<10}")
    print("-" * 65)

    for i, bet in enumerate(final_bets):
        # Check matches / Comprobar aciertos
        hits = np.sum(bet == winning_comb)
        hits_breakdown[hits] += 1
        
        # Check prize / Comprobar premio
        prize = 0.0
        if hits >= 2:
            prize = round_data.REAL_PRIZES.get(hits, 0.0)
            total_winnings += prize
            
        bet_str = " ".join([LABELS[x] for x in bet])
        
        if hits >= 2 or i < 5:
            prefix = "✅" if hits >= 2 else "  "
            print(f"{prefix} {i+1:<2} {bet_str} {hits} ac.      {prize:.2f}")

    # --- FINAL REPORT & METRICS / REPORTE FINAL Y MÉTRICAS ---
    cost = len(final_bets)
    profit = total_winnings - cost
    roi = (profit / cost) * 100 if cost > 0 else 0
    
    theoretical_total_ev = np.sum(evs[final_indices])
    simulated_ev_per_round = np.mean(final_earnings)
    
    # --- RISK METRICS CALCULATION / CÁLCULO MÉTRICAS DE RIESGO ---
    # Probability of making profit (> cost)
    # Probabilidad de rentabilizar (> coste)
    prob_profit = np.mean(final_earnings > cost) * 100
    
    # Probability of winning anything (> 0)
    # Probabilidad de cobrar algo (> 0)
    prob_any_prize = np.mean(final_earnings > 0) * 100
    
    # Standard Deviation (Volatility)
    # Desviación Típica (Volatilidad)
    std_dev_earnings = np.std(final_earnings)

    print("-" * 65)
    print(f"📊 SUMMARY JORNADA {round_data.JORNADA_ID}:")
    print(f"   Investment:    {cost:.2f} €")
    print(f"   Real Winnings: {total_winnings:.2f} €")
    print(f"   NET PROFIT:    {profit:+.2f} €")
    print(f"   REAL ROI:      {roi:+.2f} %")
    print("-" * 30)
    print(f"   Expected EV (Theory): {theoretical_total_ev:.2f} €")
    print(f"   Expected EV (Sim):    {simulated_ev_per_round:.2f} €")
    print("-" * 30)
    print(f"🛡️  RISK / STABILITY METRICS (Simulation):")
    print(f"   Prob. Profit:       {prob_profit:.2f}% (Win > {cost}€)")
    print(f"   Prob. Any Prize:    {prob_any_prize:.2f}%")
    print(f"   Volatility (Std):   {std_dev_earnings:.2f} €")
    print("-" * 30)
    print(f"   Hits: 6({hits_breakdown[6]}) 5({hits_breakdown[5]}) 4({hits_breakdown[4]}) 3({hits_breakdown[3]}) 2({hits_breakdown[2]})")
    
    print(f"\nTotal Time: {time.time() - start_total:.2f}s")

if __name__ == "__main__":
    run_test()