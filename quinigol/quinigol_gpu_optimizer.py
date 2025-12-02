import time
import numpy as np
import os
import sys

# Intentamos importar CuPy. Si no tienes tarjeta NVIDIA configurada, avisará.
try:
    import cupy as cp
    print(f"✅ CuPy detectado. Usando GPU: {cp.cuda.runtime.getDeviceCount()} dispositivo(s)")
    # Liberamos memoria por si acaso
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
except ImportError:
    print("❌ ERROR CRÍTICO: CuPy no está instalado.")
    print("Ejecuta: pip install cupy-cuda12x")
    sys.exit(1)

# --- CONFIGURATION / CONFIGURACIÓN ---
JACKPOT = 255000.0        
ESTIMATION = 100000.0     
BET_PRICE = 1.0           
PRIZE_DISTRIBUTION = np.array([0.10, 0.09, 0.08, 0.08, 0.20]) 

# --- GPU HIGH PRECISION SETTINGS ---
PORTFOLIO_SIZE = 10       
N_SIMULATIONS = 10000000  # ¡SUBIMOS A 10 MILLONES! (Tu GPU puede con ello)
CANDIDATE_POOL = 1000     # Analizamos las 1000 mejores (Tu GPU se come esto fácil)

# --- DATA LOADING (MANUAL) ---
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

LAE_PROBS_MATRIX = REAL_PROBS_MATRIX.copy() 
try:
    with open("quinigol/laequinig.txt", mode="r") as f:
        pass # Tu lógica de carga aquí
except: pass

try:
    COMBINATIONS = np.load("combinations/kinigmotza.npy").astype(np.int32)
except:
    print("Generando datos aleatorios para test...")
    COMBINATIONS = np.random.randint(0, 16, (50000, 6)).astype(np.int32)

# --- CPU PRE-FILTER (Keep Numba/Numpy for the lightweight initial filter) ---
# Usamos CPU para esto porque es rápido y filtrar 50k filas no requiere GPU.

def get_analytical_ev_cpu(combs, real_probs, lae_probs, est, jack, dist):
    # Cálculo vectorial simple en CPU con Numpy
    n = combs.shape[0]
    p_real6 = np.ones(n)
    p_lae6 = np.ones(n)
    
    for m in range(6):
        # Indexing avanzado de numpy
        col_signs = combs[:, m]
        p_real6 *= real_probs[m, col_signs]
        p_lae6 *= lae_probs[m, col_signs]
    
    # Approx EV basado en pleno
    lambda6 = est * p_lae6
    # Share: (1 - exp(-L)) / L
    # Evitar div por 0
    with np.errstate(divide='ignore', invalid='ignore'):
        share = (1.0 - np.exp(-lambda6)) / lambda6
        share[lambda6 < 1e-9] = 1.0
        
    pot_gross = jack + (est * dist[0])
    prize = pot_gross * share
    # Tax simple
    prize = np.where(prize > 40000, (prize-40000)*0.8 + 40000, prize)
    
    return p_real6 * prize

# --- GPU CORE LOGIC ---

def run_gpu_optimization(candidates_cpu, real_probs_cpu, est, jack, prize_dist, target_size):
    """
    Todo el proceso pesado ocurre aquí dentro, en la VRAM de la 3060 Ti.
    """
    t_start = time.time()
    
    # 1. MOVER DATOS A LA GPU (Host to Device)
    print("   [GPU] Moving data to VRAM...")
    # Probabilidades reales
    real_probs_gpu = cp.asarray(real_probs_cpu)
    # Candidatos (500 o 1000 filas)
    candidates_gpu = cp.asarray(candidates_cpu)
    
    # 2. GENERAR ESCENARIOS (Simulación Masiva)
    print(f"   [GPU] Generating {N_SIMULATIONS:,} scenarios...")
    # Matriz gigante: 10 Millones x 6 (int8) = 60 MB (Muy poco para la GPU)
    scenarios_gpu = cp.zeros((N_SIMULATIONS, 6), dtype=cp.int8)
    
    for m in range(6):
        # Probabilidades de este partido
        probs = real_probs_gpu[m]
        # Generamos aleatorios vectorizados
        # cp.random.choice es muy rápido
        scenarios_gpu[:, m] = cp.random.choice(16, size=N_SIMULATIONS, p=probs).astype(cp.int8)
        
    # 3. CONSTRUIR MATRIZ DE ACIERTOS (El cuello de botella de la CPU)
    # Matriz: 10M x 1000 (int8) = ~10 GB ?? CUIDADO.
    # 10M * 1000 bytes = 10.000 MB = 10GB.
    # TU GPU TIENE 8GB. ESTO VA A FALLAR SI LO HACEMOS DE GOLPE.
    # SOLUCIÓN: No guardamos la matriz gigante. Calculamos "on the fly" o por lotes.
    
    # Vamos a calcular los "earnings" (ganancias) directamente para ahorrar memoria.
    # Matriz de ganancias acumuladas (float32 ocupa 4 bytes).
    # 10M * 4 bytes = 40 MB. Esto sí cabe.
    current_portfolio_earnings = cp.zeros(N_SIMULATIONS, dtype=cp.float32)
    
    selected_indices_local = []
    
    # Pre-cálculo de premios (escalares)
    prizes = np.zeros(7)
    prizes[6] = (jack + est * prize_dist[0]) # Tax se aplicará después si se quiere, simplificado
    prizes[5] = (est * prize_dist[1] / 5.0)
    prizes[4] = (est * prize_dist[2] / 50.0)
    prizes[3] = (est * prize_dist[3] / 500.0)
    
    # Convertir a GPU
    p_val_6 = float(prizes[6])
    p_val_5 = float(prizes[5])
    p_val_4 = float(prizes[4])
    p_val_3 = float(prizes[3])
    
    print(f"   [GPU] Optimizing Portfolio (Target: {target_size})...")
    
    for step in range(target_size):
        best_cand_idx = -1
        best_sharpe = -1.0
        best_cand_earnings = None
        
        # Iteramos sobre candidatos (1000 iteraciones es nada para la GPU)
        for c in range(candidates_gpu.shape[0]):
            if c in selected_indices_local:
                continue
            
            # --- COMPARACIÓN GPU VECTORIZADA ---
            # Comparamos UN candidato contra los 10 Millones de escenarios
            # Broadcasting: (10M, 6) == (6,) -> (10M, 6) bool
            # Sumamos aciertos: (10M,) int8
            
            cand_vector = candidates_gpu[c]
            matches = (scenarios_gpu == cand_vector)
            hits = cp.sum(matches, axis=1, dtype=cp.int8) # Resultado: Vector de 10M de enteros
            
            # Calculamos ganancias de este candidato
            # Usamos float32 para ahorrar memoria y velocidad en GPU
            c_earnings = cp.zeros(N_SIMULATIONS, dtype=cp.float32)
            
            # Asignación condicional vectorizada (Kernel rápido)
            c_earnings[hits == 6] = p_val_6
            c_earnings[hits == 5] = p_val_5
            c_earnings[hits == 4] = p_val_4
            c_earnings[hits == 3] = p_val_3
            
            # Sumar a lo que ya tenemos
            new_total = current_portfolio_earnings + c_earnings
            
            # Calcular métricas (Reducción paralela)
            mean_profit = cp.mean(new_total)
            std_dev = cp.std(new_total)
            
            if std_dev == 0: sharpe = 0.0
            else: sharpe = mean_profit / std_dev
            
            # --- FILTRO DIVERSIDAD ---
            # Lo hacemos en CPU (es pequeño) o GPU. 
            # Al ser pocos seleccionados, GPU no aporta mucho aquí, pero lo hacemos por consistencia.
            is_similar = False
            for sel in selected_indices_local:
                sel_vector = candidates_gpu[sel]
                # Contar coincidencias entre el candidato actual y los ya elegidos
                equal_signs = cp.sum(cand_vector == sel_vector)
                if equal_signs >= 5: # Si son iguales en 5 o 6 partidos
                    is_similar = True
                    break
            
            if is_similar:
                continue
            
            # Guardar el mejor
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_cand_idx = c
                best_cand_earnings = c_earnings

        if best_cand_idx != -1:
            selected_indices_local.append(best_cand_idx)
            current_portfolio_earnings += best_cand_earnings
            # Traer el dato escalar a CPU para imprimir
            metric_cpu = float(best_sharpe)
            print(f"     Step {step+1}: Added Candidate #{best_cand_idx} (Sharpe: {metric_cpu:.4f})")
        else:
            break
            
    # Traer resultados finales a CPU
    final_earnings_cpu = cp.asnumpy(current_portfolio_earnings)
    
    # Liberar memoria VRAM
    del scenarios_gpu
    del candidates_gpu
    del current_portfolio_earnings
    cp.get_default_memory_pool().free_all_blocks()
    
    return selected_indices_local, final_earnings_cpu

# --- EXECUTION ---

if __name__ == "__main__":
    start_total = time.time()
    print(f"--- QUINIGOL GPU OPTIMIZER (RTX ENABLED) ---")
    print(f"Simulations: {N_SIMULATIONS:,} | Candidates: {CANDIDATE_POOL}")
    
    # 1. Filtro Analítico (CPU)
    print("1. [CPU] Filtering top candidates...")
    evs = get_analytical_ev_cpu(COMBINATIONS, REAL_PROBS_MATRIX, LAE_PROBS_MATRIX, ESTIMATION, JACKPOT, PRIZE_DISTRIBUTION)
    top_indices = evs.argsort()[::-1][:CANDIDATE_POOL]
    top_combinations = COMBINATIONS[top_indices]
    
    # 2. Optimización GPU
    print("2. [GPU] Starting Monte Carlo Simulation...")
    sel_local_indices, final_earnings = run_gpu_optimization(
        top_combinations, REAL_PROBS_MATRIX, ESTIMATION, JACKPOT, PRIZE_DISTRIBUTION, PORTFOLIO_SIZE
    )
    
    final_global_indices = top_indices[sel_local_indices]
    
    # 3. Resultados
    print("\n" + "="*40)
    print(f"   GPU REPORT")
    print("="*40)
    
    wins_profit = np.mean(final_earnings > len(final_global_indices)) * 100
    mean_profit = np.mean(final_earnings)
    
    print(f"Ganancia Media: {mean_profit:.2f} €")
    print(f"Probabilidad Rentabilidad: {wins_profit:.2f}%")
    
    print("\n--- BETS ---")
    labels = ["0-0","0-1","0-2","0-M","1-0","1-1","1-2","1-M","2-0","2-1","2-2","2-M","M-0","M-1","M-2","M-M"]
    
    os.makedirs("results", exist_ok=True)
    with open("results/gpu_portfolio.txt", "w") as f:
        for idx in final_global_indices:
            comb = COMBINATIONS[idx]
            txt = " | ".join([labels[c] for c in comb])
            line = "".join([labels[c].replace("-", "") for c in comb])
            print(txt)
            f.write(line + "\n")
            
    print(f"\n🚀 Total Time: {time.time() - start_total:.2f}s")