import cupy as cp
import numpy as np
import time
from numba import njit, prange

# ==========================================
#  PARTE 1: CPU FILTRADO (NUMBA)
# ==========================================

@njit(fastmath=True)
def calc_taxed_prize(gross):
    if gross > 40000.0: return (gross - 40000.0) * 0.8 + 40000.0
    return gross

@njit(fastmath=True)
def poisson_prize_share(pot, lambda_val):
    if lambda_val < 1e-9: return calc_taxed_prize(pot)
    share = (1.0 - np.exp(-lambda_val)) / lambda_val
    return calc_taxed_prize(pot * share)

@njit(parallel=True, fastmath=True)
def get_top_candidates_cpu_5(combs, real_probs, lae_probs, est, jack, dist):
    """
    Calcula EV Analítico para 5 PARTIDOS.
    Categorías: 5 (Pleno), 4, 3, 2.
    """
    n = combs.shape[0]
    evs = np.zeros(n, dtype=np.float64)
    
    # dist = [cat5_share, cat4_share, cat3_share, cat2_share]
    cat_pots = est * dist 
    
    pot5 = jack + cat_pots[0]
    pot4 = cat_pots[1]
    pot3 = cat_pots[2]
    pot2 = cat_pots[3]
    
    for i in prange(n):
        p_real = np.zeros(5, dtype=np.float64)
        p_lae = np.zeros(5, dtype=np.float64)
        
        for m in range(5):
            s = combs[i, m]
            p_real[m] = real_probs[m, s]
            p_lae[m] = lae_probs[m, s]
            
        # --- COMBINATORIA 5 ELEMENTOS ---

        # 5 ACIERTOS (PLENO) - 1 caso
        prob_real_5 = 1.0
        prob_lae_5 = 1.0
        for m in range(5):
            prob_real_5 *= p_real[m]
            prob_lae_5 *= p_lae[m]
            
        # 4 ACIERTOS (1 FALLO) - 5 casos (5C1)
        prob_real_4 = 0.0
        prob_lae_4 = 0.0
        for m in range(5):
            # Falla m, acierta el resto
            term_real = (1.0 - p_real[m])
            term_lae = (1.0 - p_lae[m])
            for x in range(5):
                if x != m:
                    term_real *= p_real[x]
                    term_lae *= p_lae[x]
            prob_real_4 += term_real
            prob_lae_4 += term_lae

        # 3 ACIERTOS (2 FALLOS) - 10 casos (5C2)
        prob_real_3 = 0.0
        prob_lae_3 = 0.0
        for m in range(5):
            for k in range(m + 1, 5):
                # Fallan m y k
                term_real = (1.0 - p_real[m]) * (1.0 - p_real[k])
                term_lae = (1.0 - p_lae[m]) * (1.0 - p_lae[k])
                for x in range(5):
                    if x != m and x != k:
                        term_real *= p_real[x]
                        term_lae *= p_lae[x]
                prob_real_3 += term_real
                prob_lae_3 += term_lae

        # 2 ACIERTOS (3 FALLOS / ACIERTA 2) - 10 casos (5C2 pero lógica inversa)
        prob_real_2 = 0.0
        prob_lae_2 = 0.0
        for m in range(5):
            for k in range(m + 1, 5):
                # Acierta m y k, falla resto
                term_real = p_real[m] * p_real[k]
                term_lae = p_lae[m] * p_lae[k]
                for x in range(5):
                    if x != m and x != k:
                        term_real *= (1.0 - p_real[x])
                        term_lae *= (1.0 - p_lae[x])
                prob_real_2 += term_real
                prob_lae_2 += term_lae

        # --- EV ---
        l5 = est * prob_lae_5
        ev5 = prob_real_5 * poisson_prize_share(pot5, l5)
        
        l4 = est * prob_lae_4
        ev4 = prob_real_4 * poisson_prize_share(pot4, l4)
        
        l3 = est * prob_lae_3
        ev3 = prob_real_3 * poisson_prize_share(pot3, l3)
        
        l2 = est * prob_lae_2
        ev2 = prob_real_2 * poisson_prize_share(pot2, l2)
        
        evs[i] = ev5 + ev4 + ev3 + ev2
        
    return evs

# ==========================================
#  PARTE 2: GPU SIMULACION (CUPY)
# ==========================================

def to_cpu(gpu_array):
    if hasattr(gpu_array, 'get'):
        return gpu_array.get()
    return gpu_array

def generate_scenarios_gpu_5(probs_matrix_cpu, n_sims):
    """Genera escenarios para 5 partidos"""
    probs_matrix = cp.asarray(probs_matrix_cpu, dtype=cp.float32)
    
    rand_vals = cp.random.random((5, n_sims), dtype=cp.float32)
    cdf = cp.cumsum(probs_matrix, axis=1)
    
    scenarios = cp.empty((n_sims, 5), dtype=cp.int8)
    
    for m in range(5):
        scenarios[:, m] = cp.searchsorted(cdf[m], rand_vals[m]).astype(cp.int8)
        
    cp.clip(scenarios, 0, 15, out=scenarios)
    return cp.ascontiguousarray(scenarios.T) # (5, N_Sims)

def precompute_scenario_prizes_gpu_5(scenarios_T_gpu, lae_probs_cpu, estimation, jackpot, dist):
    """
    Calcula premios dinámicos para escenarios de 5 partidos.
    Return shape: (N_Sims, 6) -> Indices 5,4,3,2 usados.
    """
    n_sims = scenarios_T_gpu.shape[1]
    lae_probs = cp.asarray(lae_probs_cpu, dtype=cp.float32)
    
    p_matrix = cp.empty((5, n_sims), dtype=cp.float32)
    
    for m in range(5):
        p_matrix[m] = lae_probs[m, scenarios_T_gpu[m]]
        
    q_matrix = 1.0 - p_matrix

    # --- COMBINATORIA GPU (5 Partidos) ---
    
    # 5 Aciertos
    prob_5 = cp.prod(p_matrix, axis=0)
    
    # 4 Aciertos (1 Fallo)
    prob_4 = cp.zeros(n_sims, dtype=cp.float32)
    for i in range(5):
        term = q_matrix[i].copy()
        for j in range(5):
            if i != j: term *= p_matrix[j]
        prob_4 += term

    # 3 Aciertos (2 Fallos)
    prob_3 = cp.zeros(n_sims, dtype=cp.float32)
    for i in range(5):
        for j in range(i + 1, 5):
            term = q_matrix[i] * q_matrix[j]
            for k in range(5):
                if k != i and k != j: term *= p_matrix[k]
            prob_3 += term

    # 2 Aciertos (3 Fallos / Acierta 2)
    prob_2 = cp.zeros(n_sims, dtype=cp.float32)
    for i in range(5):
        for j in range(i + 1, 5):
            term = p_matrix[i] * p_matrix[j]
            for k in range(5):
                if k != i and k != j: term *= q_matrix[k]
            prob_2 += term

    # --- POISSON SHARES ---
    # Usamos indices 5,4,3,2. El indice 0 y 1 quedan vacios.
    dynamic_prizes = cp.zeros((n_sims, 6), dtype=cp.float32)
    
    pots = cp.array([
        jackpot + (estimation * dist[0]), # Cat 5 hits
        estimation * dist[1],             # Cat 4 hits
        estimation * dist[2],             # Cat 3 hits
        estimation * dist[3]              # Cat 2 hits
    ], dtype=cp.float32)

    def calc_share(pot_val, prob_array):
        lambda_val = estimation * prob_array
        mask_nz = lambda_val > 1e-9
        share = cp.ones(n_sims, dtype=cp.float32)
        l_valid = lambda_val[mask_nz]
        share[mask_nz] = (1.0 - cp.exp(-l_valid)) / l_valid
        gross = pot_val * share
        net = cp.where(gross > 40000, (gross - 40000) * 0.8 + 40000, gross)
        return net

    dynamic_prizes[:, 5] = calc_share(pots[0], prob_5)
    dynamic_prizes[:, 4] = calc_share(pots[1], prob_4)
    dynamic_prizes[:, 3] = calc_share(pots[2], prob_3)
    
    p2_values = calc_share(pots[3], prob_2)
    dynamic_prizes[:, 2] = cp.where(p2_values < 1.0, 0.0, p2_values)
    
    return dynamic_prizes

def greedy_portfolio_selection_gpu_5(candidate_indices, all_combinations_cpu, scenarios_T_gpu, dynamic_prizes_gpu, target_size, mode):
    """
    Optimizador Greedy para 5 partidos.
    CORREGIDO: Cálculo automático de batch size para evitar OutOfMemory.
    """
    n_sims = scenarios_T_gpu.shape[1]
    
    # Mover candidatos a GPU
    candidates_pool_gpu = cp.asarray(all_combinations_cpu[candidate_indices], dtype=cp.int8)
    n_cands = candidates_pool_gpu.shape[0]
    
    current_earnings = cp.zeros(n_sims, dtype=cp.float32)
    selected_indices_local = []
    selected_mask = cp.zeros(n_cands, dtype=cp.bool_) 
    
    # --- MEMORY SAFE BATCHING (AUTO-ADJUST) ---
    # Estimamos el consumo. La indexación vectorizada es costosa en RAM.
    # Con 200k sims, un batch grande explota la memoria.
    
    # Bytes por par (simulación * candidato) estimado (indices + float results)
    BYTES_PER_SIM_CAND = 32 
    # Queremos usar máx 500MB de RAM por tanda para ir sobrados
    SAFE_MEMORY_LIMIT = 0.5 * 1024**3  
    
    calculated_batch = int(SAFE_MEMORY_LIMIT / (n_sims * BYTES_PER_SIM_CAND))
    # Limites de seguridad: mínimo 10, máximo 2000
    BATCH_SIZE = max(10, min(calculated_batch, 2000))
    
    print(f"   ⚖️  Auto-adjusted Batch Size: {BATCH_SIZE} (Safe for {n_sims:,} sims)")
    
    # Usar int32 para índices ahorra memoria (vs int64 por defecto)
    sim_range = cp.arange(n_sims, dtype=cp.int32)

    for step in range(target_size):
        t_step = time.time()
        best_metric = -float('inf')
        best_cand_idx = -1
        
        cost_so_far = float(step)
        threshold = cost_so_far + 1.0
        
        for i in range(0, n_cands, BATCH_SIZE):
            end = min(i + BATCH_SIZE, n_cands)
            if cp.all(selected_mask[i:end]): continue
            
            batch_cands = candidates_pool_gpu[i:end]
            current_bs = batch_cands.shape[0]
            
            # --- CALCULO DE ACIERTOS (5 PARTIDOS) ---
            hits_batch = cp.zeros((current_bs, n_sims), dtype=cp.int8)
            for m in range(5):
                # Broadcasting cuidadoso
                hits_batch += (batch_cands[:, m:m+1] == scenarios_T_gpu[m:m+1])
            
            # Indexación avanzada (Aquí es donde explotaba la memoria con batch grande)
            prizes_batch = dynamic_prizes_gpu[sim_range[None, :], hits_batch]
            
            total_earnings_batch = current_earnings[None, :] + prizes_batch
            
            # Métrica (Prob Profit)
            # Usamos mean axis=1
            metrics_batch = cp.mean(total_earnings_batch > threshold, axis=1)
            
            # Enmascarar ya seleccionados del batch actual
            metrics_batch[selected_mask[i:end]] = -1.0
            
            local_max = cp.max(metrics_batch)
            if local_max > best_metric:
                best_metric = local_max
                best_cand_idx = i + int(cp.argmax(metrics_batch))

        if best_cand_idx != -1:
            selected_indices_local.append(best_cand_idx)
            selected_mask[best_cand_idx] = True
            
            best_comb = candidates_pool_gpu[best_cand_idx]
            b_hits = cp.zeros(n_sims, dtype=cp.int8)
            for m in range(5):
                b_hits += (best_comb[m] == scenarios_T_gpu[m])
            
            current_earnings += dynamic_prizes_gpu[sim_range, b_hits]
            
            prob_profit = (cp.count_nonzero(current_earnings > threshold) / n_sims) * 100
            print(f"     [Step {step+1}] Index {best_cand_idx} | Profit: {prob_profit:.2f}% | Time: {time.time()-t_step:.2f}s")
        else:
            break
            
    return selected_indices_local, to_cpu(current_earnings)