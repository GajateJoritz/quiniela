import cupy as cp
import numpy as np
import time

# --- UTILS ---

def to_cpu(gpu_array):
    """Helper to move data back to CPU safely."""
    if hasattr(gpu_array, 'get'):
        return gpu_array.get()
    return gpu_array

# --- 1. SCENARIO GENERATION (GPU) ---

def generate_scenarios_gpu(probs_matrix_cpu, n_sims):
    # Move probs to GPU
    probs_matrix = cp.asarray(probs_matrix_cpu, dtype=cp.float32)
    
    # Generate random numbers
    rand_vals = cp.random.random((6, n_sims), dtype=cp.float32)
    cdf = cp.cumsum(probs_matrix, axis=1)
    
    # Pre-allocate scenarios
    scenarios = cp.empty((n_sims, 6), dtype=cp.int8)
    
    for m in range(6):
        scenarios[:, m] = cp.searchsorted(cdf[m], rand_vals[m]).astype(cp.int8)
        
    cp.clip(scenarios, 0, 15, out=scenarios)
    
    # OPTIMIZATION: Return Transposed version (6, N_Sims) for faster memory access later
    # OPTIMIZACIÓN: Devolver versión transpuesta (6, N_Sims) para acceso a memoria más rápido después
    # This aligns memory for the 6-loop iteration in greedy
    return cp.ascontiguousarray(scenarios.T)

# --- 2. PRIZE CALCULATION (GPU) ---

def precompute_scenario_prizes_gpu(scenarios_T_gpu, lae_probs_cpu, estimation, jackpot, dist):
    """
    Calculates dynamic prizes. 
    Input scenarios_T_gpu is expected to be (6, N_Sims).
    """
    n_sims = scenarios_T_gpu.shape[1]
    lae_probs = cp.asarray(lae_probs_cpu, dtype=cp.float32)
    
    # 1. EXTRACT PROBABILITIES
    # We iterate matches because scenarios are transposed
    p_matrix = cp.empty((6, n_sims), dtype=cp.float32)
    
    for m in range(6):
        # scenarios_T_gpu[m] is already contiguous -> Fast
        p_matrix[m] = lae_probs[m, scenarios_T_gpu[m]]
        
    q_matrix = 1.0 - p_matrix

    # 2. COMBINATORICS (Transposed Logic)
    
    # P6 (All hits)
    prob_6 = cp.prod(p_matrix, axis=0) # Reduce along matches
    
    # P5 (1 Miss)
    prob_5 = cp.zeros(n_sims, dtype=cp.float32)
    for i in range(6):
        term = q_matrix[i].copy()
        for j in range(6):
            if i != j: term *= p_matrix[j]
        prob_5 += term

    # P4 (2 Misses)
    prob_4 = cp.zeros(n_sims, dtype=cp.float32)
    for i in range(6):
        for j in range(i + 1, 6):
            term = q_matrix[i] * q_matrix[j]
            for k in range(6):
                if k != i and k != j: term *= p_matrix[k]
            prob_4 += term

    # P3 (3 Misses)
    prob_3 = cp.zeros(n_sims, dtype=cp.float32)
    for i in range(6):
        for j in range(i + 1, 6):
            for k in range(j + 1, 6):
                term = q_matrix[i] * q_matrix[j] * q_matrix[k]
                for l in range(6):
                    if l != i and l != j and l != k: term *= p_matrix[l]
                prob_3 += term

    # P2 (4 Misses)
    prob_2 = cp.zeros(n_sims, dtype=cp.float32)
    for i in range(6):
        for j in range(i + 1, 6):
            term = p_matrix[i] * p_matrix[j]
            for k in range(6):
                if k != i and k != j: term *= q_matrix[k]
            prob_2 += term

    # 3. POISSON SHARES
    dynamic_prizes = cp.zeros((n_sims, 7), dtype=cp.float32)
    pots = cp.array([
        jackpot + (estimation * dist[0]), 
        estimation * dist[1], estimation * dist[2], 
        estimation * dist[3], estimation * dist[4]
    ], dtype=cp.float32)

    def calc_share(pot_idx, prob_array):
        lambda_val = estimation * prob_array
        mask_nz = lambda_val > 1e-9
        share = cp.ones(n_sims, dtype=cp.float32)
        l_valid = lambda_val[mask_nz]
        share[mask_nz] = (1.0 - cp.exp(-l_valid)) / l_valid
        gross = pots[pot_idx] * share
        net = cp.where(gross > 40000, (gross - 40000) * 0.8 + 40000, gross)
        return net

    dynamic_prizes[:, 6] = calc_share(0, prob_6)
    dynamic_prizes[:, 5] = calc_share(1, prob_5)
    dynamic_prizes[:, 4] = calc_share(2, prob_4)
    dynamic_prizes[:, 3] = calc_share(3, prob_3)
    dynamic_prizes[:, 2] = calc_share(4, prob_2)
    
    return dynamic_prizes

# --- 3. GREEDY OPTIMIZER (GPU - MEMORY OPTIMIZED) ---

def greedy_portfolio_selection_gpu(candidate_indices, all_combinations_cpu, scenarios_T_gpu, dynamic_prizes_gpu, target_size, mode):
    """
    GPU Accelerated Greedy Selection.
    Optimized for MEMORY EFFICIENCY (Avoids 3D Broadcasting).
    """
    # scenarios_T_gpu is (6, N_Sims)
    n_sims = scenarios_T_gpu.shape[1]
    
    print(f"   🚀 GPU OPTIMIZER: Processing {len(candidate_indices)} candidates vs {n_sims} scenarios...")
    
    # Move candidates to GPU (Batch, 6)
    candidates_pool_gpu = cp.asarray(all_combinations_cpu[candidate_indices], dtype=cp.int8)
    n_cands = candidates_pool_gpu.shape[0]
    
    # Earnings Accumulator
    current_earnings = cp.zeros(n_sims, dtype=cp.float32)
    
    selected_indices_local = []
    selected_mask = cp.zeros(n_cands, dtype=cp.bool_) 
    
    # --- MEMORY SAFE BATCHING ---
    # We now calculate hits match-by-match, so we don't need the massive 3D array.
    # We can process more candidates at once safely.
    # Target 500MB VRAM per batch is extremely safe.
    # 500MB / (N_Sims * 9 bytes) approx.
    
    BYTES_PER_SIM = 12 # Hits(1) + Prizes(4) + Earnings(4) + Misc overhead
    SAFE_MEMORY_LIMIT = 1.0 * 1024**3  # 1 GB (Equilibrio perfecto)
    
    calculated_batch = int(SAFE_MEMORY_LIMIT / (n_sims * BYTES_PER_SIM))
    # Cap at 5000 to maintain responsiveness
    BATCH_SIZE = max(50, min(calculated_batch, 5000))
    
    print(f"   ⚖️  Optimized Batch Size: {BATCH_SIZE} (Low Memory Footprint)")
    
    mode_names = {1: "PROBABILITY OF PROFIT", 3: "RECOVER 50%"}
    print(f"   > Strategy: {mode_names.get(mode, 'Unknown')}")
    print(f"   > Starting Greedy Selection Loop ({target_size} steps)...")
    print(f"   > Press CTRL+C to stop early.\n")

    # Pre-allocate indices
    sim_range = cp.arange(n_sims)

    try:
        for step in range(target_size):
            t_step = time.time()
            
            best_metric = -float('inf')
            best_cand_idx = -1
            
            cost_so_far = float(step)
            new_cost = cost_so_far + 1.0
            threshold_recover = new_cost * 0.5
            
            # --- BATCH LOOP ---
            for i in range(0, n_cands, BATCH_SIZE):
                end = min(i + BATCH_SIZE, n_cands)
                
                # Skip optimization
                if cp.all(selected_mask[i:end]):
                    continue
                
                # (Batch, 6)
                batch_cands = candidates_pool_gpu[i:end]
                current_bs = batch_cands.shape[0]
                
                # 1. CALCULATE HITS (MEMORY OPTIMIZED)
                # Instead of creating (Batch, Sims, 6), we loop over matches and add up.
                # Creates only (Batch, Sims) -> 6x less memory usage!
                
                hits_batch = cp.zeros((current_bs, n_sims), dtype=cp.int8)
                
                for m in range(6):
                    # Compare Match 'm' for all candidates in batch vs all sims
                    # batch_cands[:, m] -> (Batch,) -> (Batch, 1)
                    # scenarios_T_gpu[m] -> (Sims,) -> (1, Sims)
                    # Result (Batch, Sims) added to accumulator
                    hits_batch += (batch_cands[:, m:m+1] == scenarios_T_gpu[m:m+1])
                
                # 2. VECTORIZED PRIZE LOOKUP
                # Lookup prizes: (Batch, Sims)
                prizes_batch = dynamic_prizes_gpu[sim_range[None, :], hits_batch]
                
                # 3. VECTORIZED METRICS
                # Broadcast addition: (1, Sims) + (Batch, Sims)
                total_earnings_batch = current_earnings[None, :] + prizes_batch
                
                # Metric Function
                if mode == 1: # Prob Profit
                    # Count where total > cost, sum rows, divide by n_sims
                    metrics_batch = cp.mean(total_earnings_batch > new_cost, axis=1)
                elif mode == 3: # Recover 50%
                    metrics_batch = cp.mean(total_earnings_batch >= threshold_recover, axis=1)
                else:
                    metrics_batch = cp.zeros(current_bs)

                # 4. MASKING & SELECTION
                # Apply mask to the batch part
                local_mask = selected_mask[i:end]
                metrics_batch[local_mask] = -1.0
                
                # Find max in this batch
                local_max = cp.max(metrics_batch)
                if local_max > best_metric:
                    best_metric = local_max
                    local_argmax = cp.argmax(metrics_batch)
                    best_cand_idx = i + int(local_argmax)

            # --- UPDATE STATE ---
            if best_cand_idx != -1:
                selected_indices_local.append(best_cand_idx)
                selected_mask[best_cand_idx] = True
                
                # Recalculate hits for selected one (fast single op)
                best_comb = candidates_pool_gpu[best_cand_idx]
                
                # Since scenarios are transposed, we can't use simple == broadcast easily
                # We do loop over matches or transpose back logic
                b_hits = cp.zeros(n_sims, dtype=cp.int8)
                for m in range(6):
                    b_hits += (best_comb[m] == scenarios_T_gpu[m])
                
                # Update Earnings
                current_earnings += dynamic_prizes_gpu[sim_range, b_hits]
                
                # Display Metrics
                prob_profit_disp = (cp.count_nonzero(current_earnings > new_cost) / n_sims) * 100
                prob_any_disp = (cp.count_nonzero(current_earnings > 0) / n_sims) * 100
                
                # Overlap
                max_match = 0
                if len(selected_indices_local) > 1:
                    prev_indices = cp.array(selected_indices_local[:-1], dtype=cp.int32)
                    prev_combs = candidates_pool_gpu[prev_indices]
                    matches = (prev_combs == best_comb)
                    overlaps = cp.sum(matches, axis=1)
                    max_match = int(cp.max(overlaps))
                    
                diff_msg = f"Ovlp:{max_match}" if len(selected_indices_local) > 1 else "Base"
                print(f"     [Step {step+1}] #{best_cand_idx:<5} | Profit: {prob_profit_disp:5.2f}% | Any: {prob_any_disp:5.2f}% | {diff_msg} [{time.time()-t_step:.2f}s]")
                
            else:
                print("   ⚠️ No improvement.")
                break
                
    except KeyboardInterrupt:
        print("\n\n🛑 STOPPED BY USER (CTRL+C).")
            
    return selected_indices_local, to_cpu(current_earnings)