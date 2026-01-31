import numpy as np
from numba import njit, prange

# --- CONSTANTES ---
# Indices en el array de premios
IDX_15 = 0
IDX_14 = 1
IDX_13 = 2
IDX_12 = 3
IDX_11 = 4
IDX_10 = 5

@njit(fastmath=True)
def calc_taxed_prize(gross):
    """Aplica impuesto del 20% a premios > 40.000"""
    if gross > 40000.0:
        return (gross - 40000.0) * 0.8 + 40000.0
    return gross

@njit(fastmath=True)
def poisson_prize_share(pot, lambda_val):
    """Calcula la parte proporcional del bote según estimación de acertantes (Poisson)"""
    if lambda_val < 1e-9:
        return calc_taxed_prize(pot)
    share = (1.0 - np.exp(-lambda_val)) / lambda_val
    return calc_taxed_prize(pot * share)

@njit(parallel=True, fastmath=True)
def get_top_candidates_quiniela(combs_1x2, combs_pleno, probs_real_1x2, probs_real_pleno, probs_lae_1x2, probs_lae_pleno, est, jack, dist):
    """
    Calcula el EV Teórico EXACTO sumando todas las categorías (10 a 15).
    Optimizado dividiendo el producto total para evitar bucles internos redundantes.
    """
    n = combs_1x2.shape[0]
    evs = np.zeros(n, dtype=np.float64)
    
    # Pre-calcular botes
    # dist: [Cat15, Cat14, Cat13, Cat12, Cat11, Cat10]
    pot15 = jack + (est * dist[0])
    pots = np.zeros(6, dtype=np.float64)
    pots[0] = pot15
    for k in range(1, 6):
        pots[k] = est * dist[k]
        
    for i in prange(n):
        # Arrays locales para la columna actual
        p_real = np.zeros(14, dtype=np.float64)
        p_lae = np.zeros(14, dtype=np.float64)
        
        # Ratios (1-p)/p para cálculo rápido de fallos
        # Evitamos división por cero con un epsilon pequeño
        ratio_real = np.zeros(14, dtype=np.float64)
        ratio_lae = np.zeros(14, dtype=np.float64)
        
        epsilon = 1e-12
        
        # 1. Cargar probabilidades base y calcular P(14) inicial
        pr_14 = 1.0
        pl_14 = 1.0
        
        for m in range(14):
            s = combs_1x2[i, m]
            # Probabilidades
            val_r = probs_real_1x2[m, s]
            val_l = probs_lae_1x2[m, s]
            
            p_real[m] = val_r
            p_lae[m] = val_l
            
            pr_14 *= val_r
            pl_14 *= val_l
            
            # Precalcular ratio para sustituir acierto por fallo
            # ratio = P(Fallo) / P(Acierto) = (1-p)/p
            ratio_real[m] = (1.0 - val_r) / (val_r + epsilon)
            ratio_lae[m]  = (1.0 - val_l) / (val_l + epsilon)

        # Probabilidades Pleno
        idx_p = combs_pleno[i]
        pr_pleno = probs_real_pleno[idx_p]
        pl_pleno = probs_lae_pleno[idx_p]
        
        # --- CÁLCULO PROBABILIDADES DERIVADAS (OPTIMIZADO) ---
        
        # P(13): Suma de cambiar 1 acierto por 1 fallo
        pr_13 = 0.0
        pl_13 = 0.0
        for m in range(14):
            pr_13 += pr_14 * ratio_real[m]
            pl_13 += pl_14 * ratio_lae[m]
            
        # P(12): Suma de cambiar 2 aciertos por 2 fallos
        pr_12 = 0.0
        pl_12 = 0.0
        for m in range(14):
            for k in range(m + 1, 14):
                pr_12 += pr_14 * ratio_real[m] * ratio_real[k]
                pl_12 += pl_14 * ratio_lae[m] * ratio_lae[k]
                
        # P(11): Suma de cambiar 3 aciertos por 3 fallos
        pr_11 = 0.0
        pl_11 = 0.0
        for m in range(14):
            for k in range(m + 1, 14):
                term_r_mk = pr_14 * ratio_real[m] * ratio_real[k]
                term_l_mk = pl_14 * ratio_lae[m] * ratio_lae[k]
                for j in range(k + 1, 14):
                    pr_11 += term_r_mk * ratio_real[j]
                    pl_11 += term_l_mk * ratio_lae[j]
                    
        # P(10): Suma de cambiar 4 aciertos por 4 fallos
        pr_10 = 0.0
        pl_10 = 0.0
        for m in range(14):
            for k in range(m + 1, 14):
                term_r_mk = pr_14 * ratio_real[m] * ratio_real[k]
                term_l_mk = pl_14 * ratio_lae[m] * ratio_lae[k]
                for j in range(k + 1, 14):
                    term_r_mkj = term_r_mk * ratio_real[j]
                    term_l_mkj = term_l_mk * ratio_lae[j]
                    for x in range(j + 1, 14):
                        pr_10 += term_r_mkj * ratio_real[x]
                        pl_10 += term_l_mkj * ratio_lae[x]
        
        # --- CÁLCULO DE EV ---
        
        # Cat 15 (14 aciertos + Pleno)
        # Prob ganar: pr_14 * pr_pleno
        # Lambda (acertantes esperados): est * (pl_14 * pl_pleno)
        val_15 = (pr_14 * pr_pleno) * poisson_prize_share(pots[0], est * pl_14 * pl_pleno)
        
        # Cat 14 (14 aciertos base)
        val_14 = pr_14 * poisson_prize_share(pots[1], est * pl_14)
        
        # Cat 13
        val_13 = pr_13 * poisson_prize_share(pots[2], est * pl_13)
        
        # Cat 12
        val_12 = pr_12 * poisson_prize_share(pots[3], est * pl_12)
        
        # Cat 11
        val_11 = pr_11 * poisson_prize_share(pots[4], est * pl_11)
        
        # Cat 10
        val_10 = pr_10 * poisson_prize_share(pots[5], est * pl_10)
        
        # EV TOTAL
        evs[i] = val_15 + val_14 + val_13 + val_12 + val_11 + val_10

    return evs

@njit(parallel=True)
def generate_scenarios_quiniela(probs_1x2, probs_pleno, n_sims):
    """
    Genera escenarios simulados basados en Probabilidades REALES.
    """
    scenarios = np.zeros((n_sims, 15), dtype=np.int8)
    
    # 1. Bloque 1X2 (14 Partidos)
    for m in range(14):
        cdf = np.cumsum(probs_1x2[m])
        # Aseguramos que el último valor es >= 1.0 para evitar errores numéricos
        cdf[2] = max(cdf[2], 1.0)
        
        for i in prange(n_sims):
            r = np.random.random()
            if r < cdf[0]: scenarios[i, m] = 0
            elif r < cdf[1]: scenarios[i, m] = 1
            else: scenarios[i, m] = 2
            
    # 2. Pleno (1 Partido, 16 resultados)
    cdf_p = np.cumsum(probs_pleno)
    cdf_p[15] = max(cdf_p[15], 1.0)
    
    for i in prange(n_sims):
        r = np.random.random()
        idx = 0
        # Búsqueda lineal simple es suficiente para 16 elementos
        while idx < 15 and r > cdf_p[idx]:
            idx += 1
        scenarios[i, 14] = idx
        
    return scenarios

@njit(parallel=True, fastmath=True)
def precompute_scenario_prizes_quiniela(scenarios, probs_lae_1x2, probs_lae_pleno, est, jack, dist):
    """
    Calcula los premios dinámicos para cada escenario simulado.
    Calcula EXACTAMENTE cuánta gente acertaría 14, 13, 12, 11 y 10 en ese escenario
    usando las probabilidades apostadas (LAE).
    """
    n_sims = scenarios.shape[0]
    prizes = np.zeros((n_sims, 6), dtype=np.float32)
    
    # Botes fijos base
    pot15 = jack + (est * dist[0])
    pots = np.zeros(6, dtype=np.float32)
    pots[0] = pot15
    for k in range(1, 6):
        pots[k] = est * dist[k]
        
    epsilon = 1e-12
    
    for i in prange(n_sims):
        # Para el escenario 'i', extraemos las probs LAE de los signos que salieron
        pl_scen = np.zeros(14, dtype=np.float32)
        ratio_scen = np.zeros(14, dtype=np.float32)
        
        # Prob Base (P14)
        prob_14 = 1.0
        
        for m in range(14):
            res = scenarios[i, m]
            val = probs_lae_1x2[m, res]
            pl_scen[m] = val
            prob_14 *= val
            
            # Ratio (1-p)/p para combinatoria rápida
            ratio_scen[m] = (1.0 - val) / (val + epsilon)
            
        res_pleno = scenarios[i, 14]
        prob_pleno = probs_lae_pleno[res_pleno]
        
        # --- CALCULO PROBS AGREGADAS (Misma lógica que get_top_candidates pero sobre el escenario) ---
        
        # Prob 15
        prob_15 = prob_14 * prob_pleno
        
        # Prob 13
        prob_13 = 0.0
        for m in range(14):
            prob_13 += prob_14 * ratio_scen[m]
            
        # Prob 12
        prob_12 = 0.0
        for m in range(14):
            for k in range(m + 1, 14):
                prob_12 += prob_14 * ratio_scen[m] * ratio_scen[k]
                
        # Prob 11 (Completa)
        prob_11 = 0.0
        for m in range(14):
            for k in range(m + 1, 14):
                term = prob_14 * ratio_scen[m] * ratio_scen[k]
                for j in range(k + 1, 14):
                    prob_11 += term * ratio_scen[j]
                    
        # Prob 10 (Completa)
        prob_10 = 0.0
        for m in range(14):
            for k in range(m + 1, 14):
                term_mk = prob_14 * ratio_scen[m] * ratio_scen[k]
                for j in range(k + 1, 14):
                    term_mkj = term_mk * ratio_scen[j]
                    for x in range(j + 1, 14):
                        prob_10 += term_mkj * ratio_scen[x]
                        
        # --- ASIGNACIÓN DE PREMIOS ---
        
        # 15 Aciertos
        prizes[i, 0] = poisson_prize_share(pots[0], est * prob_15)
        # 14 Aciertos
        prizes[i, 1] = poisson_prize_share(pots[1], est * prob_14)
        # 13 Aciertos
        prizes[i, 2] = poisson_prize_share(pots[2], est * prob_13)
        # 12 Aciertos
        prizes[i, 3] = poisson_prize_share(pots[3], est * prob_12)
        
        # 11 Aciertos (Regla < 1€)
        prize_11 = poisson_prize_share(pots[4], est * prob_11)
        if prize_11 < 1.0: prize_11 = 0.0
        prizes[i, 4] = prize_11
        
        # 10 Aciertos (Regla < 1€)
        prize_10 = poisson_prize_share(pots[5], est * prob_10)
        if prize_10 < 1.0: prize_10 = 0.0
        prizes[i, 5] = prize_10
        
    return prizes

@njit(parallel=True, fastmath=True)
def calculate_candidate_metric_quiniela(cand_1x2, cand_pleno, scenarios, current_earnings, dynamic_prizes, mode, cost_so_far):
    """
    Calcula la métrica (Profit o Sortino) añadiendo una columna candidata a la cartera existente.
    """
    n = len(current_earnings)
    temp_earnings = np.zeros(n, dtype=np.float64)
    bet_cost = 0.75
    new_cost = cost_so_far + bet_cost
    
    for i in prange(n):
        # 1. Contar aciertos 1X2
        hits = 0
        for m in range(14):
            if cand_1x2[m] == scenarios[i, m]:
                hits += 1
        
        val = current_earnings[i]
        
        # 2. Sumar premio correspondiente
        if hits == 14:
            val += dynamic_prizes[i, IDX_14]
            # Si acierta 14, miramos el pleno
            if cand_pleno == scenarios[i, 14]:
                val += dynamic_prizes[i, IDX_15]
        elif hits == 13:
            val += dynamic_prizes[i, IDX_13]
        elif hits == 12:
            val += dynamic_prizes[i, IDX_12]
        elif hits == 11:
            val += dynamic_prizes[i, IDX_11]
        elif hits == 10:
            val += dynamic_prizes[i, IDX_10]
            
        temp_earnings[i] = val
        
    # --- MÉTRICAS ---
    
    if mode == 1: # Probabilidad de Beneficio
        wins = 0.0
        for i in prange(n):
            if temp_earnings[i] > new_cost:
                wins += 1.0
        return wins / n
        
    elif mode == 2: # Ratio Sortino
        sum_val = 0.0
        sum_sq_down = 0.0
        
        for i in prange(n):
            val = temp_earnings[i]
            sum_val += val
            # Downside risk: solo sumamos si perdemos dinero
            if val < new_cost:
                diff = new_cost - val
                sum_sq_down += diff * diff
        
        mean_profit = (sum_val / n) - new_cost
        
        if sum_sq_down < 1e-9:
            return 9999999.0 # Riesgo cero
            
        downside_dev = np.sqrt(sum_sq_down / n)
        return mean_profit / downside_dev
    
    return 0.0

@njit(parallel=True)
def update_earnings_quiniela(current_earnings, cand_1x2, cand_pleno, scenarios, dynamic_prizes):
    """
    Actualiza el array de ganancias acumuladas tras confirmar una selección.
    """
    n = len(current_earnings)
    for i in prange(n):
        hits = 0
        for m in range(14):
            if cand_1x2[m] == scenarios[i, m]:
                hits += 1
        
        if hits == 14:
            current_earnings[i] += dynamic_prizes[i, IDX_14]
            if cand_pleno == scenarios[i, 14]:
                current_earnings[i] += dynamic_prizes[i, IDX_15]
        elif hits == 13:
            current_earnings[i] += dynamic_prizes[i, IDX_13]
        elif hits == 12:
            current_earnings[i] += dynamic_prizes[i, IDX_12]
        elif hits == 11:
            current_earnings[i] += dynamic_prizes[i, IDX_11]
        elif hits == 10:
            current_earnings[i] += dynamic_prizes[i, IDX_10]