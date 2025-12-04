import numpy as np
from numba import njit, prange
import sys

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
    
    pot6 = jack + cat_pots[0]
    
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


@njit(parallel=True, fastmath=True)
def precompute_scenario_prizes(scenarios, lae_probs, estimation, jackpot, dist):
    """
    Calcula cuánto pagaría cada categoría de premio en cada uno de los 5M de escenarios.
    Basado en la probabilidad LAE de que ese escenario ocurra.
    """
    n_sims = scenarios.shape[0]
    # Matriz: [Simulacion, Categoría] -> Valor del premio en Euros
    # Indices: 6=Pleno, 5=5ac, ... 
    dynamic_prizes = np.zeros((n_sims, 7), dtype=np.float32) # float32 ahorra RAM
    
    # Botes Totales (Pots)
    pot6 = jackpot + (estimation * dist[0])
    pot5 = estimation * dist[1]
    pot4 = estimation * dist[2]
    pot3 = estimation * dist[3]
    pot2 = estimation * dist[4]
    
    for i in prange(n_sims):
        # 1. Recuperar probabilidades LAE de LOS RESULTADOS QUE SALIERON en este escenario
        p_lae = np.zeros(6, dtype=np.float32)
        for m in range(6):
            res = scenarios[i, m]
            p_lae[m] = lae_probs[m, res]
            
        # 2. Calcular cuánta gente tendría X aciertos contra este escenario
        # (Es la misma lógica que get_top_candidates pero usando p_lae sobre sí mismo)
        
        # Probabilidad de que un apostante random tenga 6 aciertos (Coincida exacto)
        prob_6 = 1.0
        for m in range(6): prob_6 *= p_lae[m]
            
        # Probabilidad de que un apostante random tenga 5 aciertos
        prob_5 = 0.0
        for m in range(6):
            term = (1.0 - p_lae[m])
            for x in range(6):
                if x != m: term *= p_lae[x]
            prob_5 += term
            
        # Probabilidad de que un apostante random tenga 4 aciertos
        prob_4 = 0.0
        for m in range(6):
            for k in range(m + 1, 6):
                term = (1.0 - p_lae[m]) * (1.0 - p_lae[k])
                for x in range(6):
                    if x != m and x != k: term *= p_lae[x]
                prob_4 += term
        
        # Probabilidad de que un apostante random tenga 3 aciertos (Aprox rápida)
        # Para 3 y 2 usamos aproximación si queremos velocidad extrema, 
        # pero Numba aguanta el cálculo completo.
        prob_3 = 0.0
        for m in range(6):
            for k in range(m + 1, 6):
                for j in range(k + 1, 6):
                    term = (1.0 - p_lae[m]) * (1.0 - p_lae[k]) * (1.0 - p_lae[j])
                    for x in range(6):
                        if x != m and x != k and x != j: term *= p_lae[x]
                    prob_3 += term

        # Probabilidad de que un apostante random tenga 2 aciertos
        prob_2 = 0.0
        for m in range(6):
            for k in range(m + 1, 6):
                # Acierta m y k, falla el resto
                term = p_lae[m] * p_lae[k]
                for x in range(6):
                    if x != m and x != k: term *= (1.0 - p_lae[x])
                prob_2 += term

        # 3. Calcular Premios Dinámicos (Euros) usando Poisson Share
        # Lambda = Cuántos ganadores se esperan
        
        # Cat 6
        l6 = estimation * prob_6
        dynamic_prizes[i, 6] = poisson_prize_share(pot6, l6)
        
        # Cat 5
        l5 = estimation * prob_5
        dynamic_prizes[i, 5] = poisson_prize_share(pot5, l5)
        
        # Cat 4
        l4 = estimation * prob_4
        dynamic_prizes[i, 4] = poisson_prize_share(pot4, l4)
        
        # Cat 3
        l3 = estimation * prob_3
        dynamic_prizes[i, 3] = poisson_prize_share(pot3, l3)
        
        # Cat 2
        l2 = estimation * prob_2
        dynamic_prizes[i, 2] = poisson_prize_share(pot2, l2)
        
    return dynamic_prizes

@njit(fastmath=True)
def calculate_metric_numba(current_earnings, new_hits, dynamic_prizes, mode, cost_so_far):
    """
    Calcula la métrica de calidad de la cartera según el modo elegido.
    """
    n = len(current_earnings)
 
    # Variables acumuladoras
    sum_val = 0.0      # Para media (EV)
    sum_sq_down = 0.0  # Para Sortino (Riesgo de bajada)
    wins_count = 0.0   # Para Probabilidad de Beneficio
    
    # El coste aumenta al añadir una apuesta (Coste actual + 1)
    new_cost = cost_so_far + 1.0
    
    for i in range(n):
        val = current_earnings[i] # Ganancia actual en simulación i
        h = new_hits[i]           # Aciertos del nuevo candidato en simulación i

        # Sumamos premio DINÁMICO de esta simulación específica 'i'
        if h >= 2: # Solo si hay premio (2 a 6)
            val += dynamic_prizes[i, h]
        
        # --- CÁLCULOS SEGÚN MODO ---
        
        if mode == 1: # PROB_PROFIT
            # Contamos si en este escenario ganamos más de lo que gastamos
            if val > new_cost:
                wins_count += 1.0
                
        elif mode == 2: # SORTINO RATIO
            sum_val += val
            # Solo sumamos al riesgo si perdemos dinero (val < new_cost)
            # El riesgo es la desviación de las pérdidas al cuadrado
            if val < new_cost:
                diff = new_cost - val
                sum_sq_down += diff * diff
                
        elif mode == 3: # PROB_RECOVER_50 (Ganar > 50% coste)
            if val >= (new_cost * 0.5):
                wins_count += 1.0

    # --- RETORNO DE RESULTADOS ---
    
    if mode == 1: # Probabilidad de Rentabilizar
        return wins_count / n
        
    elif mode == 2: # Sortino Ratio
        mean_profit = (sum_val / n) - new_cost # Beneficio neto medio
        if sum_sq_down < 1e-9: 
            return 999999.0 
        downside_deviation = np.sqrt(sum_sq_down / n)
        return mean_profit / downside_deviation
    
    elif mode == 3: # Probabilidad de Recuperar 50%
        return wins_count / n

    return 0.0

@njit
def update_earnings_numba(current_earnings, new_hits, dynamic_prizes):
    """Actualiza el array de ganancias acumuladas (Incluye 2 aciertos)"""
    n = len(current_earnings)
    for i in range(n):
        h = new_hits[i]
        if h >= 2:
            current_earnings[i] += dynamic_prizes[i, h]

# --- ON-THE-FLY CALCULATION (ZERO RAM) ---
# Eliminamos precompute_hits_matrix para ahorrar 5GB de RAM.
# Ahora calculamos los aciertos DENTRO del bucle de selección.

@njit(parallel=True, fastmath=True)
def calculate_candidate_metric(candidate_comb, scenarios, current_earnings, dynamic_prizes, prizes, mode, cost_so_far):
    """
    Calcula la métrica de un candidato comparándolo al vuelo con los escenarios.
    NO usa memoria extra.
    """
    n = len(current_earnings)
    
    # Acumuladores (Thread-safe reduction es complejo en numba parallel puro para múltiples variables,
    # así que hacemos un truco: cada hilo calcula su trozo y devolvemos la suma al final no se puede facil.
    # En su lugar, usamos un bucle prange y guardamos en un array temporal muy pequeño si fuera necesario,
    # pero para Sortino necesitamos ver cada simulación.
    
    # ESTRATEGIA: Como necesitamos sumar val^2 y val, y estos dependen de if/else,
    # no podemos usar reducciones simples. Haremos el bucle paralelo pero escribiendo en un array temporal
    # de tamaño n_sims (earnings temporales) y luego reduciendo.
    
    # 1. Calcular nuevas ganancias totales (Cartera + Candidato)
    temp_earnings = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        # Calcular aciertos on-the-fly
        h = 0
        for m in range(6):
            if candidate_comb[m] == scenarios[i, m]: h += 1
            
        val = current_earnings[i]
        
        # Sumar premio dinámico
        if h >= 2: val += dynamic_prizes[i, h]
        
        temp_earnings[i] = val

    # 2. Reducción (Calcular métricas sobre el array resultante)
    # Esto es rápido.
    
    new_cost = cost_so_far + 1.0
    
    if mode == 1: # PROB_PROFIT
        wins = 0.0
        for i in prange(n):
            if temp_earnings[i] > new_cost: wins += 1.0
        return wins / n
        
    elif mode == 2: # SORTINO
        sum_val = 0.0
        sum_sq_down = 0.0
        
        for i in prange(n):
            val = temp_earnings[i]
            sum_val += val
            if val < new_cost:
                diff = new_cost - val
                sum_sq_down += diff * diff
                
        mean_profit = (sum_val / n) - new_cost
        if sum_sq_down < 1e-9: return 999999.0
        return mean_profit / np.sqrt(sum_sq_down / n)
        
    elif mode == 3: # PROB_RECOVER_50
        wins = 0.0
        threshold = new_cost * 0.5
        for i in prange(n):
            if temp_earnings[i] >= threshold: wins += 1.0
        return wins / n
        
    return 0.0

@njit(parallel=True)
def update_earnings_on_the_fly(current_earnings, best_comb, scenarios, dynamic_prizes):
    n = len(current_earnings)
    for i in prange(n):
        h = 0
        for m in range(6):
            if best_comb[m] == scenarios[i, m]: h += 1
        
        if h >= 2: current_earnings[i] += dynamic_prizes[i, h]