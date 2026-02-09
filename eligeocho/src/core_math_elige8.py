import numpy as np
from numba import njit, prange

@njit(fastmath=True)
def calc_taxed_prize(gross):
    """Impuesto del 20% sobre 40.000€"""
    if gross > 40000.0:
        return (gross - 40000.0) * 0.8 + 40000.0
    return gross

@njit(fastmath=True)
def get_sum_of_products_subset_8(probs_14):
    """
    Calcula la suma de los productos de todos los subconjuntos de tamaño 8
    de un array de 14 probabilidades.
    Equivale a sumar la probabilidad de acierto de las 3003 combinaciones posibles.
    Algoritmo: Programación Dinámica (O(14*8)).
    """
    # dp[k] almacenará la suma de productos de longitud k encontrados hasta ahora
    # Necesitamos llegar hasta k=8
    dp = np.zeros(9, dtype=np.float64)
    dp[0] = 1.0  # Caso base: producto de 0 elementos es 1
    
    for i in range(14):
        p = probs_14[i]
        # Actualizamos dp de atrás hacia adelante para no usar valores de esta misma iteración
        for j in range(8, 0, -1):
            dp[j] = dp[j] + dp[j-1] * p
            
    return dp[8]

@njit(fastmath=True)
def poisson_prize_share_corrected(pot, total_winners_prob, total_bets):
    """
    pot: Bote a repartir.
    total_winners_prob: Suma de probs de las 3003 combinaciones (dp[8]).
    total_bets: Recaudación en columnas.
    """
    # Normalizamos: dp[8] es la suma de probs.
    # Si todos jugaran al azar, la prob de elegir una combinación concreta es 1/3003.
    # Asumimos que la gente se distribuye proporcionalmente a la probabilidad (modelo de mercado eficiente).
    
    # Número esperado de ganadores en TODO el sistema
    lambda_val = total_bets * (total_winners_prob / 3003.0)
    
    if lambda_val < 1e-9:
        return calc_taxed_prize(pot)
    
    prob_any_winner = 1.0 - np.exp(-lambda_val)
    share = prob_any_winner / lambda_val
    return calc_taxed_prize(pot * share)

@njit(parallel=True, fastmath=True)
def calculate_elige8_ev_refined(match_indices, outcomes, real_probs, lae_probs, lae_expected_hit_probs, estimation, pot):
    """
    Calcula el EV considerando que hay 3003 formas de ganar.
    
    Args:
        match_indices: (N, 8) Índices de tus 8 partidos.
        outcomes: (N, 8) Tus pronósticos.
        real_probs: (14, 3) Probabilidad real.
        lae_probs: (14, 3) Probabilidad LAE.
        lae_expected_hit_probs: (14,) Probabilidad esperada de que la gente acierte cada partido
                                (promedio ponderado para los partidos que NO elegimos).
        estimation: Columnas jugadas.
        pot: Bote.
    """
    n = match_indices.shape[0]
    evs = np.zeros(n, dtype=np.float64)
    probs_real = np.zeros(n, dtype=np.float64)
    
    # Buffer temporal para las 14 probabilidades de un escenario concreto
    # Como Numba parallel no permite allocar arrays dinámicos fácilmente dentro del loop,
    # lo hacemos "in-place" o confiamos en la optimización de arrays pequeños.
    
    for i in prange(n):
        p_real_cum = 1.0
        
        # Construimos el array de "Probabilidades de que la gente acierte" para este escenario.
        # Para los 8 partidos que YO juego, la probabilidad de acierto de la gente es la del signo que YO digo 
        # (porque estoy evaluando el caso en que YO gano).
        # Para los 6 partidos que NO juego, uso la esperanza matemática (lae_expected_hit_probs).
        
        current_scenario_lae_probs = np.zeros(14, dtype=np.float64)
        
        # 1. Rellenar con la media por defecto (para los no elegidos)
        for k in range(14):
            current_scenario_lae_probs[k] = lae_expected_hit_probs[k]
            
        # 2. Sobrescribir con los valores concretos de mi apuesta (para los 8 elegidos)
        #    y calcular mi probabilidad real de ganar.
        for k in range(8):
            m_idx = match_indices[i, k]
            res = outcomes[i, k]
            
            # Probabilidad de que YO acierte
            p_real_cum *= real_probs[m_idx, res]
            
            # Si yo acierto, el resultado FUE 'res'.
            # Por tanto, la gente acertó ese partido con probabilidad lae_probs[m_idx, res]
            current_scenario_lae_probs[m_idx] = lae_probs[m_idx, res]
        
        probs_real[i] = p_real_cum
        
        # 3. Calcular cuánta gente gana en TOTAL (sumando las 3003 combinaciones)
        #    dado este escenario de resultados.
        sum_probs_3003 = get_sum_of_products_subset_8(current_scenario_lae_probs)
        
        estimated_prize = poisson_prize_share_corrected(pot, sum_probs_3003, estimation)
        
        evs[i] = p_real_cum * estimated_prize
        
    return evs, probs_real