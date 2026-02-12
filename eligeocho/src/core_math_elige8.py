import numpy as np
from numba import njit, prange

@njit(fastmath=True)
def calc_taxed_prize(gross):
    """Impuesto del 20% sobre 40.000€"""
    if gross > 40000.0:
        return (gross - 40000.0) * 0.8 + 40000.0
    return gross

@njit(fastmath=True)
def poisson_prize_share_heuristic(pot, total_bets, selection_prob, win_prob_given_selection):
    """
    Calcula premio basado en probabilidad de elección directa.
    
    Args:
        selection_prob: Probabilidad de que un apostante elija esta combinación (0.0 a 1.0).
        win_prob_given_selection: Probabilidad de acertarla una vez elegida (Real Prob).
    """
    # 1. Porcentaje total de boletos que coinciden con esta combinación Y aciertan
    # (Gente que la eligió * Gente que acertó sus pronósticos)
    total_share = selection_prob * win_prob_given_selection
    
    # 2. Número esperado de acertantes
    lambda_val = total_bets * total_share
    
    if lambda_val < 1e-9:
        return calc_taxed_prize(pot)
    
    # 3. Reparto
    prob_any_winner = 1.0 - np.exp(-lambda_val)
    share = prob_any_winner / lambda_val
    return calc_taxed_prize(pot * share)

@njit(parallel=True, fastmath=True)
def calculate_elige8_ev_heuristic(match_indices, outcomes, real_probs, estimation, pot, match_log_weights, min_score, max_score, max_share=0.15, min_share=0.00001):
    """
    Calcula EV usando interpolación heurística de popularidad.
    
    Args:
        match_log_weights: (14,) Logaritmo de los pesos de popularidad de cada partido.
        min_score: Suma de log-pesos de la combinación más difícil posible.
        max_score: Suma de log-pesos de la combinación más fácil posible.
        max_share: % de la población que juega la combinación más fácil (0.15 = 15%).
        min_share: % de la población que juega la combinación más rara.
    """
    n = match_indices.shape[0]
    evs = np.zeros(n, dtype=np.float64)
    probs_real = np.zeros(n, dtype=np.float64)
    
    # Precalcular log-ratio para interpolación
    # Formula: Share = Min * (Max/Min)^NormalizedPos
    # Log(Share) = Log(Min) + NormalizedPos * (Log(Max) - Log(Min))
    log_min = np.log(min_share)
    log_diff_share = np.log(max_share) - log_min
    score_range = max_score - min_score
    if score_range < 1e-9: score_range = 1.0 # Evitar div/0
    
    for i in prange(n):
        p_real_cum = 1.0
        current_score = 0.0
        
        for k in range(8):
            m_idx = match_indices[i, k]
            res = outcomes[i, k]
            
            # 1. Probabilidad Real de Ganar
            p_real_cum *= real_probs[m_idx, res]
            
            # 2. Popularidad (Score)
            # Sumamos el peso del partido (cuánto atrae a la gente)
            current_score += match_log_weights[m_idx]
            
        probs_real[i] = p_real_cum
        
        # 3. Calcular Probabilidad de Selección (Crowding)
        # Normalizamos el score entre 0 (nadie la juega) y 1 (todos la juegan)
        normalized_pos = (current_score - min_score) / score_range
        
        # Interpolamos exponencialmente el Share
        # Si normalized_pos es 1 (Favoritos) -> usa max_share (15%)
        log_share = log_min + (normalized_pos * log_diff_share)
        selection_prob = np.exp(log_share)
        
        # 4. Calcular Premio
        # Asumimos que si la gente elige esta combinación, apuesta al signo favorito (o al que pusimos)
        # Simplificación: El selection_prob ya captura la "masificación" de la columna.
        # Usamos p_real_cum como proxy de la dificultad del signo.
        
        estimated_prize = poisson_prize_share_heuristic(pot, estimation, selection_prob, p_real_cum)
        
        evs[i] = p_real_cum * estimated_prize
        
    return evs, probs_real