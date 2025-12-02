import numpy as np
import time

# --- CONFIGURATION / CONFIGURACIÓN ---
N_SIMULATIONS = 100000 # Número de simulaciones (partidos ficticios jugados)

# Definimos 6 partidos y sus probabilidades REALES (simplificadas para el ejemplo)
# Imagina que en cada partido el resultado '0' (índice 0) tiene un 50% de salir.
# Esto es para que sea fácil ver si el algoritmo funciona.
# Matriz (6 partidos x 16 resultados posibles)
REAL_PROBS = np.zeros((6, 16), dtype=np.float64)

# Llenamos la matriz:
# Resultado 0 (ej "0-0"): 50% probabilidad
# Resultado 1 (ej "1-0"): 30% probabilidad
# Resultado 2 (ej "0-1"): 20% probabilidad
# Resto: 0%
for i in range(6):
    REAL_PROBS[i, 0] = 0.50
    REAL_PROBS[i, 1] = 0.30
    REAL_PROBS[i, 2] = 0.20

# --- SIMPLE PORTFOLIO / CARTERA SENCILLA ---
# Vamos a crear 3 apuestas manualmente para ver cómo se comportan.

# Apuesta 1: Todo al resultado más probable (0)
bet_1 = [0, 0, 0, 0, 0, 0] 

# Apuesta 2: Casi igual a la 1, solo cambia el último partido (solapamiento alto)
bet_2 = [0, 0, 0, 0, 0, 1] 

# Apuesta 3: Todo al resultado secundario (1) (totalmente distinta, diversificación)
bet_3 = [1, 1, 1, 1, 1, 1]

# Nuestra cartera
my_portfolio = np.array([bet_1, bet_2, bet_3], dtype=np.int32)

# --- CORE ALGORITHM / ALGORITMO NÚCLEO ---

def simulate_matches(probs_matrix, n_sims):
    """
    Genera N resultados ganadores aleatorios basados en las probabilidades.
    Devuelve una matriz (n_sims x 6) con los resultados que 'ocurrieron'.
    """
    n_matches = 6
    simulated_results = np.zeros((n_sims, n_matches), dtype=np.int32)
    
    print(f"Generando {n_sims} jornadas ficticias...")
    
    for m in range(n_matches):
        # Probabilidades de este partido
        p = probs_matrix[m]
        # numpy.random.choice elige un índice (0-15) basado en la prob 'p'
        simulated_results[:, m] = np.random.choice(16, size=n_sims, p=p)
        
    return simulated_results

def calculate_portfolio_performance(portfolio, simulated_results):
    """
    Compara nuestra cartera contra los resultados simulados y cuenta aciertos.
    """
    n_sims = simulated_results.shape[0]
    n_bets = portfolio.shape[0]
    
    # Estadísticas globales
    hits_per_sim = np.zeros((n_sims, n_bets), dtype=np.int32)
    
    # Comparamos cada apuesta
    # (Esto se puede optimizar más, pero para el ejemplo es didáctico así)
    for i in range(n_bets):
        # Comparamos la apuesta 'i' con TODAS las simulaciones a la vez
        # bet_matches: matriz booleana (True donde acertamos)
        bet = portfolio[i]
        matches = (simulated_results == bet)
        # Sumamos los True por fila (axis=1) para ver aciertos totales en cada simulación
        hits_per_sim[:, i] = np.sum(matches, axis=1)
        
    return hits_per_sim

# --- EXECUTION / EJECUCIÓN ---

start_time = time.time()

# 1. Simular la realidad (Generar el "Ground Truth")
sim_results = simulate_matches(REAL_PROBS, N_SIMULATIONS)

# 2. Evaluar nuestra cartera
hits_matrix = calculate_portfolio_performance(my_portfolio, sim_results)

# hits_matrix es una tabla gigante de N_SIMS filas x 3 columnas (apuestas).
# Cada celda dice: "En la simulación 50, la apuesta 2 tuvo 4 aciertos".

# --- ANÁLISIS DE SOLAPAMIENTO / OVERLAP ANALYSIS ---

print("\n--- ANÁLISIS DE RESULTADOS ---")

# 1. Probabilidad INDIVIDUAL (La "suma tonta")
# Calculamos cuántas veces cada apuesta tuvo 6 aciertos por separado
for i in range(len(my_portfolio)):
    full_hits = np.sum(hits_matrix[:, i] == 6)
    pct = (full_hits / N_SIMULATIONS) * 100
    print(f"Apuesta #{i+1} Probabilidad individual de Pleno: {pct:.4f}%")

# 2. Probabilidad COLECTIVA (La realidad con solapamiento)
# Queremos saber: ¿En qué % de simulaciones AL MENOS UNA apuesta tuvo 6 aciertos?
# Usamos np.max a lo largo de las apuestas para ver el mejor resultado de la cartera en esa simulación
best_outcome_per_sim = np.max(hits_matrix, axis=1)

prob_portfolio_6 = np.sum(best_outcome_per_sim == 6) / N_SIMULATIONS * 100
prob_portfolio_5 = np.sum(best_outcome_per_sim >= 5) / N_SIMULATIONS * 100

print(f"\n--- RENDIMIENTO REAL DE LA CARTERA (Conjunto) ---")
print(f"Probabilidad de tener AL MENOS un Pleno (6): {prob_portfolio_6:.4f}%")
print(f"Probabilidad de tener AL MENOS un 5 o más:   {prob_portfolio_5:.4f}%")

# 3. Demostración del Solapamiento (Correlación)
# Vamos a ver cuántas veces ganan juntas la Apuesta 1 y la Apuesta 2 (que son parecidas)
# vs la Apuesta 1 y la 3 (que son diferentes).

# Ganar juntas (ambas tienen >= 5 aciertos en la misma simulación)
hits_1_and_2 = np.sum((hits_matrix[:, 0] >= 5) & (hits_matrix[:, 1] >= 5))
hits_1_and_3 = np.sum((hits_matrix[:, 0] >= 5) & (hits_matrix[:, 2] >= 5))

print(f"\n--- SOLAPAMIENTO (Correlación) ---")
print(f"Simulaciones donde Apuesta 1 y 2 tienen premio (>=5) A LA VEZ: {hits_1_and_2}")
print(f"Simulaciones donde Apuesta 1 y 3 tienen premio (>=5) A LA VEZ: {hits_1_and_3}")

print("\nINTERPRETACIÓN:")
print("La Apuesta 1 y 2 se solapan mucho. Si una gana, la otra casi seguro también.")
print("Esto AUMENTA la varianza (o ganas mucho o pierdes todo).")
print("La Apuesta 1 y 3 rara vez ganan juntas. Esto es DIVERSIFICACIÓN real.")