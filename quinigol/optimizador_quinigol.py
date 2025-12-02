import time
import numpy as np
from numba import njit, prange
import os

# --- CONFIGURATION / CONFIGURACIÓN ---
JACKPOT = 255000.0        # Bote actual
ESTIMATION = 100000.0     # Recaudación estimada (número de columnas jugadas por el público)
BET_PRICE = 1.0           # Precio por apuesta (para simplificar cálculos de lambda)

# Porcentajes de reparto de premios (Normativa LAE)
# Cat 1 (6 aciertos) a Cat 5 (2 aciertos). Nota: Cat 1 suma el JACKPOT.
PRIZE_DISTRIBUTION = np.array([0.10, 0.09, 0.08, 0.08, 0.20]) 

# --- MANUAL DATA INPUT / ENTRADA DE DATOS MANUAL ---
# Probabilidades reales derivadas de cuotas (antes 'bene')
real_probs_1 = [19,9.5,8,4.75,21,10,9,5.5,41,19,17,11,91,51,36,19]
real_probs_2 = [13,7.5,7.5,6,15,8,8.5,7,34,19,17,15,96,56,46,34]
real_probs_3 = [8,15,41,161,5.5,8,23,81,5.5,9,26,76,5.5,9,26,67]
real_probs_4 = [8,11,23,71,5.5,7,17,51,7,9,21,51,9,11,23,56]
real_probs_5 = [13,9,10,15,11,7,9,10,17,11,13,15,23,17,21,29]
real_probs_6 = [29,41,51,201,13,15,29,61,10,9.5,19,46,4,3.75,7.5,15]

# Agrupamos en una lista y convertimos a matriz numpy
raw_real_probs = [real_probs_1, real_probs_2, real_probs_3, real_probs_4, real_probs_5, real_probs_6]
REAL_PROBS_MATRIX = np.zeros((6, 16), dtype=np.float64)

# Normalización de probabilidades (inversa de la cuota)
for i in range(6):
    row = np.array(raw_real_probs[i], dtype=np.float64)
    inv_row = 1.0 / row
    total = np.sum(inv_row)
    REAL_PROBS_MATRIX[i, :] = inv_row / total

# --- LAE DATA LOADING / CARGA DE DATOS LAE ---
# Asumimos que 'quinigol/laequinig.txt' existe
LAE_PROBS_MATRIX = np.zeros((6, 16), dtype=np.float64)

try:
    with open("quinigol/laequinig.txt", mode="r") as data_file:
        values = []
        for line in data_file:
            line = line.replace(",", ".")
            # Parseamos los porcentajes (dividiendo por 100)
            values.append([float(x)/100 for x in line.strip().split(";")])

    # Construcción de la matriz LAE (6 partidos x 16 resultados)
    # Tu lógica combinaba renglones de 2 en 2 (Local y Visitante)
    for i in range(0, 12, 2):
        match_idx = i // 2
        local_probs = values[i]
        visit_probs = values[i+1]
        
        # Producto cartesiano: Probabilidad Local * Probabilidad Visitante
        # Orden de llenado: 00, 01, 02, 0M, 10, 11...
        k = 0
        for l in range(4): # 0, 1, 2, M local
            for v in range(4): # 0, 1, 2, M visitante
                LAE_PROBS_MATRIX[match_idx, k] = local_probs[l] * visit_probs[v]
                k += 1
except FileNotFoundError:
    print("ERROR: No se encontró 'quinigol/laequinig.txt'.")
    # Para evitar crash, llenamos con datos dummy si falla
    LAE_PROBS_MATRIX[:] = REAL_PROBS_MATRIX[:]

# --- COMBINATIONS LOADING / CARGA DE COMBINACIONES ---
try:
    COMBINATIONS = np.load("combinations/kinigmotza.npy").astype(np.int32)
except FileNotFoundError:
    print("¡AVISO! No se encontró 'combinations/kinigmotza.npy'. Usando datos aleatorios de prueba.")
    COMBINATIONS = np.random.randint(0, 16, (10000, 6)).astype(np.int32)

print(f"Processing {len(COMBINATIONS)} columns...")

# --- MATHEMATICAL CORE (JIT COMPILED) / NÚCLEO MATEMÁTICO ---

@njit(fastmath=True)
def calc_taxed_prize(gross_prize):
    """
    Aplica el impuesto español del 20% a premios > 40.000€.
    """
    if gross_prize > 40000.0:
        return (gross_prize - 40000.0) * 0.8 + 40000.0
    return gross_prize

@njit(fastmath=True)
def poisson_prize_share(pot, lambda_val):
    """
    Calcula la parte del bote que te toca usando la Aproximación de Poisson.
    Fórmula: Pot * (1 - e^(-lambda)) / lambda
    Representa la Esperanza del Bote dividido entre (N_ganadores + 1).
    """
    if lambda_val < 1e-9: # Evitar división por cero (casi nadie la tiene)
        return calc_taxed_prize(pot) # Te lo llevas todo tú solo
    
    # Factor de dilución: Qué porcentaje del pastel te toca
    share_factor = (1.0 - np.exp(-lambda_val)) / lambda_val
    
    expected_prize = pot * share_factor
    return calc_taxed_prize(expected_prize)

@njit(parallel=True, fastmath=True)
def evaluate_columns(combs, real_probs, lae_probs, estimation, current_jackpot, prize_dist):
    """
    Evalúa cada columna calculando su EV (Esperanza Matemática) basándose en
    probabilidades reales vs probabilidades del público (LAE).
    """
    n_combs = combs.shape[0]
    
    # Matriz de Resultados: [EV, Prob_Real_6, Indice_Original]
    results = np.zeros((n_combs, 3), dtype=np.float64)
    
    # Pre-cálculos de botes parciales por categoría (excluyendo el bote acumulado del 6)
    cat_pots = estimation * prize_dist # [Cat6, Cat5, Cat4, Cat3, Cat2]
    
    # Paralelización del bucle principal
    for i in prange(n_combs):
        # 1. Extraer probabilidades de los 6 signos elegidos en esta columna
        p_real = np.ones(6, dtype=np.float64)
        p_lae = np.ones(6, dtype=np.float64)
        
        # Probabilidad BASE (Acertar el signo elegido en cada partido)
        for m in range(6):
            sign = combs[i, m]
            p_real[m] = real_probs[m, sign]
            p_lae[m] = lae_probs[m, sign]
            
        # 2. Calcular Probabilidades Reales y LAE de tener exactamente K aciertos.
        # Usamos método directo iterativo para 5, 4, 3 y 2 aciertos.
        
        # --- PROBABILIDAD DE 6 ACIERTOS (PLENO) ---
        prob_real_6 = 1.0
        prob_lae_6 = 1.0
        for m in range(6):
            prob_real_6 *= p_real[m]
            prob_lae_6 *= p_lae[m]
            
        # --- PROBABILIDAD DE 5 ACIERTOS (Falla 1) ---
        prob_real_5 = 0.0
        prob_lae_5 = 0.0
        for m in range(6):
            # Falla el partido m, acierta el resto
            # Lógica: (Prod_Total / p_m) * (1 - p_m)
            if p_real[m] > 0:
                p_r = (prob_real_6 / p_real[m]) * (1.0 - p_real[m])
                prob_real_5 += p_r
            if p_lae[m] > 0:
                p_l = (prob_lae_6 / p_lae[m]) * (1.0 - p_lae[m])
                prob_lae_5 += p_l

        # --- PROBABILIDAD DE 4 ACIERTOS (Fallan 2) ---
        prob_real_4 = 0.0
        prob_lae_4 = 0.0
        # Doble bucle para fallar el par (m, k)
        for m in range(6):
            for k in range(m + 1, 6):
                # Término de fallo
                term_real = (1.0 - p_real[m]) * (1.0 - p_real[k])
                term_lae = (1.0 - p_lae[m]) * (1.0 - p_lae[k])
                
                # Término de acierto (los otros 4)
                prod_rest_real = 1.0
                prod_rest_lae = 1.0
                for x in range(6):
                    if x != m and x != k:
                        prod_rest_real *= p_real[x]
                        prod_rest_lae *= p_lae[x]
                
                prob_real_4 += term_real * prod_rest_real
                prob_lae_4 += term_lae * prod_rest_lae
                
        # --- PROBABILIDAD DE 3 ACIERTOS (Fallan 3) ---
        prob_real_3 = 0.0
        prob_lae_3 = 0.0
        for m in range(6):
            for k in range(m+1, 6):
                for j in range(k+1, 6):
                    term_real = (1.0-p_real[m])*(1.0-p_real[k])*(1.0-p_real[j])
                    term_lae = (1.0-p_lae[m])*(1.0-p_lae[k])*(1.0-p_lae[j])
                    
                    prod_rest_real = 1.0
                    prod_rest_lae = 1.0
                    for x in range(6):
                        if x!=m and x!=k and x!=j:
                            prod_rest_real *= p_real[x]
                            prod_rest_lae *= p_lae[x]
                    
                    prob_real_3 += term_real * prod_rest_real
                    prob_lae_3 += term_lae * prod_rest_lae

        # --- PROBABILIDAD DE 2 ACIERTOS (Aciertan 2) ---
        prob_real_2 = 0.0
        prob_lae_2 = 0.0
        for m in range(6): # m es el primer acierto
            for k in range(m+1, 6): # k es el segundo acierto
                # Aciertan m y k, fallan los otros 4
                term_hit_real = p_real[m] * p_real[k]
                term_hit_lae = p_lae[m] * p_lae[k]
                
                term_miss_real = 1.0
                term_miss_lae = 1.0
                for x in range(6):
                    if x!=m and x!=k:
                        term_miss_real *= (1.0 - p_real[x])
                        term_miss_lae *= (1.0 - p_lae[x])
                
                prob_real_2 += term_hit_real * term_miss_real
                prob_lae_2 += term_hit_lae * term_miss_lae

        # 3. Calcular Esperanza Matemática (EV) Global
        # Lambda = Recaudacion * Probabilidad_LAE_de_esa_categoria
        
        # Cat 6 (Bote + % Recaudacion)
        lambda_6 = estimation * prob_lae_6
        prize_6 = poisson_prize_share(current_jackpot + cat_pots[0], lambda_6)
        
        # Cat 5
        lambda_5 = estimation * prob_lae_5
        prize_5 = poisson_prize_share(cat_pots[1], lambda_5)
        
        # Cat 4
        lambda_4 = estimation * prob_lae_4
        prize_4 = poisson_prize_share(cat_pots[2], lambda_4)
        
        # Cat 3
        lambda_3 = estimation * prob_lae_3
        prize_3 = poisson_prize_share(cat_pots[3], lambda_3)
        
        # Cat 2
        lambda_2 = estimation * prob_lae_2
        prize_2 = poisson_prize_share(cat_pots[4], lambda_2)
        
        # EV Total = Suma(Prob_Real * Premio_Esperado_Neto)
        ev_total = (prob_real_6 * prize_6) + \
                   (prob_real_5 * prize_5) + \
                   (prob_real_4 * prize_4) + \
                   (prob_real_3 * prize_3) + \
                   (prob_real_2 * prize_2)
                   
        results[i, 0] = ev_total
        results[i, 1] = prob_real_6
        results[i, 2] = i # Guardamos el índice para recuperar la combinación

    return results

# --- EXECUTION / EJECUCIÓN ---
if __name__ == "__main__":
    start_time = time.monotonic()

    print("Calculating EV with Numba (Optimized)...")
    
    # Llamada a la función optimizada
    results = evaluate_columns(
        COMBINATIONS, 
        REAL_PROBS_MATRIX, 
        LAE_PROBS_MATRIX, 
        ESTIMATION, 
        JACKPOT, 
        PRIZE_DISTRIBUTION
    )

    # Filtrado y Ordenación
    # 1. Filtrar EV > 1.0 (Rentables). Ajustar según aversión al riesgo.
    ev_threshold = 1.0 
    mask = results[:, 0] > ev_threshold
    filtered_results = results[mask]

    # 2. Ordenar por EV descendente (columna 0)
    # [::-1] invierte el orden para que sea descendente
    sorted_results = filtered_results[filtered_results[:, 0].argsort()[::-1]]

    end_time = time.monotonic()

    # --- OUTPUT / RESULTADOS ---
    print(f"\nCalculation Time: {end_time - start_time:.4f} seconds")
    print(f"Columns analyzed: {len(COMBINATIONS)}")
    print(f"Profitable Columns (EV > {ev_threshold}): {len(sorted_results)}")

    print("\n--- TOP 10 COLUMNS ---")
    
    # Etiquetas para mostrar resultados legibles
    result_labels = ["0-0","0-1","0-2","0-M","1-0","1-1","1-2","1-M","2-0","2-1","2-2","2-M","M-0","M-1","M-2","M-M"]

    for i in range(min(10, len(sorted_results))):
        orig_idx = int(sorted_results[i, 2])
        ev = sorted_results[i, 0]
        prob = sorted_results[i, 1]
        comb = COMBINATIONS[orig_idx]
        
        markers = " | ".join([result_labels[c] for c in comb])
        print(f"#{i+1}: EV={ev:.2f} | Prob6={prob:.2e} | {markers}")

    # Guardar en archivo (formato compatible con tu sistema anterior)
    # Asegúrate de que la carpeta 'results' existe o cámbialo a root si prefieres
    output_path = "results/resulQuinigol_optimizado.txt"
    
    # Crear directorio si no existe para evitar errores
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        for i in range(len(sorted_results)):
            orig_idx = int(sorted_results[i, 2])
            comb = COMBINATIONS[orig_idx]
            # Formato compacto: 00010M... (quitamos los guiones)
            line = "".join([result_labels[c].replace("-", "") for c in comb])
            f.write(line + "\n")
    
    print(f"File saved: {output_path}")