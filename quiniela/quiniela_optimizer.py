import time
import numpy as np
import os
import sys
import src.core_math_quiniela as engine

# --- CONFIGURACIÓN DE ESTRATEGIA ---

# Opciones para TARGET_PLENO:
#   "ALL"   -> Genera las 16 variantes de Pleno para CADA columna de 14 partidos. (Recomendado si hay CPU)
#   "M1"   -> Fuerza que todas las columnas analizadas tengan este Pleno.
#   None    -> Usa el Pleno que venga en el archivo motza.npy original.
TARGET_PLENO = "0M"  

# Ajustes de Optimización
N_SIMULATIONS = 100000
PORTFOLIO_SIZE = 10
MIN_EV_THRESHOLD = 1.4
OPTIMIZATION_MODE = 1 # 2=Sortino

# --- CARGA DE DATOS ---
try:
    import data.current_data as current_data
    DATA_READY = True
except ImportError:
    DATA_READY = False

JACKPOT = getattr(current_data, 'JACKPOT', 0.0) if DATA_READY else 0.0
ESTIMATION = getattr(current_data, 'ESTIMATION', 4000000.0) if DATA_READY else 4000000.0
BET_PRICE = 0.75
DISTRIBUTION = np.array([0.075, 0.16, 0.075, 0.075, 0.075, 0.09], dtype=np.float64)

# Utiles para mapeo
MAPA_PLENO = ["00","01","02","0M","10","11","12","1M","20","21","22","2M","M0","M1","M2","MM"]
MAPA_PLENO_REV = {k:v for v,k in enumerate(MAPA_PLENO)}

def parse_probs_local(filepath):
    """
    Fallback para leer estimacion.txt
    CORRECCIÓN: Se añade encoding='utf-8-sig' para evitar el error del BOM (ï»¿)
    """
    if not os.path.exists(filepath):
        sys.exit(f"Falta el archivo: {filepath}")
        
    raw = []
    # --- AQUÍ ESTÁ LA CORRECCIÓN ---
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for l in f:
            # Limpieza extra para evitar líneas vacías o caracteres raros
            linea_limpia = l.strip().replace(",", ".")
            if not linea_limpia: 
                continue
                
            try:
                v = [float(x) for x in linea_limpia.split()]
                # Si viene en porcentaje (ej: 45.5), pasamos a tanto por uno
                if sum(v) > 1.5: 
                    v = [x/100.0 for x in v]
                raw.append(v)
            except ValueError:
                continue

    # Procesar 1X2 (Primeras 14 líneas)
    p1x2 = np.array(raw[:14], dtype=np.float64)
    
    # Procesar Pleno (si existe)
    pp = np.ones(16)/16.0 
    if len(raw) >= 16:
        # Asume que línea 15 son goles Local y línea 16 goles Visitante
        pp = [l*v for l in raw[14] for v in raw[15]]
        
    return p1x2, np.array(pp, dtype=np.float64)

def load_and_expand_combinations(path, strategy):
    """
    Carga motza.npy y expande las combinaciones según la estrategia de Pleno.
    """
    if not os.path.exists(path): sys.exit(f"Falta {path}")
    print(f"   Cargando base de datos: {path}...")
    data = np.load(path)
    
    # Extraemos solo la parte 1X2 (los primeros 14 signos)
    # y nos aseguramos de que sean únicos (por si acaso)
    base_1x2 = data[:, :14].astype(np.int8)
    # base_1x2 = np.unique(base_1x2, axis=0) # Opcional: eliminar duplicados de 14
    
    n_base = len(base_1x2)
    print(f"   Columnas base (14 partidos): {n_base}")

    if strategy == "ALL":
        print(f"   🚀 ESTRATEGIA 'ALL': Generando 16 variantes por columna...")
        # Repetimos cada fila 16 veces
        expanded_1x2 = np.repeat(base_1x2, 16, axis=0)
        # Generamos secuencia 0..15 repetida N veces
        expanded_pleno = np.tile(np.arange(16, dtype=np.int8), n_base)
        
        return expanded_1x2, expanded_pleno
        
    elif isinstance(strategy, str) and strategy in MAPA_PLENO_REV:
        idx = MAPA_PLENO_REV[strategy]
        print(f"   🎯 ESTRATEGIA FIJA: Aplicando Pleno '{strategy}' (idx {idx}) a todo...")
        return base_1x2, np.full(n_base, idx, dtype=np.int8)
        
    else:
        print("   ℹ️ Usando Pleno original del archivo.")
        return base_1x2, data[:, 14].astype(np.int8)

def main():
    t0 = time.time()
    print("\n=== QUINIELA OPTIMIZER (DYNAMIC EXPANSION) ===")
    
    # 1. CARGA PROBS
    lae_1x2, lae_pleno = parse_probs_local("quiniela/estimacion.txt")
    if DATA_READY:
        real_1x2 = current_data.REAL_1X2
        real_pleno = current_data.REAL_PLENO
    else:
        print("❌ Faltan datos reales. Ejecuta generar_datos.py")
        return

    # 2. CARGA Y EXPANSIÓN DE COMBINACIONES
    c_1x2, c_pleno = load_and_expand_combinations("combinations/motza.npy", TARGET_PLENO)
    print(f"   Total candidatos a evaluar: {len(c_1x2)}")
    
    # 3. FILTRO EV (Aquí es donde se descartan los Plenos improbables si su premio no compensa)
    print(f"Calculando EV...")
    evs = engine.get_top_candidates_quiniela(
        c_1x2, c_pleno, real_1x2, real_pleno, lae_1x2, lae_pleno,
        ESTIMATION, JACKPOT, DISTRIBUTION
    )
    
    mask = evs > MIN_EV_THRESHOLD
    candidates = np.where(mask)[0]
    
    # Ordenar
    candidates = candidates[np.argsort(evs[candidates])[::-1]]
    
    # Limitar pool para la simulación pesada
    # Si expandimos a 16x, el pool inicial es enorme, así que nos quedamos con los mejores 15k
    if len(candidates) > 15000: 
        print(f"   Reduciendo candidatos de {len(candidates)} a los Top 15000 para Monte Carlo.")
        candidates = candidates[:15000]
    else:
        print(f"   Pasando {len(candidates)} candidatos a Monte Carlo.")
    
    if len(candidates) == 0: 
        print("❌ Ningún candidato supera el umbral EV.")
        return

    # 4. SIMULACIÓN MONTE CARLO
    print(f"Simulando {N_SIMULATIONS} escenarios...")
    scenarios = engine.generate_scenarios_quiniela(real_1x2, real_pleno, N_SIMULATIONS)
    prizes = engine.precompute_scenario_prizes_quiniela(
        scenarios, lae_1x2, lae_pleno, ESTIMATION, JACKPOT, DISTRIBUTION
    )
    
    # 5. OPTIMIZACIÓN
    print(f"Optimizando cartera ({PORTFOLIO_SIZE} apuestas)...")
    selected = []
    earnings = np.zeros(N_SIMULATIONS, dtype=np.float64)
    pool_1x2 = c_1x2[candidates]
    pool_pleno = c_pleno[candidates]
    
    for step in range(PORTFOLIO_SIZE):
        best_idx = -1
        best_val = -float('inf')
        cost = step * BET_PRICE
        
        for i in range(len(candidates)):
            if i in selected: continue
            m = engine.calculate_candidate_metric_quiniela(
                pool_1x2[i], pool_pleno[i], scenarios, earnings, prizes,
                OPTIMIZATION_MODE, cost
            )
            if m > best_val:
                best_val = m
                best_idx = i
        
        if best_idx != -1:
            selected.append(best_idx)
            engine.update_earnings_quiniela(
                earnings, pool_1x2[best_idx], pool_pleno[best_idx], scenarios, prizes
            )
            val_pleno = MAPA_PLENO[pool_pleno[best_idx]]
            print(f"[{step+1:02d}] P15:{val_pleno} | Metric:{best_val:.4f}")
        else: break
        
    # 6. GUARDAR
    os.makedirs("results", exist_ok=True)
    sym = ["1","X","2"]
    with open("results/apuesta_quiniela_optimizada.txt", "w") as f:
        for i in selected:
            row = "".join([sym[x] for x in pool_1x2[i]]) + " + " + MAPA_PLENO[pool_pleno[i]]
            print(row)
            f.write(row+"\n")
            
    print(f"\nROI: {((np.mean(earnings)-(PORTFOLIO_SIZE*BET_PRICE))/(PORTFOLIO_SIZE*BET_PRICE))*100:.2f}%")
    print(f"Tiempo: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()