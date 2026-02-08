import time
import numpy as np
import os
import sys
import src.core_math_quiniela as engine

# --- CONFIGURACIÓN DE ESTRATEGIA ---

# Opciones para TARGET_PLENO:
#   "ALL"   -> Genera las 16 variantes de Pleno para CADA columna de 14 partidos. (Recomendado si hay CPU)
#   "M1"    -> Fuerza que todas las columnas analizadas tengan este Pleno.
#   None    -> Usa el Pleno que venga en el archivo motza.npy original.
TARGET_PLENO = "01"  

# Ajustes de Optimización
N_SIMULATIONS = 100000
PORTFOLIO_SIZE = 10
MIN_EV_THRESHOLD = 1.4

# MODO DE SELECCIÓN:
#   0 = DEBUG EV (Selecciona por puro EV Analítico, sin diversificar riesgo).
#   1 = Probabilidad de Beneficio.
#   2 = Sortino Ratio (Recomendado: Equilibrio Rentabilidad/Riesgo).
OPTIMIZATION_MODE = 1

# --- CARGA DE DATOS ---
try:
    import data.current_data as current_data
    DATA_READY = True
except ImportError:
    try:
        # Intentamos importar desde la raíz por si acaso
        import current_data
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

    # Limpieza de la estrategia (quitar guiones, espacios, mayúsculas)
    if isinstance(strategy, str):
        strategy_clean = strategy.replace("-", "").replace(",", "").replace(" ", "").upper()
    else:
        strategy_clean = strategy

    if strategy_clean == "ALL":
        print(f"   🚀 ESTRATEGIA 'ALL': Generando 16 variantes por columna...")
        # Repetimos cada fila 16 veces
        expanded_1x2 = np.repeat(base_1x2, 16, axis=0)
        # Generamos secuencia 0..15 repetida N veces
        expanded_pleno = np.tile(np.arange(16, dtype=np.int8), n_base)
        
        return expanded_1x2, expanded_pleno
        
    elif isinstance(strategy_clean, str) and strategy_clean in MAPA_PLENO_REV:
        idx = MAPA_PLENO_REV[strategy_clean]
        print(f"   🎯 ESTRATEGIA FIJA: Aplicando Pleno '{strategy}' (idx {idx}) a todo...")
        return base_1x2, np.full(n_base, idx, dtype=np.int8)
        
    else:
        print("   ℹ️ Usando Pleno original del archivo.")
        return base_1x2, data[:, 14].astype(np.int8)

def main():
    t0 = time.time()
    print("\n=== QUINIELA OPTIMIZER ===")
    
    # 1. CARGA DATOS
    lae_1x2, lae_pleno = parse_probs_local("quiniela/estimacion.txt")
    if DATA_READY:
        real_1x2 = current_data.REAL_1X2
        real_pleno = current_data.REAL_PLENO
    else:
        print("❌ Error: Ejecuta generar_datos.py primero.")
        return

    c_1x2, c_pleno = load_and_expand_combinations("combinations/motza.npy", TARGET_PLENO)
    
    # 2. FILTRO EV
    print(f"Calculando EV para {len(c_1x2)} columnas...")
    evs = engine.get_top_candidates_quiniela(
        c_1x2, c_pleno, real_1x2, real_pleno, lae_1x2, lae_pleno,
        ESTIMATION, JACKPOT, DISTRIBUTION
    )
    
    mask = evs > MIN_EV_THRESHOLD
    candidates = np.where(mask)[0]
    # Ordenar por EV descendente
    candidates = candidates[np.argsort(evs[candidates])[::-1]]
    
    print(f"Candidatos viables (> {MIN_EV_THRESHOLD} EV): {len(candidates)}")
    if len(candidates) == 0: return

    # El usuario controla la carga mediante N_SIMULATIONS
    print(f"   Pasando TODOS los {len(candidates)} candidatos a la optimización.")

    # 3. SIMULACIÓN MONTE CARLO (Necesaria para métricas finales o modos 1 y 2)
    # Incluso en modo 0 simulamos para dar datos de ROI realistas al final
    print(f"Simulando {N_SIMULATIONS} escenarios...")
    scenarios = engine.generate_scenarios_quiniela(real_1x2, real_pleno, N_SIMULATIONS)
    prizes = engine.precompute_scenario_prizes_quiniela(
        scenarios, lae_1x2, lae_pleno, ESTIMATION, JACKPOT, DISTRIBUTION
    )

    # 4. SELECCIÓN DE CARTERA
    selected_indices = [] # Indices locales dentro de 'candidates'
    earnings = np.zeros(N_SIMULATIONS, dtype=np.float64)
    pool_1x2 = c_1x2[candidates]
    pool_pleno = c_pleno[candidates]
    
    # Símbolos para imprimir la quiniela
    sym = ["1", "X", "2"]

    if OPTIMIZATION_MODE == 0:
        print(f"\n⚡ MODO DEBUG (EV PURO): Seleccionando las {PORTFOLIO_SIZE} mejores columnas...")
        count = min(PORTFOLIO_SIZE, len(candidates))
        selected_indices = list(range(count))
        # Actualizamos earnings para estadísticas finales
        for i in selected_indices:
            engine.update_earnings_quiniela(earnings, pool_1x2[i], pool_pleno[i], scenarios, prizes)
            
            # IMPRIMIR AL ENCONTRAR (MODO 0)
            real_idx = candidates[i]
            ev_val = evs[real_idx]
            p14 = pool_1x2[i]
            pp = pool_pleno[i]
            txt_14 = "".join([sym[x] for x in p14])
            txt_pleno = MAPA_PLENO[pp]
            
            print(f"[{i+1:02d}] {txt_14} + {txt_pleno} | EV:{ev_val:.4f}")
            
    else:
        print(f"\n⚙️  MODO OPTIMIZADOR ({OPTIMIZATION_MODE}): Buscando cartera de bajo riesgo...")
        print("   (Pulsa Ctrl+C para detener y guardar lo calculado)")
        
        try:
            for step in range(PORTFOLIO_SIZE):
                best_idx = -1
                best_val = -float('inf')
                cost = step * BET_PRICE
                
                # Bucle de búsqueda del mejor candidato
                for i in range(len(candidates)):
                    if i in selected_indices: continue
                    
                    m = engine.calculate_candidate_metric_quiniela(
                        pool_1x2[i], pool_pleno[i], scenarios, earnings, prizes,
                        OPTIMIZATION_MODE, cost
                    )
                    if m > best_val:
                        best_val = m
                        best_idx = i
                
                if best_idx != -1:
                    selected_indices.append(best_idx)
                    engine.update_earnings_quiniela(
                        earnings, pool_1x2[best_idx], pool_pleno[best_idx], scenarios, prizes
                    )
                    
                    # IMPRIMIR AL ENCONTRAR (MODO 1 y 2)
                    real_idx = candidates[best_idx]
                    ev_val = evs[real_idx]
                    p14 = pool_1x2[best_idx]
                    pp = pool_pleno[best_idx]
                    txt_14 = "".join([sym[x] for x in p14])
                    txt_pleno = MAPA_PLENO[pp]
                    
                    print(f"[{step+1:02d}] {txt_14} + {txt_pleno} | Metric:{best_val:.4f} | EV:{ev_val:.4f}")

                else: 
                    print("   No se encontraron más columnas que mejoren la cartera.")
                    break
                    
        except KeyboardInterrupt:
            print("\n\n🛑 DETENIDO POR EL USUARIO (Ctrl+C). Guardando resultados parciales...")

    # 5. RESULTADOS
    print("\n=== CARTERA SELECCIONADA ===")
    os.makedirs("results", exist_ok=True)
    
    total_theoretical_ev = 0.0
    
    with open("results/apuesta_quiniela_optimizada.txt", "w") as f:
        for i, local_idx in enumerate(selected_indices):
            # Recuperar índice original para sacar el EV
            real_idx = candidates[local_idx]
            ev_val = evs[real_idx]
            total_theoretical_ev += ev_val
            
            p14 = pool_1x2[local_idx]
            pp  = pool_pleno[local_idx]
            
            txt_14 = "".join([sym[x] for x in p14])
            txt_pleno = MAPA_PLENO[pp]
            
            # Formato de salida con EV
            linea = f"{txt_14} + {txt_pleno}"
            info_extra = f" | EV: {ev_val:.4f} €"
            
            print(f"{i+1:02d}: {linea}{info_extra}")
            f.write(linea + "\n") # En el txt limpio solo guardamos la apuesta
            
    # Estadísticas Globales
    invested = len(selected_indices) * BET_PRICE
    if invested > 0:
        expected_return = np.mean(earnings) # Esto es el EV Simulado Total
        roi = ((expected_return - invested) / invested) * 100
        
        print("-" * 30)
        print(f"Inversión Total:      {invested:.2f} €")
        print(f"EV Teórico Total:     {total_theoretical_ev:.2f} €")
        print(f"EV Simulado Total:    {expected_return:.2f} €")
        print(f"ROI Estimado:         {roi:+.2f}%")
    else:
        print("\n⚠️ No se seleccionaron apuestas.")
        
    print(f"Tiempo Total:         {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()