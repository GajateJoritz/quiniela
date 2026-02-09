import time
import numpy as np
import sys
import os
from itertools import combinations, product

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

try:
    import eligeocho.src.core_math_elige8 as engine
    import quiniela.data.current_data as current_data
except ImportError as e:
    print(f"❌ Error importando módulos: {e}")
    sys.exit(1)

# ==============================================================================
# ⚙️ CONFIGURACIÓN GENERAL
# ==============================================================================
PRECIO_APUESTA = 0.50
RECAUDACION = 100000.0  # Dinero total estimado en la caja
BOTE_REPARTO = RECAUDACION * 0.55
ESTIMACION_COLUMNAS = RECAUDACION / PRECIO_APUESTA

TOP_N = 50  # Número de apuestas a mostrar

# ------------------------------------------------------------------------------
# 🔀 SELECCIÓN DE ESTRATEGIA
# ------------------------------------------------------------------------------
# Opciones disponibles:
#   "EV"            -> Ordena por Valor Esperado (Rentabilidad a largo plazo, más riesgo).
#   "PROB_RENTABLE" -> Ordena por Probabilidad de Acierto (Más seguridad, premios menores).

METODO_ORDENACION = "PROB_RENTABLE" 

# Configuración específica para método "EV"
MIN_EV = 1.4  # Solo mostrar si EV > 1.4€ (2.8x la inversión)

# Configuración específica para método "PROB_RENTABLE"
# Mínimo premio estimado para considerar la apuesta (para evitar premios de 0.20€)
MIN_PREMIO_ESTIMADO = 0.60 

# ==============================================================================

def load_lae_estimations(file_path):
    """Carga estimacion.txt gestionando comas decimales"""
    print(f"📂 Leyendo estimaciones LAE desde: {file_path}")
    lae_probs = []
    
    if not os.path.exists(file_path):
        print(f"❌ Error: No existe {file_path}")
        sys.exit(1)

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line: continue
        try:
            clean_line = line.replace(',', '.')
            numbers = [float(p) for p in clean_line.split()]
            if len(numbers) >= 3:
                lae_probs.append([n / 100.0 for n in numbers[:3]])
        except ValueError:
            continue
        if len(lae_probs) == 14: break

    matrix = np.array(lae_probs, dtype=np.float64)
    if matrix.shape[0] < 14:
        print(f"⚠️ Error: estimacion.txt incompleto ({matrix.shape[0]} filas).")
        sys.exit(1)
    return matrix

def generate_all_candidates():
    """Genera 19.7M de combinaciones"""
    print("🔄 Generando espacio de búsqueda (3003 combinaciones x resultados)...")
    partidos_indices = np.array(list(combinations(range(14), 8)), dtype=np.int8)
    n_groups = len(partidos_indices)
    
    resultados_base = np.array(list(product([0, 1, 2], repeat=8)), dtype=np.int8)
    n_outcomes = len(resultados_base)
    
    # Broadcasting para crear las matrices gigantes
    final_matches = np.repeat(partidos_indices, n_outcomes, axis=0)
    final_outcomes = np.tile(resultados_base, (n_groups, 1))
    
    print(f"   Total apuestas a evaluar: {len(final_matches):,}")
    return final_matches, final_outcomes

def calculate_background_lae_probs(real_probs, lae_probs):
    """Calcula probabilidad esperada para partidos no elegidos"""
    expected_hit = np.zeros(14, dtype=np.float64)
    for i in range(14):
        expected_hit[i] = (real_probs[i, 0] * lae_probs[i, 0] +
                           real_probs[i, 1] * lae_probs[i, 1] +
                           real_probs[i, 2] * lae_probs[i, 2])
    return expected_hit

def main():
    t0 = time.time()
    print(f"\n=== OPTIMIZADOR ELIGE 8 (Modo: {METODO_ORDENACION}) ===")
    print(f"   Recaudación: {RECAUDACION:,.0f} € | Bote: {BOTE_REPARTO:,.0f} €")
    
    # 1. Cargar Datos
    try:
        real_probs = np.array(current_data.REAL_1X2, dtype=np.float64)[:14]
        path_est = os.path.join(BASE_DIR, 'quiniela', 'estimacion.txt')
        lae_probs = load_lae_estimations(path_est)
    except Exception as e:
        print(f"❌ Error de datos: {e}")
        return

    # 2. Preparar cálculos auxiliares
    lae_expected = calculate_background_lae_probs(real_probs, lae_probs)

    # 3. Generar Candidatos
    matches, outcomes = generate_all_candidates()
    
    # 4. Calcular EV Masivo
    print("⚡ Calculando EV y Probabilidades...")
    t_calc = time.time()
    
    evs, probs_real = engine.calculate_elige8_ev_refined(
        matches, outcomes, 
        real_probs, lae_probs, lae_expected,
        ESTIMACION_COLUMNAS, BOTE_REPARTO
    )
    
    print(f"   Cálculo completado en {time.time() - t_calc:.2f}s")
    
    # 5. Filtrado y Ordenación según Estrategia
    print(f"🎯 Aplicando estrategia: {METODO_ORDENACION}")
    
    top_indices = []
    
    if METODO_ORDENACION == "EV":
        # Estrategia Clásica: Maximizar Valor Esperado
        threshold = PRECIO_APUESTA * MIN_EV
        good_indices = np.where(evs > threshold)[0]
        
        print(f"   Apuestas con EV > {threshold:.2f}€: {len(good_indices)}")
        
        if len(good_indices) == 0:
            top_indices = np.argsort(evs)[::-1][:TOP_N]
        else:
            sorted_good = np.argsort(evs[good_indices])[::-1]
            top_indices = good_indices[sorted_good[:TOP_N]]
            
    elif METODO_ORDENACION == "PROB_RENTABLE":
        # Estrategia Conservadora: Maximizar Probabilidad (sujeto a rentabilidad)
        
        # Calcular premio estimado (EV / Prob) evitando división por cero
        safe_probs = np.where(probs_real < 1e-12, 1e-12, probs_real)
        estimated_prizes = evs / safe_probs
        
        # Filtros:
        # 1. EV > Coste (Matemáticamente no perdemos dinero a largo plazo)
        # 2. Premio Estimado > Mínimo (No queremos acertar para ganar 0.20€)
        mask_rentable = (evs > PRECIO_APUESTA) & (estimated_prizes > MIN_PREMIO_ESTIMADO)
        
        good_indices = np.where(mask_rentable)[0]
        print(f"   Apuestas rentables (Premio > {MIN_PREMIO_ESTIMADO}€): {len(good_indices)}")
        
        if len(good_indices) == 0:
            print("   ⚠️ No hay apuestas rentables seguras. Mostrando las de mayor probabilidad absoluta...")
            top_indices = np.argsort(probs_real)[::-1][:TOP_N]
        else:
            # Ordenar por PROBABILIDAD REAL descendente
            sorted_by_prob = np.argsort(probs_real[good_indices])[::-1]
            top_indices = good_indices[sorted_by_prob[:TOP_N]]

    else:
        print("❌ Método de ordenación no reconocido.")
        return

    # 6. Mostrar Informe
    print(f"\n🏆 TOP {TOP_N} ELIGE 8")
    
    # Cabecera dinámica
    if METODO_ORDENACION == "PROB_RENTABLE":
        print(f"{'#':<4} {'PARTIDOS (Indices+1)':<25} {'PRONÓSTICO':<12} {'PROB REAL':<12} {'PREMIO EST':<12} {'EV (€)':<10}")
    else:
        print(f"{'#':<4} {'PARTIDOS (Indices+1)':<25} {'PRONÓSTICO':<20} {'PROB REAL':<15} {'EV (€)':<10}")
        
    print("-" * 95)
    
    sym = ["1", "X", "2"]
    for i, idx in enumerate(top_indices):
        m_idxs = matches[idx]
        res_idxs = outcomes[idx]
        
        str_partidos = str(m_idxs + 1)
        str_prono = "".join([sym[r] for r in res_idxs])
        val_prob = probs_real[idx]
        val_ev = evs[idx]
        
        if METODO_ORDENACION == "PROB_RENTABLE":
            # Calcular premio estimado para mostrarlo
            val_premio = val_ev / max(1e-12, val_prob)
            mark = "🔥" if val_prob > 0.05 else "" 
            print(f"{i+1:<4} {str_partidos:<25} {str_prono:<12} {val_prob*100:.4f}%     {val_premio:.2f} €       {val_ev:.4f} {mark}")
        else:
            mark = "💎" if val_ev > 2.0 else ""
            print(f"{i+1:<4} {str_partidos:<25} {str_prono:<20} {val_prob*100:.5f}%     {val_ev:.4f} {mark}")

    print("-" * 95)
    print(f"Tiempo total: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()