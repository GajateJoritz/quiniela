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
RECAUDACION = 55000.0
BOTE_REPARTO = RECAUDACION * 0.55
ESTIMACION_COLUMNAS = RECAUDACION / PRECIO_APUESTA

TOP_N = 20  # Número de apuestas a mostrar

# ------------------------------------------------------------------------------
# 🔀 SELECCIÓN DE ESTRATEGIA
# ------------------------------------------------------------------------------
# Opciones disponibles:
#   "EV"            -> Ordena por Valor Esperado (Rentabilidad a largo plazo, más riesgo).
#   "PROB_RENTABLE" -> Ordena por Probabilidad de Acierto (Más seguridad, premios menores).

METODO_ORDENACION = "PROB_RENTABLE" 

# Configuración específica para método "EV"
MIN_EV = 0.75  # Solo mostrar si EV > 1.4€ (2.8x la inversión)

# Configuración específica para método "PROB_RENTABLE"
# Mínimo premio estimado para considerar la apuesta (para evitar premios de 0.20€)
#MIN_PREMIO_ESTIMADO = 0.70 

# ==============================================================================
# CONFIGURACIÓN DE MASIFICACIÓN (HEURÍSTICA)
# MAX_SHARE: Qué % de la gente crees que juega la combinación de 8 favoritos.
# 0.15 (15%) es conservador. 0.25 (25%) es agresivo.
CROWD_MAX_SHARE = 0.20 
# MIN_SHARE: Qué % juega una combinación rara aleatoria.
CROWD_MIN_SHARE = 1.0 / 100000.0 

# ==============================================================================

def load_lae_estimations(file_path):
    """Carga estimacion.txt"""
    print(f"📂 Leyendo estimaciones LAE desde: {file_path}")
    lae_probs = []
    if not os.path.exists(file_path):
        print(f"❌ Error: No existe {file_path}")
        sys.exit(1)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                nums = [float(p.replace(',', '.')) for p in line.split()]
                if len(nums) >= 3: lae_probs.append([n/100.0 for n in nums[:3]])
            except: continue
            if len(lae_probs) == 14: break
    return np.array(lae_probs, dtype=np.float64)

def generate_all_candidates():
    """Genera 19.7M de combinaciones"""
    print("🔄 Generando espacio de búsqueda...")
    matches = np.array(list(combinations(range(14), 8)), dtype=np.int8)
    outcomes_base = np.array(list(product([0, 1, 2], repeat=8)), dtype=np.int8)
    
    n_groups = len(matches)
    n_outcomes = len(outcomes_base)
    
    final_matches = np.repeat(matches, n_outcomes, axis=0)
    final_outcomes = np.tile(outcomes_base, (n_groups, 1))
    
    print(f"   Total apuestas: {len(final_matches):,}")
    return final_matches, final_outcomes

def main():
    t0 = time.time()
    print(f"\n=== OPTIMIZADOR ELIGE 8 (Modelo Heurístico) ===")
    print(f"   Recaudación: {RECAUDACION:,.0f} € | Bote: {BOTE_REPARTO:,.0f} €")
    print(f"   Columnas Estimadas: {ESTIMACION_COLUMNAS:,} (Precio: {PRECIO_APUESTA}€)")
    
    # 1. Cargar Datos
    try:
        real_probs = np.array(current_data.REAL_1X2, dtype=np.float64)[:14]
        path_est = os.path.join(BASE_DIR, 'quiniela', 'estimacion.txt')
        lae_probs = load_lae_estimations(path_est)
    except Exception as e:
        print(f"❌ Error de datos: {e}")
        return

    # 2. Calcular Pesos de Popularidad (Log-Weights)
    # El 'atractivo' de un partido es su probabilidad máxima LAE
    match_appeal = np.max(lae_probs, axis=1) 
    # Usamos logaritmos para sumar linealmente
    match_log_weights = np.log(match_appeal + 1e-9)
    
    # Calcular los límites del Score (Min y Max posible)
    # Ordenamos los pesos para encontrar los 8 mayores y 8 menores
    sorted_weights = np.sort(match_log_weights)
    min_possible_score = np.sum(sorted_weights[:8]) # Los 8 menos populares
    max_possible_score = np.sum(sorted_weights[-8:]) # Los 8 más populares (Favoritos)
    
    print(f"⚖️  Calibrando Masificación:")
    print(f"   Score Favoritos (Max): {max_possible_score:.4f} -> {CROWD_MAX_SHARE*100}% Población")
    print(f"   Score Sorpresas (Min): {min_possible_score:.4f} -> {CROWD_MIN_SHARE*100}% Población")

    # 3. Generar Candidatos
    matches, outcomes = generate_all_candidates()
    
    # 4. Calcular EV
    print("⚡ Calculando EV Heurístico...")
    t_calc = time.time()
    
    evs, probs_real = engine.calculate_elige8_ev_heuristic(
        matches, outcomes, 
        real_probs, 
        ESTIMACION_COLUMNAS, BOTE_REPARTO,
        match_log_weights,
        min_possible_score, max_possible_score,
        CROWD_MAX_SHARE, CROWD_MIN_SHARE
    )
    
    print(f"   Cálculo completado en {time.time() - t_calc:.2f}s")
    
    # 5. Filtrado y Ordenación
    print(f"🎯 Estrategia: {METODO_ORDENACION}")
    
    # Filtro EV Mínimo
    mask = evs >= MIN_EV
    good_indices = np.where(mask)[0]
    print(f"   Apuestas con EV >= {MIN_EV}€: {len(good_indices)}")
    
    top_indices = []
    if len(good_indices) == 0:
        print("   ⚠️ Ninguna cumple el criterio. Mostrando Top Probabilidad...")
        top_indices = np.argsort(probs_real)[::-1][:TOP_N]
    else:
        if METODO_ORDENACION == "PROB_RENTABLE":
            # Ordenar las rentables por probabilidad
            sorted_local = np.argsort(probs_real[good_indices])[::-1]
            top_indices = good_indices[sorted_local[:TOP_N]]
        else:
            # Ordenar por EV
            sorted_local = np.argsort(evs[good_indices])[::-1]
            top_indices = good_indices[sorted_local[:TOP_N]]

    # 6. Informe
    print(f"\n🏆 TOP {TOP_N} ELIGE 8")
    print(f"{'#':<4} {'PARTIDOS (Indices+1)':<25} {'PRONÓSTICO':<12} {'PROB REAL':<12} {'PREMIO EST':<12} {'EV (€)':<10}")
    print("-" * 95)
    
    sym = ["1", "X", "2"]
    for i, idx in enumerate(top_indices):
        m_idxs = matches[idx]
        res_idxs = outcomes[idx]
        
        str_partidos = str(m_idxs + 1)
        str_prono = "".join([sym[r] for r in res_idxs])
        val_prob = probs_real[idx]
        val_ev = evs[idx]
        val_premio = val_ev / max(1e-12, val_prob)
        
        mark = "💎" if val_ev > 2.0 else ""
        print(f"{i+1:<4} {str_partidos:<25} {str_prono:<12} {val_prob*100:.4f}%     {val_premio:.2f} €       {val_ev:.4f} {mark}")

    print("-" * 95)

if __name__ == "__main__":
    main()