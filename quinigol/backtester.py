import os
import sys
import time
import numpy as np
import importlib.util

# Añadir el path para importar módulos internos
sys.path.append(os.path.join(os.path.dirname(__file__), 'quinigol'))
import src.core_math as engine

# --- CONFIGURACIÓN BACKTEST ---
HISTORICAL_FOLDER = "quinigol/data/historical"
RESULTS_FILE = "results/backtest_results.txt"

# Configuración del Optimizador (Debe coincidir con tu estrategia real)
PORTFOLIO_SIZE = 10
N_SIMULATIONS = 5000000 # Un poco menos que en producción para ir rápido, o 5M para rigor
OPTIMIZATION_MODE = 1   # Sortino

def load_historical_module(filepath):
    """Carga dinámicamente un archivo .py de jornada como un módulo."""
    spec = importlib.util.spec_from_file_location("hist_module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def calculate_real_prizes(jornada_data):
    """
    Calcula o recupera los premios REALES que se pagaron esa jornada.
    Si el archivo de jornada tiene el escrutinio real (premios exactos), úsalo.
    Si no, los estimamos basándonos en la recaudación real.
    """
    # En una implementación perfecta, tu archivo jornada_XX.py tendría:
    # REAL_PRIZES = {6: 250000, 5: 450, 4: 30, ...}
    
    # Si no los tenemos, usamos la estimación matemática sobre la recaudación REAL
    # (Esto es una aproximación muy fiel si no tienes el dato exacto del premio en euros)
    est = jornada_data.ESTIMATION
    jackpot = jornada_data.JACKPOT
    dist = np.array([0.10, 0.09, 0.08, 0.08, 0.20])
    
    prizes = {}
    prizes[6] = engine.calc_taxed_prize(jackpot + est * dist[0]) # Aprox: Bote + 10%
    
    # Para categorías inferiores, el premio depende de cuánta gente acertó.
    # En backtesting riguroso, deberíamos saber el nº de acertantes reales.
    # Si no lo tienes, usamos una estimación estándar de dificultad basada en LAE.
    
    # NOTA: Para empezar, usaremos la estimación de dificultad del propio escenario ganador.
    # Es decir: "Si salió este resultado, ¿cuánto debería pagar según la matemática?"
    # Esto elimina la suerte de "hubo 3 acertantes o 4", y te da el "valor justo".
    
    return prizes

def run_backtest():
    print(f"🚀 INICIANDO BACKTEST (Estrategia: {OPTIMIZATION_MODE})")
    print(f"📂 Leyendo carpeta: {HISTORICAL_FOLDER}")
    
    files = [f for f in os.listdir(HISTORICAL_FOLDER) if f.startswith("jornada_") and f.endswith(".py")]
    files.sort() # Ordenar cronológicamente (01, 02...)
    
    total_invested = 0.0
    total_won = 0.0
    history_log = []
    
    for f_name in files:
        path = os.path.join(HISTORICAL_FOLDER, f_name)
        print(f"\n>>> Procesando {f_name}...")
        
        # 1. Cargar datos de la jornada (El "Pasado")
        data = load_historical_module(path)
        
        # Validar que tenga resultado ganador para comprobar
        if not hasattr(data, 'WINNING_COMBINATION'):
            print("   ⚠️ Saltando: No tiene WINNING_COMBINATION definida.")
            continue
            
        winning_comb = np.array(data.WINNING_COMBINATION, dtype=np.int32)
        
        # 2. Ejecutar el Optimizador (El "Cerebro")
        # Filtro
        evs = engine.get_top_candidates(
            np.load("quinigol/combinations/kinigmotza.npy").astype(np.int32), # Combinaciones
            data.REAL_PROBS_MATRIX, 
            data.LAE_PROBS_MATRIX, 
            data.ESTIMATION, 
            data.JACKPOT, 
            np.array([0.10, 0.09, 0.08, 0.08, 0.20])
        )
        
        # Selección de candidatos
        # (Simplificado: Cogemos top 2000 por EV para la simulación)
        top_indices = evs.argsort()[::-1][:2000]
        top_combs = np.load("quinigol/combinations/kinigmotza.npy").astype(np.int32)[top_indices]
        
        # Simulación Montecarlo
        scenarios = engine.generate_scenarios(data.REAL_PROBS_MATRIX, N_SIMULATIONS)
        dyn_prizes = engine.precompute_scenario_prizes(scenarios, data.LAE_PROBS_MATRIX, data.ESTIMATION, data.JACKPOT, np.array([0.10, 0.09, 0.08, 0.08, 0.20]))
        hits_matrix = engine.precompute_hits_matrix(top_combs, scenarios)
        
        # Selección Greedy
        from quinigol.quinigol_optimizer import greedy_portfolio_selection
        sel_indices, _ = greedy_portfolio_selection(
            hits_matrix, dyn_prizes, top_indices, 
            np.load("quinigol/combinations/kinigmotza.npy").astype(np.int32), 
            PORTFOLIO_SIZE, OPTIMIZATION_MODE
        )
        
        # 3. Escrutinio Real (La "Verdad")
        # Obtenemos las apuestas seleccionadas
        my_bets = np.load("quinigol/combinations/kinigmotza.npy").astype(np.int32)[top_indices[sel_indices]]
        
        # Comprobamos aciertos contra el resultado REAL de esa jornada
        match_hits = []
        round_winnings = 0.0
        
        # Calcular premios "Justos" para este resultado específico
        # (Usamos la lógica de Poisson inversa: Dado este resultado ganador, ¿cuánto debería pagar?)
        # Esto es mejor que inventar números si no tienes el escrutinio oficial a mano.
        
        # Probabilidad LAE del resultado ganador
        p_lae_winner = 1.0
        for m in range(6): p_lae_winner *= data.LAE_PROBS_MATRIX[m, winning_comb[m]]
        
        # Premios estimados para ESTE resultado
        prize_6 = engine.poisson_prize_share(data.JACKPOT + data.ESTIMATION*0.10, data.ESTIMATION * p_lae_winner)
        # (Podríamos calcular 5, 4... simplificamos asumiendo la dificultad del 6 como proxy)
        
        for bet in my_bets:
            hits = np.sum(bet == winning_comb)
            match_hits.append(hits)
            
            # Asignar premio (Aproximación realista si no tenemos datos oficiales)
            if hits == 6: round_winnings += prize_6
            elif hits == 5: round_winnings += 500.0 # Estimación conservadora
            elif hits == 4: round_winnings += 50.0  # Estimación conservadora
            elif hits == 3: round_winnings += 5.0   # Estimación conservadora
            elif hits == 2: round_winnings += 1.5   # Estimación conservadora
            
        cost = len(my_bets)
        profit = round_winnings - cost
        
        total_invested += cost
        total_won += round_winnings
        
        print(f"   Inversión: {cost}€ | Ganado: {round_winnings:.2f}€ | Balance: {profit:+.2f}€")
        print(f"   Mejores Aciertos: {sorted(match_hits, reverse=True)[:3]}")
        history_log.append((f_name, profit))

    # --- RESUMEN FINAL ---
    print("\n" + "="*40)
    print("RESUMEN DE BACKTESTING")
    print("="*40)
    print(f"Jornadas Analizadas: {len(files)}")
    print(f"Inversión Total:     {total_invested:.2f} €")
    print(f"Retorno Total:       {total_won:.2f} €")
    print(f"BALANCE NETO:        {total_won - total_invested:+.2f} €")
    roi = ((total_won - total_invested) / total_invested) * 100 if total_invested > 0 else 0
    print(f"ROI:                 {roi:+.2f}%")
    print("-" * 40)

if __name__ == "__main__":
    run_backtest()