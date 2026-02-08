def calcular_cuotas_120(cuotas_90, combos):
    """
    cuotas_90: Lista de 16 floats
    combos: Diccionario con listas [GanaLocalET, GanaVisitanteET, GanaLocalPens, GanaVisitantePens]
            Claves: '0-0', '1-1', '2-2'
    """
    # 1. Convertir cuotas a probabilidad y normalizar (quitar margen bookie)
    probs_90 = [1/c for c in cuotas_90]
    total_prob = sum(probs_90)
    probs_90 = [p/total_prob for p in probs_90]
    
    # Matriz destino (copia inicial a 0)
    probs_120 = [0.0] * 16
    
    # Índices en tu array plano de 16
    idx_map = {
        '0-0': 0, '0-1': 1, '1-0': 4, 
        '1-1': 5, '1-2': 6, '2-1': 9,
        '2-2': 10, '2-M': 11, 'M-2': 14 
    }
    
    # Indices de los empates que vamos a procesar
    draw_indices = {0: '0-0', 5: '1-1', 10: '2-2'}

    for i in range(16):
        if i in draw_indices:
            # Es un empate, hay que redistribuirlo
            key = draw_indices[i]
            prob_base = probs_90[i]
            
            # Pesos de los escenarios futuros
            odds_escenarios = combos[key] # [AlavesET, RealET, AlavesPens, RealPens]
            inv_odds = [1/x for x in odds_escenarios]
            sum_inv = sum(inv_odds)
            pesos = [x/sum_inv for x in inv_odds] # Normalizados a 1
            
            # Reparto de masa
            # 1. Gana Local en Prórroga (Suma +1 gol al local)
            idx_local_gana = idx_map.get(key.replace('0-0','1-0').replace('1-1','2-1').replace('2-2','M-2'))
            probs_120[idx_local_gana] += prob_base * pesos[0]
            
            # 2. Gana Visitante en Prórroga (Suma +1 gol al visitante)
            idx_visit_gana = idx_map.get(key.replace('0-0','0-1').replace('1-1','1-2').replace('2-2','2-M'))
            probs_120[idx_visit_gana] += prob_base * pesos[1]
            
            # 3. Penaltis (Se queda el marcador igual)
            # Sumamos pesos[2] y pesos[3] porque ambos significan empate al 120'
            probs_120[i] += prob_base * (pesos[2] + pesos[3])
            
        else:
            # Resultados que no son empate a 90' se quedan igual
            probs_120[i] += probs_90[i]
            
    # Convertir prob final a cuota
    return [round(1/p, 2) if p > 0 else 0 for p in probs_120]

# TUS DATOS DE ENTRADA
cuotas_input = [7.00,7.00,11.00,21.00,7.50,6.00,10.00,19.00,13.00,12.00,17.00,29.00,26.00,23.00,29.00,56.00]
combos_input = {
    '0-0': [29, 26, 21, 21],
    '1-1': [26, 21, 17, 17],
    '2-2': [61, 61, 46, 51]
}

# Ejecutar
resultado = calcular_cuotas_120(cuotas_input, combos_input)
# print(resultado)