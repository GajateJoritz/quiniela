# --- AUTOMATED DATA FILE (LAE TXT SOURCE) ---
# Jornada: 29
# Generated: 2025-12-03 14:16:21.437882

import numpy as np

# --- FINANCIALS ---
JACKPOT = 0.0
ESTIMATION = 100000.0
BET_PRICE = 1.0

# --- LAE PROBABILITIES ---
LAE_PROBS_MATRIX = np.zeros((6, 16), dtype=np.float64)

# Raw data from TXT
raw_txt_data = [
    [0.133, 0.377, 0.371, 0.11900000000000001],
    [0.326, 0.441, 0.182, 0.0509999999999999],
    [0.545, 0.363, 0.059000000000000004, 0.033],
    [0.045, 0.098, 0.23600000000000002, 0.621],
    [0.494, 0.39399999999999996, 0.079, 0.033],
    [0.049, 0.10099999999999999, 0.24100000000000002, 0.609],
    [0.449, 0.43799999999999994, 0.081, 0.032],
    [0.049, 0.12300000000000001, 0.313, 0.515],
    [0.057999999999999996, 0.217, 0.364, 0.361],
    [0.166, 0.392, 0.315, 0.127],
    [0.251, 0.489, 0.21100000000000002, 0.0490000000000001],
    [0.067, 0.25, 0.376, 0.307],
]

try:
    for i in range(6):
        # Aseguramos que existen datos para el partido i
        if (i*2 + 1) < len(raw_txt_data):
            p_local = np.array(raw_txt_data[i*2])
            p_visit = np.array(raw_txt_data[i*2 + 1])
            
            # Normalizar para que sume 1.0 exacto
            if np.sum(p_local) > 0: p_local /= np.sum(p_local)
            if np.sum(p_visit) > 0: p_visit /= np.sum(p_visit)
            
            # Producto Cartesiano (Hipótesis Independencia LAE)
            LAE_PROBS_MATRIX[i, :] = np.outer(p_local, p_visit).flatten()
except Exception as e:
    print(f"Error procesando matriz LAE: {e}")

