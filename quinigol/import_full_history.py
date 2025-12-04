import os
import numpy as np

# --- CONFIGURACIÓN ---
FILE_LAE = 'lae.txt'
FILE_REAL = 'real.txt'
OUTPUT_DIR = os.path.join('quinigol', 'data', 'historical')

def parse_file_content(filename):
    """Lee un archivo de texto y devuelve una lista de jornadas (cada una con 6 partidos)."""
    try:
        with open(filename, 'r') as f:
            content = f.read().strip()
    except FileNotFoundError:
        print(f"❌ Error: No encuentro '{filename}'.")
        return None

    # Separar por bloques (doble salto de línea suele separar jornadas en tu formato)
    blocks = content.split('\n\n')
    blocks = [b for b in blocks if b.strip()]
    
    parsed_jornadas = []
    
    for i, block in enumerate(blocks):
        lines = block.strip().split('\n')
        lines = [l.strip() for l in lines if l.strip()]
        
        if len(lines) != 48:
            print(f"⚠️ {filename}: Jornada {i+1} tiene {len(lines)} líneas (se esperaban 48). Saltando.")
            parsed_jornadas.append(None)
            continue
            
        matches = []
        for m in range(6):
            # Bloque de 8 líneas por partido
            # Formato input: L0, V0, L1, V1, L2, V2, LM, VM
            raw = lines[m*8 : (m+1)*8]
            try:
                vals = [float(x.replace(',', '.')) for x in raw]
                
                # Reordenar a: L0, L1, L2, LM, V0, V1, V2, VM
                # Indices input: 0,  1,  2,  3,  4,  5,  6,  7
                # Indices target:0,  2,  4,  6,  1,  3,  5,  7
                p_local = [vals[0], vals[2], vals[4], vals[6]]
                p_visit = [vals[1], vals[3], vals[5], vals[7]]
                
                matches.append(p_local + p_visit)
            except ValueError:
                print(f"❌ Error numérico en {filename} Jornada {i+1}")
                matches = None
                break
        
        parsed_jornadas.append(matches)
        
    return parsed_jornadas

def generate_files():
    print("--- IMPORTADOR DE HISTÓRICO ---")
    
    data_lae = parse_file_content(FILE_LAE)
    data_real = parse_file_content(FILE_REAL)
    
    if not data_lae or not data_real:
        print("❌ Faltan datos. Abortando.")
        return

    # Asegurar directorio
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Procesar jornadas (usamos el mínimo de longitud de ambos archivos)
    count = min(len(data_lae), len(data_real))
    print(f"procesando {count} jornadas coincidentes...")
    
    for i in range(count):
        jornada_num = i + 1
        matches_lae = data_lae[i]
        matches_real = data_real[i]
        
        if matches_lae is None or matches_real is None:
            continue
            
        # Generar contenido Python
        content = f"""import numpy as np

# --- HISTORICAL DATA JORNADA {jornada_num} ---
# Auto-generated from lae.txt and real.txt

JORNADA_ID = "{jornada_num}"

# FINANCIALS (Placeholder - Debes rellenar esto si quieres backtest de dinero exacto)
JACKPOT = 0.0 
ESTIMATION = 100000.0

# RESULTADO GANADOR (Placeholder - Debes rellenar esto)
# Formato Indices 0-15: [ResP1, ResP2, ResP3, ResP4, ResP5, ResP6]
WINNING_COMBINATION = [0, 0, 0, 0, 0, 0] 

# --- RAW DATA (PERCENTAGES) ---
# Format: [L0, L1, L2, LM, V0, V1, V2, VM]
RAW_LAE_DATA = {matches_lae}
RAW_REAL_DATA = {matches_real}

# --- MATRIX GENERATION ---

# 1. LAE PROBS MATRIX
LAE_PROBS_MATRIX = np.zeros((6, 16), dtype=np.float64)
for i in range(6):
    probs = np.array(RAW_LAE_DATA[i]) / 100.0
    p_local = probs[0:4]
    p_visit = probs[4:8]
    # Normalize
    if np.sum(p_local) > 0: p_local /= np.sum(p_local)
    if np.sum(p_visit) > 0: p_visit /= np.sum(p_visit)
    LAE_PROBS_MATRIX[i, :] = np.outer(p_local, p_visit).flatten()

# 2. REAL PROBS MATRIX
# Note: real.txt contained percentages, not odds. So we divide by 100.
REAL_PROBS_MATRIX = np.zeros((6, 16), dtype=np.float64)
for i in range(6):
    probs = np.array(RAW_REAL_DATA[i]) / 100.0
    p_local = probs[0:4]
    p_visit = probs[4:8]
    # Normalize
    if np.sum(p_local) > 0: p_local /= np.sum(p_local)
    if np.sum(p_visit) > 0: p_visit /= np.sum(p_visit)
    REAL_PROBS_MATRIX[i, :] = np.outer(p_local, p_visit).flatten()
"""
        
        filename = f"jornada_{str(jornada_num).zfill(2)}.py"
        with open(os.path.join(OUTPUT_DIR, filename), 'w') as f:
            f.write(content)
            
    print(f"✅ Éxito. Se han creado {count} archivos en '{OUTPUT_DIR}'.")
    print("👉 RECUERDA: Tienes que abrir cada archivo y poner manualmente 'WINNING_COMBINATION' si quieres comprobar aciertos.")

if __name__ == "__main__":
    generate_files()