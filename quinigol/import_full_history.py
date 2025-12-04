import os
import numpy as np

# --- CONFIGURACIÓN ---
FILE_LAE = 'lae.txt'
FILE_REAL = 'real.txt'
OUTPUT_DIR = os.path.join('quinigol', 'data', 'historical_new')

def parse_file_content(filename):
    """Lee el archivo y extrae bloques de 48 líneas (6 partidos * 8 datos)."""
    try:
        with open(filename, 'r') as f:
            content = f.read().strip()
    except FileNotFoundError:
        print(f"❌ Error: No encuentro '{filename}'.")
        return None

    # Separamos por doble salto de línea (bloques de jornadas)
    blocks = content.split('\n\n')
    blocks = [b for b in blocks if b.strip()]
    
    parsed_jornadas = []
    print(f"📂 {filename}: Encontrados {len(blocks)} bloques.")

    for i, block in enumerate(blocks):
        lines = block.strip().split('\n')
        lines = [l.strip() for l in lines if l.strip()]
        
        if len(lines) != 48:
            print(f"⚠️ {filename}: Jornada {i+1} tiene {len(lines)} líneas (se esperaban 48). Saltando.")
            parsed_jornadas.append(None)
            continue
            
        matches = []
        # Procesar los 6 partidos de la jornada
        for m in range(6):
            # Cogemos las 8 líneas de este partido
            # Tu formato: L0, V0, L1, V1, L2, V2, LM, VM
            raw = lines[m*8 : (m+1)*8]
            try:
                v = [float(x.replace(',', '.')) for x in raw]
                
                # Reordenamos para tener: [L0, L1, L2, LM, V0, V1, V2, VM]
                # Índices Input: 0(L0), 1(V0), 2(L1), 3(V1), 4(L2), 5(V2), 6(LM), 7(VM)
                p_local = [v[0], v[2], v[4], v[6]]
                p_visit = [v[1], v[3], v[5], v[7]]
                
                matches.append(p_local + p_visit)
            except ValueError:
                print(f"❌ Error numérico en {filename} Jornada {i+1}")
                matches = None
                break
        
        parsed_jornadas.append(matches)
    return parsed_jornadas

def generate_files():
    print("--- GENERADOR DE HISTÓRICO (LAE + REAL) ---")
    
    data_lae = parse_file_content(FILE_LAE)
    data_real = parse_file_content(FILE_REAL)
    
    if not data_lae or not data_real:
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    count = min(len(data_lae), len(data_real))
    print(f"🔄 Generando {count} archivos de jornada...")
    
    for i in range(count):
        j_num = i + 1
        # Datos brutos
        lae_data = data_lae[i]
        real_data = data_real[i]
        
        if not lae_data or not real_data: continue
        
        # Contenido del archivo .py
        content = f"""import numpy as np

# --- JORNADA {j_num} ---
JORNADA_ID = "{j_num}"

# 1. DATOS FINANCIEROS (A RELLENAR)
# Copia aquí el escrutinio real de la web de loterías
JACKPOT = 0.0          # Bote que había esa jornada
ESTIMATION = 100000.0  # Recaudación real

# 2. RESULTADO GANADOR (A RELLENAR)
# Índices: 0(0-0), 1(0-1), 2(0-2), 3(0-M), 4(1-0)... 15(M-M)
WINNING_COMBINATION = [0, 0, 0, 0, 0, 0] 

# 3. ESCRUTINIO REAL (PREMIOS PAGADOS)
# Rellena el premio que cobró cada categoría. 
# Si no hubo acertantes, pon 0.0.
REAL_PRIZES = {{
    6: 0.0,  # Premio unitario para 6 aciertos
    5: 0.0,  # Premio unitario para 5 aciertos
    4: 0.0,  # Premio unitario para 4 aciertos
    3: 0.0,  # Premio unitario para 3 aciertos
    2: 0.0   # Premio unitario para 2 aciertos
}}

# --- DATOS MATRICIALES (GENERADOS AUTOMÁTICAMENTE) ---

# Porcentajes LAE
RAW_LAE_DATA = {lae_data}

# Porcentajes REALES (Transformados de % a Probabilidad 0-1)
RAW_REAL_DATA = {real_data}

# Construcción Matrices
LAE_PROBS_MATRIX = np.zeros((6, 16), dtype=np.float64)
REAL_PROBS_MATRIX = np.zeros((6, 16), dtype=np.float64)

for i in range(6):
    # LAE
    probs_l = np.array(RAW_LAE_DATA[i]) / 100.0
    pl, pv = probs_l[0:4], probs_l[4:8]
    if np.sum(pl) > 0: pl /= np.sum(pl)
    if np.sum(pv) > 0: pv /= np.sum(pv)
    LAE_PROBS_MATRIX[i, :] = np.outer(pl, pv).flatten()
    
    # REAL
    probs_r = np.array(RAW_REAL_DATA[i]) / 100.0
    prl, prv = probs_r[0:4], probs_r[4:8]
    if np.sum(prl) > 0: prl /= np.sum(prl)
    if np.sum(prv) > 0: prv /= np.sum(prv)
    REAL_PROBS_MATRIX[i, :] = np.outer(prl, prv).flatten()
"""
        
        # Guardar archivo
        filename = f"jornada_{str(j_num).zfill(2)}.py"
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
            
    print(f"✅ ¡Listo! Archivos guardados en {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_files()