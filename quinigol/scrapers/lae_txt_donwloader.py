import requests
import datetime
import urllib.parse
import re

# --- CONFIGURATION / CONFIGURACIÓN ---
# Ruta base confirmada
BASE_PATH = "https://www.loteriasyapuestas.es/f/loterias/documentos/El%20Quinigol/Estad%C3%ADsticas%20de%20pron%C3%B3sticos"

# Estos se usarán solo si el usuario deja el input vacío
DEFAULT_JACKPOT = 0.0
DEFAULT_ESTIMATION = 100000.0

def get_current_season_strings():
    """Calcula las cadenas de temporada (Ej: 2025_2026 y 25-26)."""
    today = datetime.date.today()
    year = today.year
    month = today.month
    
    if month >= 8: # Inicio de temporada (Ago-Dic)
        y1, y2 = year, year + 1
    else: # Fin de temporada (Ene-Jul)
        y1, y2 = year - 1, year
        
    return {
        "FULL": f"{y1}_{y2}",
        "SHORT": f"{str(y1)[-2:]}-{str(y2)[-2:]}"
    }

def download_lae_txt(jornada):
    """Descarga el archivo de pronósticos de LAE."""
    season = get_current_season_strings()
    jornada_str = str(jornada).zfill(2)
    
    suffixes = ["_M", "", "_V", "_S"]
    
    print(f"--- Buscando Jornada {jornada} (Temp. {season['SHORT']}) ---")
    
    for suffix in suffixes:
        filename = f"pronosQNG_{jornada_str}_{season['SHORT']}{suffix}.txt"
        season_path = urllib.parse.quote(f"Temporada {season['FULL']}")
        url = f"{BASE_PATH}/{season_path}/{filename}"
        
        try:
            print(f"Probando: {filename} ... ", end="")
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36"}
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                print("✅ ¡ENCONTRADO!")
                # Intentamos decodificar, latin-1 es habitual en txt viejos de España
                try: return response.content.decode('latin-1')
                except: return response.content.decode('utf-8')
            else:
                print(f"❌")
                
        except Exception as e:
            print(f"Error conexión: {e}")

    print("\n❌ No se encontró el archivo. ¿Seguro que LAE lo ha subido ya?")
    return None

def parse_txt_content(raw_text):
    """Extrae los porcentajes usando expresiones regulares (más robusto)."""
    lines = raw_text.strip().split('\n')
    all_probs_rows = []
    
    for line in lines:
        # 1. Limpieza básica
        clean_line = line.strip().replace(',', '.') # Coma decimal a punto
        
        # 2. Extraer TODOS los números (enteros o flotantes)
        # Regex captura: 13.3, 5, 0.09, 1, 2...
        found_numbers = re.findall(r"(\d+(?:\.\d+)?)", clean_line)
        
        # 3. Convertir a float
        valid_floats = []
        for num in found_numbers:
            try:
                val = float(num)
                valid_floats.append(val)
            except: continue
            
        # 4. LÓGICA DE DETECCIÓN MEJORADA
        # Las líneas suelen tener índices al principio (1;1;LEGANÉS...)
        # Lo que nos importa son los ÚLTIMOS 4 números (0, 1, 2, M)
        
        if len(valid_floats) >= 4:
            # Cogemos los últimos 4 candidatos
            candidates = valid_floats[-4:]
            
            # Verificación: ¿Suman ~100? (Margen de error de 2.0 por redondeos)
            total_sum = sum(candidates)
            if 98.0 <= total_sum <= 102.0:
                # Son porcentajes válidos! Normalizamos (0-1) y guardamos
                probs_norm = [x/100.0 for x in candidates]
                all_probs_rows.append(probs_norm)
                # print(f"DEBUG: Fila aceptada -> {candidates}") # Descomentar para debug

    return all_probs_rows

def generate_current_data_file(probs_rows, jornada, jackpot, estimation):
    """Crea el archivo current_data.py con SOLO los datos de LAE y financieros."""
    output_file = "quinigol/current_data.py"
    
    # Formatear la lista para escribirla como código Python
    probs_string = "[\n"
    for row in probs_rows:
        probs_string += f"    {row},\n"
    probs_string += "]"

    content = f"""# --- AUTOMATED DATA FILE (LAE TXT SOURCE) ---
# Jornada: {jornada}
# Generated: {datetime.datetime.now()}

import numpy as np

# --- FINANCIALS ---
JACKPOT = {jackpot}
ESTIMATION = {estimation}
BET_PRICE = 1.0

# --- LAE PROBABILITIES ---
LAE_PROBS_MATRIX = np.zeros((6, 16), dtype=np.float64)

# Raw data from TXT
raw_txt_data = {probs_string}

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
    print(f"Error procesando matriz LAE: {{e}}")

# NOTE: Real probabilities are managed in the main optimizer script manually.
"""
    
    with open(output_file, "w") as f:
        f.write(content)
    print(f"✅ Archivo generado: '{output_file}'")

if __name__ == "__main__":
    print("--- DESCARGADOR AUTOMÁTICO LAE (TXT PARSER) ---")
    try:
        # INPUTS POR CONSOLA
        j_input = input("Jornada: ")
        jornada = int(j_input)
        
        jack_input = input(f"Bote (Enter para default {DEFAULT_JACKPOT}): ")
        est_input = input(f"Estimación Recaudación (Enter para default {DEFAULT_ESTIMATION}): ")
        
        # Gestión de valores por defecto si se pulsa Enter
        jackpot = float(jack_input) if jack_input.strip() else DEFAULT_JACKPOT
        estimation = float(est_input) if est_input.strip() else DEFAULT_ESTIMATION
        
        raw_content = download_lae_txt(jornada)
        
        if raw_content:
            parsed_data = parse_txt_content(raw_content)
            print(f"📊 Filas válidas extraídas: {len(parsed_data)} (Esperadas: 12)")
            
            if len(parsed_data) > 0:
                if len(parsed_data) != 12:
                    print("⚠️ ¡OJO! No se encontraron exactamente 12 filas. Revisa el archivo generado.")
                
                generate_current_data_file(parsed_data, jornada, jackpot, estimation)
            else:
                print("❌ No se encontró ninguna línea de datos válida.")
                print("   (Verifica que el archivo TXT tenga porcentajes que sumen ~100)")
                
    except ValueError:
        print("❌ Error: Introduce números válidos (usa punto para decimales en el dinero).")