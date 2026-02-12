import requests
import xml.etree.ElementTree as ET
import numpy as np
import datetime

# --- CONFIGURACIÓN ---
TEMPORADA = "2026"
JORNADA = "41"
JACKPOT_DEFECTO = 0.00
ESTIMACION_DEFECTO = 1100000.0


def odds_to_probs(odds):
    """Convierte cuotas a probabilidad real normalizada."""
    inv = [1.0/x for x in odds]
    tot = sum(inv)
    return [x/tot for x in inv]

def generar_fichero_datos():
    print(f"--- GENERADOR DE DATOS (Jornada {JORNADA}) ---")
    
    # 1. DESCARGA 1X2 (API)
    url = f"https://api.eduardolosilla.es/servicios/v1/probabilidad_real?jornada={JORNADA}&temporada={TEMPORADA}"
    print(f"🌍 Descargando 1X2: {url}")
    
    probs_1x2_list = []
    try:
        r = requests.get(url, timeout=10)
        root = ET.fromstring(r.content)
        temp = {}
        for elem in root:
            for part in elem:
                num = part.get("num")
                if num == "15": continue
                idx = int(num)
                # API 0-100 -> 0-1
                p1 = float(part.get("porcDec_1","0").replace(",",".")) / 100.0
                px = float(part.get("porcDec_X","0").replace(",",".")) / 100.0
                p2 = float(part.get("porcDec_2","0").replace(",",".")) / 100.0
                temp[idx] = [p1, px, p2]
        
        for i in range(1, 15):
            probs_1x2_list.append(temp.get(i, [0.33, 0.33, 0.33]))
            
    except Exception as e:
        print(f"❌ Error descargando: {e}")
        return

    # 2. PROCESAR PLENO AL 15 (AUTOMÁTICO SIEMPRE)
    # Usamos las cuotas base para definir la probabilidad real del suceso
    print("ℹ️  Calculando probabilidades del Pleno según cuotas de mercado.")
    raw_odds = [11.00,8.00,9.00,9.00,15.00,7.50,9.00,8.50,29.00,17.00,17.00,15.00,67.00,34.00,36.00,29.00]
    probs_pleno = odds_to_probs(raw_odds)

    # 3. ESCRIBIR current_data.py
    with open("quiniela/data/current_data.py", "w", encoding="utf-8") as f:
        f.write(f"# Generado autom. el {datetime.datetime.now()}\n")
        f.write("import numpy as np\n\n")
        f.write(f"TEMPORADA = '{TEMPORADA}'\nJORNADA = '{JORNADA}'\n")
        f.write(f"JACKPOT = {JACKPOT_DEFECTO}\nESTIMATION = {ESTIMACION_DEFECTO}\n\n")
        
        f.write("# Probabilidades Reales 1X2 (14 partidos)\n")
        f.write("REAL_1X2 = np.array([\n")
        for row in probs_1x2_list:
            f.write(f"    [{row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f}],\n")
        f.write("], dtype=np.float64)\n\n")
        
        f.write("# Probabilidades Reales Pleno al 15 (16 resultados)\n")
        f.write("REAL_PLENO = np.array([\n    ")
        for val in probs_pleno:
            f.write(f"{val:.6f}, ")
        f.write("\n], dtype=np.float64)\n")
        
    print("✅ Archivo 'current_data.py' creado con éxito.")

if __name__ == "__main__":
    generar_fichero_datos()