import time
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# --- CONFIGURATION / CONFIGURACIÓN ---
URL_LAE = "https://www.loteriasyapuestas.es/es/quinigol"
OUTPUT_FILE = "current_data.py"

def setup_driver():
    """
    Configura Chrome con opciones para evitar detección y manejar la carga dinámica de LAE.
    """
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless") # Descomenta para modo invisible, pero LAE a veces bloquea headless
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    # User agent real para que la web no nos trate como robot
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

def clean_float(text):
    """Convierte '1.500.000,50 €' o '27,9 %' a float de Python."""
    if not text: return 0.0
    # Quitamos símbolos de moneda, porcentaje y puntos de miles
    clean = text.replace("€", "").replace("%", "").replace("BOTE", "").strip()
    clean = clean.replace(".", "") # El punto en español es miles, lo quitamos
    clean = clean.replace(",", ".") # La coma es decimal, la pasamos a punto
    try:
        return float(clean)
    except ValueError:
        return 0.0

def scrape_official_lae():
    driver = setup_driver()
    data = {
        "jackpot": 0.0,
        "estimation": 100000.0, # Valor por defecto conservador
        "matches": []
    }

    try:
        print(f"--- Connecting to Official LAE: {URL_LAE} ---")
        driver.get(URL_LAE)
        wait = WebDriverWait(driver, 20)

        # 1. HANDLE COOKIES / GESTIÓN DE COOKIES (Crítico en LAE)
        try:
            # Buscamos el botón de aceptar cookies (suele ser ID 'onetrust-accept-btn-handler')
            print("   Waiting for Cookie consent...")
            cookie_btn = wait.until(EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler")))
            cookie_btn.click()
            print("   Cookies accepted.")
            time.sleep(2) # Esperar a que se cierre el modal
        except:
            print("   No cookie popup found or already accepted.")

        # 2. GET JACKPOT / OBTENER BOTE
        # El bote suele estar en un contenedor grande en la cabecera
        try:
            # Buscamos clases comunes del bote en LAE
            bote_elem = driver.find_element(By.CLASS_NAME, "bote-cifra")
            data["jackpot"] = clean_float(bote_elem.text)
            print(f"💰 Bote detectado: {data['jackpot']}")
        except:
            print("⚠️ No se pudo leer el Bote (usando default 255k).")
            data["jackpot"] = 255000.0

        # 3. GET PERCENTAGES / OBTENER PORCENTAJES
        # En la web nueva de LAE, los % suelen estar en la sección de "Pronósticos"
        # A veces hay que cambiar de pestaña, pero generalmente en la Home de Quinigol
        # salen las barras de progreso.
        
        print("   Scraping matches and percentages...")
        
        # Parseamos el HTML con BeautifulSoup que es más rápido para búsquedas complejas
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # En LAE, cada partido suele ser un contenedor 'region-partido' o similar.
        # Buscamos la tabla de boletos.
        
        # ESTRATEGIA: Buscar los textos de los equipos y las barras de porcentaje.
        # Las barras suelen tener el valor en el atributo 'title' o en texto oculto
        # formato "27,9%"
        
        # Buscamos contenedores de partidos (suelen ser 6)
        # Ajuste específico para la estructura actual:
        # Buscamos elementos que contengan la estructura del boleto
        
        match_containers = soup.find_all(class_="cuerpoRegion")
        
        count = 0
        for container in match_containers:
            if count >= 6: break
            
            # Obtener nombres de equipos
            try:
                # LAE usa clases como 'nombreLocal' y 'nombreVisitante'
                local_name = container.find(class_="nombreLocal").get_text(strip=True)
                visit_name = container.find(class_="nombreVisitante").get_text(strip=True)
            except:
                continue # No es un contenedor de partido válido

            # Obtener Porcentajes
            # LAE suele poner los % en 4 columnas para local y 4 para visitante (0,1,2,M)
            # Buscamos los textos con decimales
            
            # Truco: Buscamos todos los textos que parezcan porcentajes dentro de este partido
            # Regex: Buscar números con coma (ej: 27,9)
            texts = container.find_all(string=re.compile(r"\d+,\d+"))
            
            # Filtramos solo los que parecen probabilidades válidas (suman ~100)
            probs_found = []
            for t in texts:
                val = clean_float(t)
                # Filtramos valores absurdos (a veces el bote aparece aquí)
                if 0 <= val <= 100:
                    probs_found.append(val)
            
            # Necesitamos exactamente 8 valores (L0,L1,L2,LM, V0,V1,V2,VM)
            # A veces LAE muestra 16 (repite barras), cogemos los únicos o los primeros 8
            
            final_probs = []
            if len(probs_found) >= 8:
                # A veces el orden es confuso. En LAE visualmente es:
                # Local: 0 1 2 M  | Visitante: 0 1 2 M
                # Asumimos que el scraping lee en orden de lectura (izquierda a derecha, arriba abajo)
                final_probs = probs_found[:8]
            else:
                print(f"⚠️ Warning: Match {local_name}-{visit_name} only has {len(probs_found)} percentages. Check selectors.")
                final_probs = [25.0] * 8 # Dummy distribution
            
            count += 1
            
            match_data = {
                "id": count,
                "teams": f"{local_name} vs {visit_name}",
                "lae_probs": final_probs
            }
            data["matches"].append(match_data)
            print(f"   Match {count}: {match_data['teams']} -> {final_probs}")

    except Exception as e:
        print(f"❌ Error scraping LAE: {e}")
        
    finally:
        driver.quit()
        return data

def generate_python_file(data):
    """Genera el archivo current_data.py compatible con tu optimizador."""
    
    content = f"""# --- AUTOMATED DATA FILE (OFFICIAL LAE SOURCE) ---
# Date: {time.strftime("%Y-%m-%d %H:%M:%S")}

import numpy as np

# --- FINANCIALS ---
JACKPOT = {data['jackpot']}
ESTIMATION = {data['estimation']} # Default/Estimated
BET_PRICE = 1.0

# --- LAE PROBABILITIES (PUBLIC BETTING) ---
# Source: LoteriasYapuestas.es (High Precision Decimals)
"""
    
    for m in data['matches']:
        content += f"# Match {m['id']}: {m['teams']}\n"
    
    content += "\nRAW_LAE_DATA = [\n"
    for m in data['matches']:
        probs = m['lae_probs']
        content += f"    {probs}, \n"
    content += "]\n\n"

    # Plantilla para Cuotas Reales (Esto lo rellenas tú a mano)
    content += """# --- REAL PROBABILITIES (BOOKMAKERS) ---
# FILL THIS MANUALLY with data from Flashscore/Bet365
# Format: [0-0, 0-1, 0-2, 0-M, 1-0, 1-1, 1-2, 1-M, 2-0, 2-1, 2-2, 2-M, M-0, M-1, M-2, M-M]

real_probs_1 = [19,9.5,8,4.75,21,10,9,5.5,41,19,17,11,91,51,36,19] 
real_probs_2 = [13,7.5,7.5,6,15,8,8.5,7,34,19,17,15,96,56,46,34] 
real_probs_3 = [8,15,41,161,5.5,8,23,81,5.5,9,26,76,5.5,9,26,67] 
real_probs_4 = [8,11,23,71,5.5,7,17,51,7,9,21,51,9,11,23,56] 
real_probs_5 = [13,9,10,15,11,7,9,10,17,11,13,15,23,17,21,29] 
real_probs_6 = [29,41,51,201,13,15,29,61,10,9.5,19,46,4,3.75,7.5,15] 

raw_real_probs = [real_probs_1, real_probs_2, real_probs_3, real_probs_4, real_probs_5, real_probs_6]

# --- MATRIX GENERATION ---
REAL_PROBS_MATRIX = np.zeros((6, 16), dtype=np.float64)
for i in range(6):
    row = np.array(raw_real_probs[i], dtype=np.float64)
    inv_row = 1.0 / row
    REAL_PROBS_MATRIX[i, :] = inv_row / np.sum(inv_row)

LAE_PROBS_MATRIX = np.zeros((6, 16), dtype=np.float64)
for i in range(6):
    # LAE gives 8 values: [L0,L1,L2,LM, V0,V1,V2,VM] in percentages (0-100)
    probs_8 = np.array(RAW_LAE_DATA[i]) / 100.0 
    p_local = probs_8[0:4]
    p_visit = probs_8[4:8]
    
    # Normalize to ensure sum is exactly 1.0
    if np.sum(p_local) > 0: p_local /= np.sum(p_local)
    if np.sum(p_visit) > 0: p_visit /= np.sum(p_visit)
    
    # Reconstruct 16 probabilities (Hypothesis: Independence)
    # P(Result) = P(Local_Goals) * P(Visit_Goals)
    matrix_16 = np.outer(p_local, p_visit).flatten()
    LAE_PROBS_MATRIX[i, :] = matrix_16
"""

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n✅ File generated: {OUTPUT_FILE}")

if __name__ == "__main__":
    data = scrape_official_lae()
    # Solo generamos archivo si encontramos datos, para no sobrescribir con ceros
    if len(data["matches"]) > 0:
        generate_python_file(data)
    else:
        print("❌ No matches found. File not updated.")