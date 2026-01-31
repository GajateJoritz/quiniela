import requests
import xml.etree.ElementTree as ET
import numpy as np

# --- CONFIGURACIÓN DE LA JORNADA A DESCARGAR ---
# Valores por defecto si no se especifican argumentos
TEMPORADA_DEFECTO = "2026" 
JORNADA_DEFECTO = "39" 

def get_real_data(temporada=TEMPORADA_DEFECTO, jornada=JORNADA_DEFECTO):
    """
    Descarga los porcentajes reales de la API de Eduardo Losilla para los 14 partidos.
    Para el Pleno al 15, utiliza una distribución hardcodeada (Opción C).
    
    Retorna:
        - probs_1x2: Matriz (14, 3) con probabilidades reales.
        - probs_pleno: Array (16,) con probabilidades del pleno.
    """
    url = f"https://api.eduardolosilla.es/servicios/v1/probabilidad_real?jornada={jornada}&temporada={temporada}"
    print(f"🌍 Conectando a API: {url}")
    
    # 1. DESCARGA Y PARSEO DE 14 PARTIDOS
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception as e:
        print(f"❌ Error descargando/procesando datos: {e}")
        return None, None

    # Matriz para los 14 partidos
    probs_1x2 = np.zeros((14, 3), dtype=np.float64)
    count = 0
    
    for elem in root:
        for part in elem:
            num_str = part.get("num")
            if num_str == "15":
                continue 
            
            idx = int(num_str) - 1
            if 0 <= idx < 14:
                p1 = float(part.get("porcDec_1", 0).replace(",", "."))
                px = float(part.get("porcDec_X", 0).replace(",", "."))
                p2 = float(part.get("porcDec_2", 0).replace(",", "."))
                
                probs_1x2[idx] = [p1/100.0, px/100.0, p2/100.0]
                count += 1

    if count < 14:
        print(f"⚠️ Alerta: Solo se encontraron datos para {count} partidos.")
    else:
        print(f"✅ Datos reales descargados correctamente para 14 partidos.")

    # 2. CONFIGURACIÓN DEL PLENO AL 15 (OPCIÓN C)
    # Datos hardcodeados de tu quiniela.py original
    print("ℹ️ Usando porcentajes fijos (Opción C) para el Pleno al 15.")
    
    raw_pleno = [34,23,26,34,17,10,13,13,17,10,11,12,13,7.5,8,8]
    
    # Convertir a tanto por uno
    probs_pleno = np.array([x/100.0 for x in raw_pleno], dtype=np.float64)
    
    # Asegurar que suman 1.0 (Normalización por seguridad)
    suma = np.sum(probs_pleno)
    if abs(suma - 1.0) > 0.001:
        probs_pleno = probs_pleno / suma

    return probs_1x2, probs_pleno

if __name__ == "__main__":
    p14, pp = get_real_data()
    if p14 is not None:
        print("\nEjemplo Partido 1 (1X2):", p14[0])
        print("Ejemplo Pleno (Suma):", np.sum(pp))
        print("Valores Pleno:", pp)