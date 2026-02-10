import numpy as np
import re
import sys
import os

# -------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE RUTAS E IMPORTACIONES
# -------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    import src.core_math_quiniela as engine
    print("✅ Motor matemático (src.core_math_quiniela) cargado correctamente.")
except ImportError:
    print("\n❌ ERROR CRÍTICO: No se encuentra 'src.core_math_quiniela'.")
    print(f"   Asegúrate de guardar este script DENTRO de la carpeta 'quiniela/'.")
    sys.exit()

# -------------------------------------------------------------------------
# 2. CONFIGURACIÓN DE LA JORNADA
# -------------------------------------------------------------------------

# Datos Oficiales (Post-Cierre)
JACKPOT = 2360075.17   # Bote (Euros) - Pon 0 si no hay
ESTIMACION = 4500000.0   # Recaudación de la jornada
PRECIO_APUESTA = 0.75    # Precio por columna (0.75€ estándar)

DIST_PREMIOS = np.array([0.075, 0.16, 0.075, 0.075, 0.075, 0.09])

# -------------------------------------------------------------------------
# 3. TUS APUESTAS
# -------------------------------------------------------------------------
TEXTO_APUESTAS = """
01: 1111111122X1X2 + 01 | EV: 1.4139 €
02: 11111X1112211X + 01 | EV: 1.4469 €
03: 1111X11112X21X + 01 | EV: 1.4907 €
04: 1111111212X121 + 01 | EV: 1.5641 €
05: 1111111221X11X + 01 | EV: 1.4216 €
06: 1111X1111211X2 + 01 | EV: 1.4314 €
07: 1112111122211X + 01 | EV: 1.4058 €
08: 11111111112X12 + 01 | EV: 1.4173 €
"""

# -------------------------------------------------------------------------
# 4. DATOS LAE (XML)
# -------------------------------------------------------------------------
TEXTO_LAE_1X2 = """
<quinielista>
<porcentajes jornada="40" temporada="2026" activo="si">
<partido num="1" local="RAYO" visitante="R.OVIEDO" porc_1="61" porc_X="25" porc_2="14"/>
<partido num="2" local="BARCELONA" visitante="MALLORCA" porc_1="92" porc_X="5" porc_2="3"/>
<partido num="3" local="SEVILLA" visitante="GIRONA" porc_1="46" porc_X="30" porc_2="24"/>
<partido num="4" local="R.SOCIEDAD" visitante="ELCHE" porc_1="70" porc_X="19" porc_2="11"/>
<partido num="5" local="ALAVÉS" visitante="GETAFE" porc_1="42" porc_X="37" porc_2="21"/>
<partido num="6" local="ATH.CLUB" visitante="LEVANTE" porc_1="74" porc_X="17" porc_2="9"/>
<partido num="7" local="AT.MADRID" visitante="BETIS" porc_1="76" porc_X="16" porc_2="8"/>
<partido num="8" local="VILLARREAL" visitante="ESPANYOL" porc_1="71" porc_X="18" porc_2="11"/>
<partido num="9" local="CÁDIZ" visitante="ALMERÍA" porc_1="30" porc_X="34" porc_2="36"/>
<partido num="10" local="VALLADOLID" visitante="CASTELLÓN" porc_1="29" porc_X="28" porc_2="43"/>
<partido num="11" local="CEUTA" visitante="CÓRDOBA" porc_1="30" porc_X="35" porc_2="35"/>
<partido num="12" local="DEPORTIVO" visitante="ALBACETE" porc_1="67" porc_X="20" porc_2="13"/>
<partido num="13" local="SPORTING" visitante="HUESCA" porc_1="65" porc_X="24" porc_2="11"/>
<partido num="14" local="MÁLAGA" visitante="C. LEONESA" porc_1="76" porc_X="15" porc_2="9"/>
<partido num="15" local="VALENCIA" visitante="R.MADRID" porc_15L_0="25" porc_15L_1="57" porc_15L_2="15" porc_15L_M="3" porc_15V_0="3" porc_15V_1="16" porc_15V_2="37" porc_15V_M="44"/>
</porcentajes>
</quinielista>
"""

# Nota: Usamos este texto para el pleno (lae) porque es más fácil que calcular el producto cruzado del XML
TEXTO_LAE_PLENO = """
Partido 15: 10 5 2 1 15 20 5 2 8 5 3 1 5 2 1 15
"""

# -------------------------------------------------------------------------
# 5. DATOS REALES (XML para 1X2 + Lista para P15)
# -------------------------------------------------------------------------
# Nota: CUOTAS_REALES_1X2 es una lista con un string XML dentro.
CUOTAS_REALES_1X2 = ["""
<quinielista>
<porcentajes jornada="40" temporada="2026" activo="si">
<partido num="1" local="RAYO" visitante="R.OVIEDO" porc_1="62" porc_X="25" porc_2="13" porcDec_1="62" porcDec_X="25" porcDec_2="13"/>
<partido num="2" local="BARCELONA" visitante="MALLORCA" porc_1="83" porc_X="11" porc_2="6" porcDec_1="82.79" porcDec_X="11.18" porcDec_2="6.03"/>
<partido num="3" local="SEVILLA" visitante="GIRONA" porc_1="45" porc_X="28" porc_2="27" porcDec_1="44.83" porcDec_X="28.19" porcDec_2="26.98"/>
<partido num="4" local="R.SOCIEDAD" visitante="ELCHE" porc_1="57" porc_X="24" porc_2="19" porcDec_1="57.35" porcDec_X="24.18" porcDec_2="18.47"/>
<partido num="5" local="ALAVÉS" visitante="GETAFE" porc_1="41" porc_X="34" porc_2="25" porcDec_1="40.48" porcDec_X="34.17" porcDec_2="25.35"/>
<partido num="6" local="ATH.CLUB" visitante="LEVANTE" porc_1="59" porc_X="24" porc_2="17" porcDec_1="58.76" porcDec_X="24.23" porcDec_2="17.01"/>
<partido num="7" local="AT.MADRID" visitante="BETIS" porc_1="65" porc_X="21" porc_2="14" porcDec_1="64.79" porcDec_X="20.69" porcDec_2="14.52"/>
<partido num="8" local="VILLARREAL" visitante="ESPANYOL" porc_1="56" porc_X="23" porc_2="21" porcDec_1="56.12" porcDec_X="23.37" porcDec_2="20.51"/>
<partido num="9" local="CÁDIZ" visitante="ALMERÍA" porc_1="36" porc_X="28" porc_2="36" porcDec_1="35.5" porcDec_X="28.31" porcDec_2="36.19"/>
<partido num="10" local="VALLADOLID" visitante="CASTELLÓN" porc_1="30" porc_X="28" porc_2="42" porcDec_1="30.21" porcDec_X="28.12" porcDec_2="41.67"/>
<partido num="11" local="CEUTA" visitante="CÓRDOBA" porc_1="30" porc_X="35" porc_2="35" porcDec_1="30" porcDec_X="35" porcDec_2="35"/>
<partido num="12" local="DEPORTIVO" visitante="ALBACETE" porc_1="53" porc_X="26" porc_2="21" porcDec_1="53.32" porcDec_X="25.57" porcDec_2="21.11"/>
<partido num="13" local="SPORTING" visitante="HUESCA" porc_1="51" porc_X="29" porc_2="20" porcDec_1="50.97" porcDec_X="28.79" porcDec_2="20.24"/>
<partido num="14" local="MÁLAGA" visitante="C. LEONESA" porc_1="54" porc_X="26" porc_2="20" porcDec_1="54.16" porcDec_X="26.31" porcDec_2="19.53"/>
<partido num="15" local="VALENCIA" visitante="R.MADRID" porc_15L_0="40" porc_15L_1="37" porc_15L_2="17" porc_15L_M="6" porc_15V_0="15" porc_15V_1="28" porc_15V_2="27" porc_15V_M="30" porcDec_15L_0="40.02" porcDec_15L_1="36.65" porcDec_15L_2="16.78" porcDec_15L_M="6.55" porcDec_15V_0="14.61" porcDec_15V_1="28.1" porcDec_15V_2="27.03" porcDec_15V_M="30.26"/>
</porcentajes>
</quinielista>
"""]

# Odds reales para el P15 (16 valores)
CUOTAS_REALES_PLENO = [17.00,9.00,8.50,7.00,17.00,8.50,8.50,7.00,29.00,15.00,15.00,12.00,61.00,36.00,34.00,21.00]

# -------------------------------------------------------------------------
# 6. FUNCIONES DE PARSEO
# -------------------------------------------------------------------------

MAPA_1X2 = {'1': 0, 'X': 1, '2': 2}
# Mapa para el formato "01" (0-1), "M2" (M-2), etc.
MAPA_PLENO_STR = {
    '00':0, '01':1, '02':2, '0M':3,
    '10':4, '11':5, '12':6, '1M':7,
    '20':8, '21':9, '22':10, '2M':11,
    'M0':12, 'M1':13, 'M2':14, 'MM':15
}

def procesar_texto_apuestas(texto):
    """
    Parsea apuestas en formato: "01: 111X... + 01 | EV: ..."
    """
    lines = texto.strip().split('\n')
    matriz_1x2 = []
    array_pleno = []
    apuestas_legibles = []

    # Regex: Busca una secuencia de 14 signos (1,X,2) seguida de un "+" y 2 caracteres para el P15
    # Ejemplo: "1111111122X1X2 + 01"
    regex = r"([1X2]{14})\s*\+\s*([012M]{2})"

    for line in lines:
        match = re.search(regex, line)
        if match:
            str_14 = match.group(1) # Los 14 signos
            str_p15 = match.group(2) # El pleno (ej: "01")
            
            # Convertir 1X2 a enteros
            fila_1x2 = [MAPA_1X2[s] for s in str_14]
            matriz_1x2.append(fila_1x2)
            
            # Convertir P15 a entero (0-15)
            # Asumimos que "01" significa 0-1, "MM" significa M-M
            if str_p15 in MAPA_PLENO_STR:
                array_pleno.append(MAPA_PLENO_STR[str_p15])
            else:
                print(f"⚠️ P15 desconocido: {str_p15}. Usando 0 (0-0).")
                array_pleno.append(0)
            
            # Formato legible para print
            p15_legible = f"{str_p15[0]}-{str_p15[1]}"
            apuestas_legibles.append(f"{str_14} + {p15_legible}")

    if not matriz_1x2:
        raise ValueError("❌ No se encontraron apuestas. Revisa el regex.")

    return np.array(matriz_1x2, dtype=np.int32), np.array(array_pleno, dtype=np.int32), apuestas_legibles

def parsear_xml_1x2(xml_text, tags=('porc_1', 'porc_X', 'porc_2')):
    """
    Extrae los porcentajes de los partidos 1 al 14 de un XML.
    tags: Tupla con los nombres de los atributos a buscar (ej: porcDec_1 para reales)
    """
    matriz = np.zeros((14, 3), dtype=np.float64)
    found_count = 0
    
    # Recorremos del 1 al 14
    for i in range(1, 15):
        # Regex para encontrar <partido num="i" ... /> y extraer los atributos
        # Se hace flexible por si el orden de atributos cambia
        # Buscamos la línea del partido específico
        pattern_line = f'<partido num="{i}"'
        
        # Buscar la posición en el texto
        start_idx = xml_text.find(pattern_line)
        if start_idx == -1: continue
        
        # Extraer el trozo de texto hasta el cierre de etiqueta
        end_idx = xml_text.find('/>', start_idx)
        if end_idx == -1: end_idx = xml_text.find('>', start_idx)
        line_content = xml_text[start_idx:end_idx]
        
        # Extraer valores
        try:
            val_1 = float(re.search(f'{tags[0]}="([\d\.]+)"', line_content).group(1))
            val_X = float(re.search(f'{tags[1]}="([\d\.]+)"', line_content).group(1))
            val_2 = float(re.search(f'{tags[2]}="([\d\.]+)"', line_content).group(1))
            
            matriz[i-1] = [val_1, val_X, val_2]
            found_count += 1
        except AttributeError:
            print(f"⚠️ Error parseando partido {i} en XML.")

    if found_count < 14:
        raise ValueError(f"❌ Solo se encontraron {found_count}/14 partidos en el XML.")
        
    # Normalizar
    return matriz / matriz.sum(axis=1)[:, None]

def procesar_lae_pleno(texto):
    """Parsea los 16 números del texto simple para el P15."""
    texto = texto.replace(',', '.')
    numeros = re.findall(r"[-+]?\d*\.\d+|\d+", texto)
    numeros = [float(n) for n in numeros]
    if len(numeros) < 16:
        raise ValueError("❌ Faltan datos en TEXTO_LAE_PLENO.")
    p = np.array(numeros[-16:])
    return p / p.sum()

def procesar_reales(lista_xml_1x2, lista_odds_pleno):
    # 1. Parsear el XML de reales (está dentro de una lista)
    xml_content = lista_xml_1x2[0]
    # Usamos las etiquetas 'porcDec_...' que son las probabilidades reales en %
    probs_1x2 = parsear_xml_1x2(xml_content, tags=('porcDec_1', 'porcDec_X', 'porcDec_2'))
    
    # 2. Procesar P15 desde la lista de Cuotas (Odds)
    v_pleno = np.array(lista_odds_pleno)
    p_pleno = 1.0 / v_pleno # Convertir cuota a probabilidad
    p_pleno = p_pleno / p_pleno.sum() # Normalizar
    
    return probs_1x2, p_pleno

# -------------------------------------------------------------------------
# 7. EJECUCIÓN
# -------------------------------------------------------------------------
def main():
    print("\n--- ⚽ CALCULADORA EV QUINIELA (XML) ---")
    
    try:
        # 1. Parsear Apuestas
        c_1x2, c_p15, str_aps = procesar_texto_apuestas(TEXTO_APUESTAS)
        print(f"✅ Apuestas cargadas: {len(str_aps)}")

        # 2. Parsear LAE (XML + Texto P15)
        # Usamos tags por defecto (porc_1, porc_X...)
        lae_1x2 = parsear_xml_1x2(TEXTO_LAE_1X2) 
        lae_p15 = procesar_lae_pleno(TEXTO_LAE_PLENO)
        
        # 3. Parsear Reales (XML + Lista Odds P15)
        # Usamos tags de decimales (porcDec_1...)
        real_1x2, real_p15 = procesar_reales(CUOTAS_REALES_1X2, CUOTAS_REALES_PLENO)
        
        print("✅ Datos de probabilidades procesados correctamente.")
        print("-" * 50)

        # 4. Calcular EV
        evs = engine.get_top_candidates_quiniela(
            c_1x2, c_p15,
            real_1x2, real_p15,
            lae_1x2, lae_p15,
            float(ESTIMACION), float(JACKPOT), DIST_PREMIOS
        )

        # 5. Mostrar
        total_ev = 0.0
        for i, val in enumerate(evs):
            roi = ((val - PRECIO_APUESTA)/PRECIO_APUESTA)*100
            icon = "✅" if val > PRECIO_APUESTA else "🔻"
            print(f"Apuesta {i+1:02d}: [{str_aps[i]}]")
            print(f"   EV: {val:.4f} € (ROI: {roi:+.2f}%) {icon}")
            total_ev += val
            
        print("="*50)
        coste = len(evs) * PRECIO_APUESTA
        print(f"Coste Total: {coste:.2f} €")
        print(f"EV Total:    {total_ev:.4f} €")
        print(f"ROI Global:  {((total_ev-coste)/coste)*100:+.2f}%")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()