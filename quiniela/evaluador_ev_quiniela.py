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
JACKPOT = 0.0
ESTIMATION = 1100000.0  # Recaudación de la jornada
PRECIO_APUESTA = 0.75    # Precio por columna (0.75€ estándar)

DIST_PREMIOS = np.array([0.075, 0.16, 0.075, 0.075, 0.075, 0.09])

# -------------------------------------------------------------------------
# 3. TUS APUESTAS
# -------------------------------------------------------------------------
TEXTO_APUESTAS = """
01: 22X22121122122 + 01 | EV: 1.1268 €
02: 22222221X12121 + 01 | EV: 1.1437 €
03: 12X22121111121 + 01 | EV: 1.1710 €
04: 11X22221121111 + 01 | EV: 1.1552 €
05: 12X22X21X1X112 + 01 | EV: 1.1567 €
06: 21122121X1X111 + 01 | EV: 1.1283 €
"""

# -------------------------------------------------------------------------
# 4. DATOS LAE (XML)
# -------------------------------------------------------------------------
TEXTO_LAE_1X2 = """
<quinielista>
<porcentajes jornada="41" temporada="2026" activo="si">
<partido num="1" local="ATH.CLUB" visitante="R.SOCIEDAD" porc_1="46" porc_X="30" porc_2="24"/>
<partido num="2" local="AT.MADRID" visitante="BARCELONA" porc_1="26" porc_X="23" porc_2="51"/>
<partido num="3" local="CHELSEA" visitante="LEEDS" porc_1="86" porc_X="8" porc_2="6"/>
<partido num="4" local="EVERTON" visitante="BOURNEMOUTH" porc_1="52" porc_X="28" porc_2="20"/>
<partido num="5" local="TOTTENHAM" visitante="NEWCASTLE" porc_1="41" porc_X="29" porc_2="30"/>
<partido num="6" local="WEST HAM" visitante="MAN.UNITED" porc_1="13" porc_X="17" porc_2="70"/>
<partido num="7" local="ASTON VILLA" visitante="BRIGHTON" porc_1="78" porc_X="13" porc_2="9"/>
<partido num="8" local="CRYSTAL PALACE" visitante="BURNLEY" porc_1="73" porc_X="18" porc_2="9"/>
<partido num="9" local="MAN.CITY" visitante="FULHAM" porc_1="89" porc_X="6" porc_2="5"/>
<partido num="10" local="NOTTINGHAM" visitante="WOLVERHAMPTON" porc_1="67" porc_X="22" porc_2="11"/>
<partido num="11" local="BRENTFORD" visitante="ARSENAL" porc_1="10" porc_X="15" porc_2="75"/>
<partido num="12" local="CELTIC" visitante="LIVINGSTON" porc_1="88" porc_X="7" porc_2="5"/>
<partido num="13" local="DUNDEE UNITED" visitante="ABERDEEN" porc_1="39" porc_X="35" porc_2="26"/>
<partido num="14" local="MOTHERWELL" visitante="RANGERS" porc_1="20" porc_X="27" porc_2="53"/>
<partido num="15" local="SUNDERLAND" visitante="LIVERPOOL" porc_15L_0="25" porc_15L_1="56" porc_15L_2="16" porc_15L_M="3" porc_15V_0="4" porc_15V_1="20" porc_15V_2="41" porc_15V_M="35"/>
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
<porcentajes jornada="41" temporada="2026" activo="si">
<partido num="1" local="ATH.CLUB" visitante="R.SOCIEDAD" porc_1="42" porc_X="29" porc_2="29" porcDec_1="42.07" porcDec_X="29.45" porcDec_2="28.48"/>
<partido num="2" local="AT.MADRID" visitante="BARCELONA" porc_1="31" porc_X="24" porc_2="45" porcDec_1="30.98" porcDec_X="24.4" porcDec_2="44.62"/>
<partido num="3" local="CHELSEA" visitante="LEEDS" porc_1="63" porc_X="21" porc_2="16" porcDec_1="63.12" porcDec_X="21.18" porcDec_2="15.7"/>
<partido num="4" local="EVERTON" visitante="BOURNEMOUTH" porc_1="39" porc_X="28" porc_2="33" porcDec_1="38.97" porcDec_X="28.53" porcDec_2="32.5"/>
<partido num="5" local="TOTTENHAM" visitante="NEWCASTLE" porc_1="30" porc_X="27" porc_2="43" porcDec_1="29.99" porcDec_X="26.58" porcDec_2="43.43"/>
<partido num="6" local="WEST HAM" visitante="MAN.UNITED" porc_1="22" porc_X="23" porc_2="55" porcDec_1="22.14" porcDec_X="22.94" porcDec_2="54.92"/>
<partido num="7" local="ASTON VILLA" visitante="BRIGHTON" porc_1="48" porc_X="27" porc_2="25" porcDec_1="48.17" porcDec_X="26.48" porcDec_2="25.35"/>
<partido num="8" local="CRYSTAL PALACE" visitante="BURNLEY" porc_1="62" porc_X="23" porc_2="15" porcDec_1="62.49" porcDec_X="22.66" porcDec_2="14.85"/>
<partido num="9" local="MAN.CITY" visitante="FULHAM" porc_1="70" porc_X="18" porc_2="12" porcDec_1="70.1" porcDec_X="17.75" porcDec_2="12.15"/>
<partido num="10" local="NOTTINGHAM" visitante="WOLVERHAMPTON" porc_1="56" porc_X="25" porc_2="19" porcDec_1="56.11" porcDec_X="25.18" porcDec_2="18.71"/>
<partido num="11" local="BRENTFORD" visitante="ARSENAL" porc_1="20" porc_X="24" porc_2="56" porcDec_1="19.64" porcDec_X="24.32" porcDec_2="56.04"/>
<partido num="12" local="CELTIC" visitante="LIVINGSTON" porc_1="78" porc_X="16" porc_2="6" porcDec_1="78.11" porcDec_X="15.5" porcDec_2="6.39"/>
<partido num="13" local="DUNDEE UNITED" visitante="ABERDEEN" porc_1="39" porc_X="28" porc_2="33" porcDec_1="39.09" porcDec_X="28" porcDec_2="32.91"/>
<partido num="14" local="MOTHERWELL" visitante="RANGERS" porc_1="29" porc_X="26" porc_2="45" porcDec_1="28.54" porcDec_X="26.23" porcDec_2="45.23"/>
<partido num="15" local="SUNDERLAND" visitante="LIVERPOOL" porc_15L_0="38" porc_15L_1="37" porc_15L_2="18" porc_15L_M="7" porc_15V_0="17" porc_15V_1="30" porc_15V_2="27" porc_15V_M="26" porcDec_15L_0="37.81" porcDec_15L_1="36.77" porcDec_15L_2="17.88" porcDec_15L_M="7.54" porcDec_15V_0="16.83" porcDec_15V_1="29.99" porcDec_15V_2="26.72" porcDec_15V_M="26.46"/>
</porcentajes>
</quinielista>
"""]




# Odds reales para el P15 (16 valores)
CUOTAS_REALES_PLENO = [11.00,8.00,9.00,9.00,15.00,7.50,9.00,8.50,29.00,17.00,17.00,15.00,67.00,34.00,36.00,29.00]

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
            float(ESTIMATION), float(JACKPOT), DIST_PREMIOS
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