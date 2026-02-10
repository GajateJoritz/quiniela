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
    import src.core_math as engine
    print("✅ Motor matemático (src.core_math) cargado correctamente.")
except ImportError:
    print("\n❌ ERROR CRÍTICO: No se encuentra el módulo 'src.core_math'.")
    print(f"   Asegúrate de guardar este script DENTRO de la carpeta 'quinigol/'.")
    sys.exit()

# -------------------------------------------------------------------------
# 2. DATOS DE LA JORNADA
# -------------------------------------------------------------------------

JACKPOT = 65000.0       # Bote
ESTIMACION = 145000.0   # Recaudación
PRECIO_APUESTA = 1.0    
DIST_PREMIOS = np.array([0.10, 0.09, 0.08, 0.08, 0.20])

# --- TUS APUESTAS ---
TEXTO_APUESTAS = """
Bet #1: 0-1 | 0-M | 1-2 | 1-0 | 1-M | 0-M
Bet #2: 1-0 | 0-M | 0-1 | 0-1 | M-0 | 1-M
Bet #3: 0-1 | 0-M | 1-2 | 1-0 | M-0 | 0-M
Bet #4: 1-0 | 0-M | 0-1 | 0-1 | 1-M | 1-M
"""

# --- DATOS LAE (Formato Equipos con comas) ---
TEXTO_LAE = """
1;1;RAYO VALLECANO; ;14,4;38,5;36,6;10,5;
2;R. OVIEDO; ;39,4;44,8;11,9;3,90000000000001;
2;3;BARCELONA; ;3,7;7,6;21,5;67,2;
4;MALLORCA; ;44,6;43,8;8,1;3,5;
3;5;ATHLETIC CLUB; ;7,1;28,1;45,6;19,2;
6;LEVANTE; ;39,2;46,1;11,2;3,49999999999999;
4;7;AT. MADRID; ;5,3;21,8;40,2;32,7;
8;BETIS; ;33,1;48,2;14,7;3,99999999999999;
5;9;VALENCIA; ;27,2;52,3;16,2;4,3;
10;REAL MADRID; ;5,7;17,4;37,7;39,2;
6;11;VILLARREAL ; ;7,1;24,8;44,9;23,2;
12;ESPANYOL; ;29,8;50,7;15,2;4,3;
"""

# --- CUOTAS REALES ---
real_probs_1 = [29.00, 41.00, 67.00, 226.00, 6.70, 11.30, 7.80, 20.70, 14.20, 9.50, 33.20, 22.90, 28.30, 25.00, 22.90, 61.00]
real_probs_2 = [29.00, 41.00, 67.00, 226.00, 12.00, 17.00, 41.00, 131.00, 8.00, 10.00, 26.00, 76.00, 2.90, 3.75, 9.50, 26.00]
real_probs_3 = [4.75, 7.00, 17.00, 46.00, 4.75, 6.00, 17.00, 46.00, 9.00, 12.000, 26.00, 71.00, 19.00, 26.00, 56.00, 131.00]
real_probs_4 = [15.00, 17.00, 34.00, 96.00, 8.00, 9.00, 19.00, 41.00, 7.50, 8.50, 17.00, 41.00, 6.00, 6.50, 13.00, 29.00]
real_probs_5 = [17.00, 9.00, 8.50, 7.00, 17.00, 8.50, 8.50, 7.00, 29.00, 15.00, 15.00, 12.00, 61.00, 36.00, 34.00, 21.00]
real_probs_6 = [13.00, 13.00, 23.00, 51.00, 8.00, 8.00, 15.00, 34.00, 8.50, 8.50, 15.00, 29.00, 8.00, 8.00, 13.00, 26.00]

CUOTAS_REALES = [real_probs_1, real_probs_2, real_probs_3, real_probs_4, real_probs_5, real_probs_6]

# -------------------------------------------------------------------------
# 3. FUNCIONES DE PROCESAMIENTO
# -------------------------------------------------------------------------

MAPA_RESULTADOS = {
    '0-0':0, '0-1':1, '0-2':2, '0-M':3,
    '1-0':4, '1-1':5, '1-2':6, '1-M':7,
    '2-0':8, '2-1':9, '2-2':10, '2-M':11,
    'M-0':12, 'M-1':13, 'M-2':14, 'M-M':15
}

def procesar_texto_apuestas(texto):
    # Regex: busca dígito o M, guión, dígito o M. Ej: 1-0, M-2, 0-M
    patron = r"[012M]-[012M]"
    coincidencias = re.findall(patron, texto)
    
    cantidad = len(coincidencias)
    if cantidad == 0:
        raise ValueError("❌ No se encontraron apuestas. Revisa el texto.")
    
    if cantidad % 6 != 0:
        print(f"⚠️ AVISO: {cantidad} resultados encontrados (no es múltiplo de 6).")
    
    # Agrupar en bloques de 6
    num_apuestas = cantidad // 6
    matriz_indices = []
    apuestas_texto = []
    
    for i in range(num_apuestas):
        bloque = coincidencias[i*6 : (i+1)*6]
        apuestas_texto.append(bloque)
        matriz_indices.append([MAPA_RESULTADOS[res] for res in bloque])
        
    return np.array(matriz_indices, dtype=np.int32), apuestas_texto

def procesar_texto_lae(texto):
    """
    Parsea el texto de LAE.
    Soporta formato decimal con comas (14,4).
    Soporta formato desglosado por equipos (12 filas con 4% cada una).
    """
    # 1. Reemplazar comas decimales por puntos para que Python entienda los floats
    texto_limpio = texto.replace(',', '.')
    
    # 2. Extraer todos los números del texto
    numeros = re.findall(r"[-+]?\d*\.\d+|\d+", texto_limpio)
    numeros = [float(n) for n in numeros]

    # --- LÓGICA DE DETECCIÓN DE FORMATO ---
    
    # FORMATO A: Matriz completa ya calculada (96 números exactos o 102 con índices)
    if len(numeros) == 96:
        matriz = np.array(numeros).reshape(6, 16)
        return normalizar_matriz(matriz)
    
    # FORMATO B: Tabla por equipos (Lo que has pasado ahora)
    # Estructura esperada: 12 filas. Cada fila tiene varios números (índices) y al final los 4 porcentajes.
    # El regex habrá extraído todo mezclado.
    # Es más seguro procesar línea por línea para coger SOLO los últimos 4 números de cada fila.
    
    lines = [l.strip() for l in texto_limpio.split('\n') if l.strip()]
    vectores_equipos = []
    
    for line in lines:
        nums_line = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        nums_line = [float(n) for n in nums_line]
        
        # Necesitamos que la línea tenga al menos 4 números (los porcentajes)
        if len(nums_line) >= 4:
            # Asumimos que los porcentajes son los ÚLTIMOS 4 números de la línea (0, 1, 2, M)
            probs_equipo = nums_line[-4:] 
            vectores_equipos.append(probs_equipo)
            
    # Deberíamos tener 12 vectores (6 partidos x 2 equipos)
    if len(vectores_equipos) == 12:
        print("ℹ️  Detectado formato LAE 'Por Equipos' (12 filas). Calculando combinaciones...")
        matriz_final = np.zeros((6, 16), dtype=np.float64)
        
        for i in range(6):
            # Equipo Local (fila par) y Visitante (fila impar)
            probs_local = np.array(vectores_equipos[2*i])
            probs_visit = np.array(vectores_equipos[2*i+1])
            
            # Normalizamos cada vector por si acaso (deben sumar 100)
            if probs_local.sum() > 0: probs_local /= probs_local.sum()
            if probs_visit.sum() > 0: probs_visit /= probs_visit.sum()
            
            # Producto exterior para sacar las 16 combinaciones
            # Matriz 4x4 -> (Local 0..M) x (Visitante 0..M)
            # Flatten orden: 0-0, 0-1, 0-2, 0-M, 1-0 ... (Row-major, que coincide con el mapa)
            comb_matrix = np.outer(probs_local, probs_visit).flatten()
            matriz_final[i, :] = comb_matrix
            
        return matriz_final
        
    else:
        # Fallback si no encaja nada
        raise ValueError(f"❌ Datos LAE inválidos. Se extrajeron {len(numeros)} números en total y {len(vectores_equipos)} filas de equipo válidas (se esperaban 12).")

def normalizar_matriz(matriz):
    suma = matriz.sum(axis=1)
    if np.any(suma == 0): raise ValueError("❌ Una fila de LAE suma 0.")
    return matriz / suma[:, None]

def procesar_cuotas_reales(lista_cuotas):
    matriz = np.array(lista_cuotas)
    probs = 1.0 / matriz
    suma = probs.sum(axis=1)
    return probs / suma[:, None]

# -------------------------------------------------------------------------
# 4. EJECUCIÓN
# -------------------------------------------------------------------------
def main():
    print("\n--- 📊 CALCULADORA DE EV REAL (POST-CIERRE) ---")
    
    try:
        # 1. Procesar datos
        matriz_lae = procesar_texto_lae(TEXTO_LAE)
        matriz_real = procesar_cuotas_reales(CUOTAS_REALES)
        matriz_apuestas, lista_apuestas_str = procesar_texto_apuestas(TEXTO_APUESTAS)
        
        print(f"✅ Datos procesados: {len(matriz_apuestas)} apuestas validadas.")
        print("-" * 50)

        # 2. Calcular EV
        evs = engine.get_top_candidates(
            matriz_apuestas, 
            matriz_real, 
            matriz_lae, 
            float(ESTIMACION), 
            float(JACKPOT), 
            DIST_PREMIOS
        )

        # 3. Resultados
        total_ev = 0.0
        
        for i, ev_val in enumerate(evs):
            texto_apuesta = ' | '.join(lista_apuestas_str[i])
            roi = ((ev_val - PRECIO_APUESTA) / PRECIO_APUESTA) * 100
            icono = "✅" if ev_val > PRECIO_APUESTA else "🔻"
            
            print(f"Apuesta #{i+1}: [{texto_apuesta}]")
            print(f"   EV: {ev_val:.4f} €  (ROI: {roi:+.1f}%) {icono}")
            print("-" * 20)
            total_ev += ev_val

        print("=" * 50)
        print(f"RESUMEN FINAL:")
        print(f" > Inversión total: {len(evs) * PRECIO_APUESTA:.2f} €")
        print(f" > EV Total Cartera: {total_ev:.4f} €")
        print(f" > EV Promedio: {total_ev/len(evs):.4f} €")
        
        if total_ev > (len(evs) * PRECIO_APUESTA):
            print("🚀 LA ESTRATEGIA TIENE VALOR POSITIVO (+EV)")
        else:
            print("⚠️ LA ESTRATEGIA TIENE VALOR NEGATIVO (-EV)")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()