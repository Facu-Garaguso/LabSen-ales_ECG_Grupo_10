"""
Análisis de ECG en reposo y post actividad física.

Etapas (cada una es una única función que llama a funciones auxiliares):
    1. etapa_visualizacion       -> estadísticas y gráficos de los registros adquiridos
    2. etapa_preprocesamiento    -> continua/deriva, interferencia de 50 Hz, saturaciones
    3. etapa_pan_tompkins        -> detección de complejos QRS, RR y frecuencia cardíaca
    4. etapa_tacograma           -> tacograma con marcado de intervalos no válidos por saturacion
    5. etapa_lomb                -> periodograma de Lomb y bandas LF/HF

Todas las figuras se escriben en un único PDF (figuras_ecg.pdf); no se muestran
en pantalla ni se guardan como imágenes sueltas.
"""

from collections import deque

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from scipy import signal
from scipy.integrate import trapezoid


# ============================================================
# CONFIGURACIÓN
# ============================================================

FS = 480  # Frecuencia de muestreo [Hz]

ARCHIVOS_ECG = {"pre": "ecg_data_pre.csv", "post": "ecg_data_post.csv"}
ESTADOS = {"pre": "ECG en reposo", "post": "ECG post actividad física"}

ARCHIVO_FIGURAS = "figuras_ecg.pdf"  # Único archivo donde se vuelcan todas las figuras
DPI_FIGURAS = 200                    # Resolución de las curvas (el texto y los ejes son vectoriales)
GUARDAR_CSV = True                   # Guardar resultados numéricos en CSV

# Ventana de los gráficos ampliados
INICIO_ZOOM = 0      # [s]
DURACION_ZOOM = 10   # [s]

# Preprocesamiento
FC_HP = 0.5          # Frecuencia de corte del pasa-altos [Hz]
ORDEN_HP = 4
F_NOTCH = 50         # Frecuencia de red eléctrica [Hz]
Q_NOTCH = 30
APLICAR_NOTCH = False  # Cambiar a True solo si el espectro confirma interferencia de 50 Hz
LIMITE_INFERIOR = 0.01  # Límites del conversor para detectar saturación [V]
LIMITE_SUPERIOR = 3.29

# Periodograma de Lomb
F_MAX_LOMB = 0.50
N_FRECUENCIAS_LOMB = 5000
LF_MIN, LF_MAX = 0.04, 0.15
HF_MIN, HF_MAX = 0.15, 0.40


# ============================================================
# UTILIDADES GENERALES
# ============================================================

def cargar_ecg(ruta):
    """Lee la columna 'Sample' de un CSV y devuelve la señal como vector, sin valores no numéricos."""
    datos = pd.read_csv(ruta, usecols=lambda c: c.strip() == "Sample")

    if datos.shape[1] == 0:
        columnas = pd.read_csv(ruta, nrows=0).columns.str.strip().tolist()
        raise ValueError(
            f"El archivo {ruta} no contiene una columna llamada 'Sample'. "
            f"Columnas encontradas: {columnas}"
        )

    señal = pd.to_numeric(datos.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    return señal[np.isfinite(señal)]


def vector_tiempo(n_muestras, fs):
    return np.arange(n_muestras) / fs


def guardar_csv(nombre, datos):
    if GUARDAR_CSV:
        pd.DataFrame(datos).to_csv(nombre, index=False)


def encabezado(texto):
    print(f"\n{'=' * 8} {texto} {'=' * 8}")


# ============================================================
# GRAFICADO
# ============================================================

def _recortar(x, y, rango):
    """Recorta (x, y) al intervalo rango=(min, max); x debe estar ordenado."""
    if rango is None:
        return x, y
    i0 = np.searchsorted(x, rango[0], side="left")
    i1 = np.searchsorted(x, rango[1], side="right")
    return x[i0:i1], y[i0:i1]


def graficar(
    pdf,
    curvas,
    titulo,
    *,
    xlabel="Tiempo [s]",
    ylabel="Amplitud [V]",
    rango_x=None,
    hlines=(),
    vlines=(),
    marcas=(),
    log_y=False,
    figsize=(14, 5),
):
    """
    Dibuja una figura y la agrega al PDF.

    curvas, marcas : listas de dicts con 'x', 'y' y opciones de matplotlib
    hlines, vlines : listas de dicts con 'y' / 'x' y opciones de matplotlib
    rango_x        : (min, max) para recortar la figura a una ventana
    """
    fig = Figure(figsize=figsize, layout="constrained")
    ax = fig.subplots()

    for curva in curvas:
        opciones = {"linewidth": 0.8, "rasterized": True, **curva}
        x, y = _recortar(opciones.pop("x"), opciones.pop("y"), rango_x)
        ax.plot(x, y, **opciones)

    for marca in marcas:
        opciones = dict(marca)
        x, y = _recortar(opciones.pop("x"), opciones.pop("y"), rango_x)
        ax.scatter(x, y, **opciones)

    for linea in hlines:
        ax.axhline(**linea)
    for linea in vlines:
        ax.axvline(**linea)

    if log_y:
        ax.set_yscale("log")
    if rango_x is not None:
        ax.set_xlim(*rango_x)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    ax.grid(True, alpha=0.3)

    if ax.get_legend_handles_labels()[0]:
        ax.legend()

    pdf.savefig(fig, dpi=DPI_FIGURAS)


def graficar_etapas_pan_tompkins(pdf, ecg, res, fs, titulo):
    t = vector_tiempo(len(ecg), fs)
    fig = Figure(figsize=(14, 11), layout="constrained")
    axes = fig.subplots(5, 1, sharex=True)

    paneles = (
        (ecg, f"{titulo} - ECG procesado"),
        (res["bandpass"], "Filtrado pasa-banda 5-15 Hz"),
        (res["derivada"], "Filtro derivativo"),
        (res["cuadrado"], "Elevación al cuadrado"),
    )
    for ax, (y, subtitulo) in zip(axes, paneles):
        ax.plot(t, y, linewidth=0.8, rasterized=True)
        ax.set_title(subtitulo)

    ax = axes[4]
    integrado = res["integrado"]
    ax.plot(t, integrado, linewidth=0.8, rasterized=True, label="Señal integrada")
    ax.plot(t, res["umbral"], "--", linewidth=1.2, label="Umbral adaptativo")

    qrs, searchback = res["picos_integrados"], res["picos_searchback"]
    if len(qrs):
        ax.scatter(qrs / fs, integrado[qrs], marker="x", s=45, color="tab:red", label="QRS detectados")
    if len(searchback):
        ax.scatter(
            searchback / fs, integrado[searchback], marker="o", s=70,
            facecolors="none", edgecolors="tab:green", label="Recuperados por search-back",
        )
    ax.set_title("Integración por ventana móvil y umbral adaptativo")
    ax.set_xlabel("Tiempo [s]")
    ax.legend()

    for ax in axes:
        ax.set_ylabel("Amplitud")
        ax.grid(True, alpha=0.3)

    pdf.savefig(fig, dpi=DPI_FIGURAS)


def graficar_preprocesamiento(pdf, titulo, t, ecg, ecg_hp, ecg_final, media, f, psd, saturacion):
    rango_zoom = (INICIO_ZOOM, INICIO_ZOOM + DURACION_ZOOM)
    centrada = {"x": t, "y": ecg - media, "label": "Original centrada para comparación", "linewidth": 0.6}
    filtrada = {"x": t, "y": ecg_hp, "label": f"Pasa-altos {FC_HP} Hz"}
    linea_50hz = [{"x": F_NOTCH, "linestyle": "--", "color": "tab:red", "label": f"{F_NOTCH} Hz"}]

    # Componente continua
    graficar(
        pdf, [{"x": t, "y": ecg, "label": "ECG adquirido", "linewidth": 0.7}],
        f"{titulo} - Componente continua",
        hlines=[{"y": media, "linestyle": "--", "color": "tab:red", "label": f"Valor medio = {media:.3f} V"}],
    )

    # Efecto del pasa-altos (completo y ampliado)
    graficar(pdf, [centrada, filtrada], f"{titulo} - Corrección de línea de base", figsize=(14, 6))
    graficar(pdf, [centrada, filtrada], f"{titulo} - Efecto del pasa-altos", rango_x=rango_zoom, figsize=(14, 6))

    # Espectro completo y alrededor de 50 Hz
    for rango, sufijo in (((0, 100), "Densidad espectral de potencia"), ((40, 60), "Análisis alrededor de 50 Hz")):
        graficar(
            pdf, [{"x": f, "y": psd, "linewidth": 1.0}], f"{titulo} - {sufijo}",
            xlabel="Frecuencia [Hz]", ylabel="PSD [V²/Hz]",
            rango_x=rango, vlines=linea_50hz, log_y=True, figsize=(12, 5),
        )

    # Saturación
    graficar(
        pdf, [{"x": t, "y": ecg, "label": "ECG", "linewidth": 0.7}],
        f"{titulo} - Evaluación de saturación",
        marcas=(
            [{"x": t[saturacion], "y": ecg[saturacion], "s": 12, "color": "tab:red",
              "label": "Muestras próximas a saturación"}]
            if saturacion.any() else []
        ),
    )

    # Señal preprocesada (completa y ampliada)
    graficar(pdf, [{"x": t, "y": ecg_final}], f"{titulo} - Señal preprocesada")
    graficar(pdf, [{"x": t, "y": ecg_final, "linewidth": 1.0}], f"{titulo} - Señal preprocesada", rango_x=rango_zoom)


def graficar_picos_r(pdf, ecg, picos_r, fs, titulo):
    t = vector_tiempo(len(ecg), fs)
    marcas = []
    if len(picos_r):
        marcas.append({"x": picos_r / fs, "y": ecg[picos_r], "marker": "x", "s": 60,
                       "color": "tab:red", "label": "Picos R", "zorder": 3})
    graficar(pdf, [{"x": t, "y": ecg, "label": "ECG procesado"}], titulo, ylabel="Amplitud", marcas=marcas)


def graficar_frecuencia_cardiaca(pdf, picos_r, fc, fs, titulo):
    if len(fc) == 0:
        print(f"No hay suficientes detecciones para graficar: {titulo}")
        return
    graficar(
        pdf, [{"x": picos_r[1:] / fs, "y": fc, "marker": "o", "markersize": 4, "linewidth": 1.0}],
        titulo, ylabel="Frecuencia cardíaca [lat/min]", figsize=(12, 4),
    )


def graficar_tacograma(pdf, tiempo, rr, validos, titulo):
    if len(rr) == 0:
        print(f"No hay suficientes picos R para graficar: {titulo}")
        return

    rr_ms = rr * 1000
    marcas = []
    if not validos.all():
        marcas.append({"x": tiempo[~validos], "y": rr_ms[~validos], "marker": "x", "s": 50,
                       "color": "tab:red", "label": "Intervalos afectados por saturación"})

    # Los inválidos se pasan como NaN para que la línea no los una
    graficar(
        pdf,
        [{"x": tiempo, "y": np.where(validos, rr_ms, np.nan), "marker": "o", "markersize": 4,
          "linewidth": 1.0, "label": "Intervalos RR válidos"}],
        titulo, ylabel="Intervalo RR [ms]", marcas=marcas, figsize=(12, 4.5),
    )


def lineas_bandas_lomb(etiquetar=True):
    """Líneas verticales que delimitan las bandas LF (0.04-0.15 Hz) y HF (0.15-0.40 Hz)."""
    return [
        {"x": f, "linestyle": "--", "linewidth": 0.8, "color": "gray",
         "label": f"{f:.2f} Hz" if etiquetar else None}
        for f in (LF_MIN, LF_MAX, HF_MAX)
    ]


def graficar_lomb(pdf, frecuencias, curvas, titulo, f_dominante=None):
    """curvas: lista de (etiqueta, potencia). Con una sola curva se marcan los límites de banda."""
    unica = len(curvas) == 1
    vlines = lineas_bandas_lomb(etiquetar=unica)
    if f_dominante is not None:
        vlines.append({"x": f_dominante, "linestyle": ":", "linewidth": 1.2, "color": "tab:red",
                       "label": f"Frecuencia dominante = {f_dominante:.3f} Hz"})

    graficar(
        pdf, [{"x": frecuencias, "y": p, "linewidth": 1.2, "label": etiqueta} for etiqueta, p in curvas],
        titulo, xlabel="Frecuencia [Hz]", ylabel="Potencia normalizada",
        rango_x=(frecuencias[0], frecuencias[-1]), vlines=vlines, figsize=(12, 5),
    )


# ============================================================
# ETAPA 2 - FUNCIONES DE ANÁLISIS DEL PREPROCESAMIENTO
# ============================================================

def mostrar_estadisticas(nombre, señal, fs):
    """Imprime número de muestras, duración y estadísticos básicos del registro."""
    duracion = len(señal) / fs
    encabezado(nombre)
    print(f"Número de muestras: {len(señal)}")
    print(f"Duración: {duracion:.2f} s ({duracion / 60:.2f} min)")
    print(f"Media: {np.mean(señal):.6f} V")
    print(f"Mínimo: {np.min(señal):.6f} V | Máximo: {np.max(señal):.6f} V")
    print(f"Desvío estándar: {np.std(señal):.6f} V")


def filtro_pasa_altos(señal, fs, fc, orden):
    """
    Elimina la componente continua y la deriva lenta de la línea de base con un
    Butterworth pasa-altos. Se aplica hacia adelante y hacia atrás (fase cero) para
    no desplazar temporalmente el complejo QRS. Se usa la forma SOS (Second-Order Sections), más estable
    numéricamente que b/a para cortes tan bajos respecto de fs.
    """
    sos = signal.butter(orden, fc, btype="highpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, señal)


def filtro_notch(señal, fs, frecuencia, q):
    """Rechaza-banda angosto (fase cero) centrado en la frecuencia de red eléctrica."""
    b, a = signal.iirnotch(frecuencia, q, fs)
    return signal.filtfilt(b, a, señal)


def calcular_psd(señal, fs):
    """Estima la densidad espectral de potencia con el método de Welch."""
    return signal.welch(señal, fs=fs, nperseg=min(8192, len(señal)))


def analizar_50hz(frecuencia, psd):
    """
    Cuantifica la interferencia de red: toma la PSD en el punto más cercano a 50 Hz y la
    compara con la mediana de la PSD vecina (45-55 Hz, excluyendo 49-51 Hz).
    Una relación mucho mayor que 1 indica presencia de la componente de 50 Hz.
    Es solo un apoyo para inspeccionar el gráfico: no decide si se aplica el notch.
    """
    indice = np.argmin(np.abs(frecuencia - 50))
    vecindad = (
        (frecuencia >= 45) & (frecuencia <= 55)
        & ((frecuencia < 49) | (frecuencia > 51))
    )
    nivel_local = np.median(psd[vecindad])
    return {
        "frecuencia": frecuencia[indice],
        "potencia": psd[indice],
        "nivel_local": nivel_local,
        "relacion": psd[indice] / nivel_local if nivel_local > 0 else np.nan,
    }


def detectar_saturacion(señal, limite_inferior, limite_superior):
    """Marca las muestras que tocan los límites de adquisición (señal recortada)."""
    return (señal <= limite_inferior) | (señal >= limite_superior)


def encontrar_segmentos(mascara):
    """Agrupa muestras consecutivas marcadas en segmentos; devuelve inicio y fin (inclusive)."""
    bordes = np.diff(np.concatenate(([0], mascara.astype(np.int8), [0])))
    return np.flatnonzero(bordes == 1), np.flatnonzero(bordes == -1) - 1


def informar_preprocesamiento(titulo, media, ecg_hp, analisis_50hz, saturacion, fs):
    """Resume por consola el resultado del preprocesamiento de un registro."""
    encabezado(f"PREPROCESAMIENTO - {titulo}")
    print(f"Media original: {media:.6f} V | Media tras pasa-altos: {np.mean(ecg_hp):.6f} V")

    print(
        f"PSD en {analisis_50hz['frecuencia']:.3f} Hz: {analisis_50hz['potencia']:.3e} V²/Hz | "
        f"Nivel local: {analisis_50hz['nivel_local']:.3e} V²/Hz | "
        f"Relación: {analisis_50hz['relacion']:.3f}"
    )
    print(f"Filtro notch de {F_NOTCH} Hz: {'aplicado' if APLICAR_NOTCH else 'no aplicado'}")

    inicios, finales = encontrar_segmentos(saturacion)
    print(
        f"Muestras próximas a saturación: {saturacion.sum()} "
        f"({100 * saturacion.mean():.4f} %) en {len(inicios)} segmentos"
    )
    if len(inicios):
        print(f"Mayor duración continua de saturación: {(finales - inicios + 1).max() / fs:.4f} s")


# ============================================================
# ETAPA 3 - FUNCIONES DE ANÁLISIS DE PAN-TOMPKINS
# ============================================================

def senal_integrada_pan_tompkins(ecg, fs):
    """
    Etapas lineales y no lineales de Pan-Tompkins que realzan el complejo QRS:
    pasa-banda 5-15 Hz (concentra la energía del QRS), derivada (pendiente de la onda R),
    elevación al cuadrado (valores positivos y énfasis en pendientes altas) e integración
    con ventana móvil de 150 ms (envolvente del QRS). Devuelve cada señal intermedia.
    """
    sos = signal.butter(2, [5.0, 15.0], btype="bandpass", fs=fs, output="sos")
    bandpass = signal.sosfiltfilt(sos, ecg)

    kernel_derivada = np.array([1, 2, 0, -2, -1]) * fs / 8.0
    derivada = signal.convolve(bandpass, kernel_derivada, mode="same")

    cuadrado = derivada ** 2

    n_ventana = max(1, int(0.150 * fs))
    integrado = signal.convolve(cuadrado, np.ones(n_ventana) / n_ventana, mode="same")

    return bandpass, derivada, cuadrado, integrado


def umbrales_adaptativos(npki, spki):
    """Umbral principal (25 % entre ruido y señal) y umbral secundario para el search-back."""
    umbral = npki + 0.25 * (spki - npki)
    return umbral, 0.5 * umbral


def detectar_qrs_adaptativo(integrado, fs, refractario):
    """
    Decide qué máximos de la señal integrada son QRS mediante umbral adaptativo.

    - SPKI / NPKI: niveles estimados de señal (QRS) y de ruido, actualizados con un
      filtro exponencial (alpha = 0.125) cada vez que se clasifica un máximo.
    - Inicialización: percentiles 75 y 25 de los máximos de los primeros 2 s.
    - Un máximo es QRS si supera el umbral y respeta el período refractario (200 ms).
    - Search-back: si pasó más de 1.66 veces el RR promedio (últimos 8 latidos) sin
      detectar un QRS, se recupera el máximo más grande entre los descartados que
      supere el umbral secundario.

    Devuelve (candidatos, picos_qrs, picos_searchback, umbral, umbral_secundario);
    los umbrales se interpolan a lo largo de todo el registro para poder graficarlos.
    """
    alpha = 0.125
    candidatos, _ = signal.find_peaks(integrado, distance=refractario)

    if len(candidatos) == 0:
        vacio = np.array([], dtype=int)
        ceros = np.zeros(len(integrado))
        return candidatos, vacio, vacio, ceros, ceros.copy()

    amplitudes = integrado[candidatos]

    iniciales = amplitudes[candidatos <= int(2.0 * fs)]
    if len(iniciales) < 3:
        iniciales = amplitudes[:8]

    npki = np.percentile(iniciales, 25)
    spki = np.percentile(iniciales, 75)
    if spki <= npki:
        spki = np.max(iniciales)
    umbral, umbral2 = umbrales_adaptativos(npki, spki)

    picos_qrs, picos_searchback = [], []
    ruido = []                      # Candidatos descartados que aún podrían recuperarse
    rr_recientes = deque(maxlen=8)  # RR [muestras] de los últimos latidos
    ultimo_qrs = None

    umbral_en_candidatos = np.empty(len(candidatos))
    umbral2_en_candidatos = np.empty(len(candidatos))

    for k, (pico, amplitud) in enumerate(zip(candidatos, amplitudes)):

        # Search-back: ¿falta un latido entre el último QRS y este candidato?
        if ultimo_qrs is not None and len(rr_recientes) >= 2:
            while pico - ultimo_qrs > 1.66 * np.mean(rr_recientes):
                elegibles = [
                    (p, a) for p, a in ruido
                    if ultimo_qrs + refractario < p < pico - refractario and a >= umbral2
                ]
                if not elegibles:
                    break

                pico_sb, amplitud_sb = max(elegibles, key=lambda item: item[1])
                picos_qrs.append(pico_sb)
                picos_searchback.append(pico_sb)
                rr_recientes.append(pico_sb - ultimo_qrs)
                ultimo_qrs = pico_sb

                spki = alpha * amplitud_sb + (1 - alpha) * spki
                umbral, umbral2 = umbrales_adaptativos(npki, spki)
                ruido = [item for item in ruido if item[0] > pico_sb]

        umbral_en_candidatos[k] = umbral
        umbral2_en_candidatos[k] = umbral2

        # Clasificación del candidato actual
        if amplitud >= umbral and (ultimo_qrs is None or pico - ultimo_qrs >= refractario):
            picos_qrs.append(pico)
            if ultimo_qrs is not None:
                rr_recientes.append(pico - ultimo_qrs)
            ultimo_qrs = pico
            spki = alpha * amplitud + (1 - alpha) * spki
            ruido.clear()  # Los descartados anteriores ya no pueden recuperarse
        else:
            ruido.append((pico, amplitud))
            npki = alpha * amplitud + (1 - alpha) * npki

        umbral, umbral2 = umbrales_adaptativos(npki, spki)

    muestras = np.arange(len(integrado))
    return (
        candidatos,
        np.array(sorted(set(picos_qrs)), dtype=int),
        np.array(sorted(set(picos_searchback)), dtype=int),
        np.interp(muestras, candidatos, umbral_en_candidatos),
        np.interp(muestras, candidatos, umbral2_en_candidatos),
    )


def localizar_picos_r(bandpass, picos_qrs, fs, refractario, mascara_saturacion=None):
    """
    Ubica el pico R exacto: el máximo de |señal pasa-banda| en ±150 ms de cada QRS detectado
    en la señal integrada (que está levemente desplazada). Se descartan los picos con saturación
    en ±100 ms, y si dos picos quedan a menos de un período refractario se conserva el mayor.
    """
    semiventana = int(0.150 * fs)
    margen_saturacion = int(0.10 * fs)
    picos = []

    for pico in picos_qrs:
        inicio = max(0, pico - semiventana)
        fin = min(len(bandpass), pico + semiventana + 1)
        pico_r = inicio + np.argmax(np.abs(bandpass[inicio:fin]))

        if mascara_saturacion is not None and mascara_saturacion[
            max(0, pico_r - margen_saturacion): pico_r + margen_saturacion + 1
        ].any():
            continue

        picos.append(pico_r)

    limpios = []
    for pico in np.sort(picos).astype(int):
        if limpios and pico - limpios[-1] < refractario:
            if abs(bandpass[pico]) > abs(bandpass[limpios[-1]]):
                limpios[-1] = pico
        else:
            limpios.append(pico)

    return np.array(limpios, dtype=int)


def pan_tompkins(ecg, fs, mascara_saturacion=None):
    """
    Detecta los complejos QRS con el algoritmo de Pan-Tompkins y calcula los intervalos RR
    [s] y la frecuencia cardíaca instantánea [lat/min] entre picos R consecutivos.
    Devuelve un diccionario con las señales intermedias, los picos y los resultados.
    """
    ecg = np.asarray(ecg, dtype=float).ravel()
    refractario = int(0.200 * fs)

    bandpass, derivada, cuadrado, integrado = senal_integrada_pan_tompkins(ecg, fs)

    candidatos, picos_qrs, picos_searchback, umbral, umbral2 = detectar_qrs_adaptativo(
        integrado, fs, refractario
    )

    picos_r = localizar_picos_r(bandpass, picos_qrs, fs, refractario, mascara_saturacion)

    rr = np.diff(picos_r) / fs
    fc = 60.0 / rr if len(rr) else np.array([])

    return {
        "bandpass": bandpass,
        "derivada": derivada,
        "cuadrado": cuadrado,
        "integrado": integrado,
        "picos_candidatos": candidatos,
        "picos_integrados": picos_qrs,
        "picos_searchback": picos_searchback,
        "picos_r": picos_r,
        "rr": rr,
        "fc": fc,
        "umbral": umbral,
        "umbral_secundario": umbral2,
    }


def informar_deteccion(titulo, res):
    """Imprime la cantidad de QRS detectados y estadísticos de RR y frecuencia cardíaca."""
    encabezado(f"PAN-TOMPKINS - {titulo}")
    print(f"Complejos QRS detectados: {len(res['picos_r'])}")
    print(f"Detecciones recuperadas por search-back: {len(res['picos_searchback'])}")

    if len(res["rr"]):
        fc = res["fc"]
        print(f"RR medio: {np.mean(res['rr']):.3f} s")
        print(f"FC media: {np.mean(fc):.2f} | mínima: {np.min(fc):.2f} | máxima: {np.max(fc):.2f} lat/min")


# ============================================================
# ETAPA 4 - FUNCIONES DE ANÁLISIS DEL TACOGRAMA
# ============================================================

def construir_tacograma(picos_r, fs, mascara_saturacion=None):
    """
    Arma el tacograma (intervalo RR en función del tiempo) a partir de los picos R.
    Cada RR se asocia al instante del segundo pico R que lo delimita. Un intervalo que
    atraviesa una zona saturada se conserva pero queda marcado como no válido.
    Devuelve (tiempo_rr, rr, validos).
    """
    picos_r = np.asarray(picos_r, dtype=int)

    if len(picos_r) < 2:
        return np.array([]), np.array([]), np.array([], dtype=bool)

    rr = np.diff(picos_r) / fs
    tiempo_rr = picos_r[1:] / fs
    validos = np.ones(len(rr), dtype=bool)

    if mascara_saturacion is not None:
        # Suma acumulada: cantidad de muestras saturadas en cada intervalo [pico_i, pico_i+1]
        acumulado = np.concatenate(([0], np.cumsum(mascara_saturacion)))
        validos = (acumulado[picos_r[1:] + 1] - acumulado[picos_r[:-1]]) == 0

    return tiempo_rr, rr, validos


def informar_tacograma(titulo, rr, validos):
    """Imprime cuántos intervalos RR son válidos y sus valores medio, mínimo y máximo."""
    encabezado(f"TACOGRAMA - {titulo}")
    print(
        f"Intervalos RR: {len(rr)} | válidos: {validos.sum()} | "
        f"descartados por saturación: {(~validos).sum()}"
    )
    if validos.any():
        rr_validos = rr[validos]
        print(
            f"RR medio: {np.mean(rr_validos):.3f} s | "
            f"mínimo: {np.min(rr_validos):.3f} s | máximo: {np.max(rr_validos):.3f} s"
        )


# ============================================================
# ETAPA 5 - FUNCIONES DE ANÁLISIS DEL PERIODOGRAMA DE LOMB
# ============================================================

def preparar_tacograma_lomb(tiempo_rr, rr, validos):
    """
    Conserva solo los RR válidos y finitos y desplaza el tiempo para que comience en cero.
    Lomb admite muestreo no uniforme, por lo que no hace falta interpolar el tacograma
    (los picos R no ocurren en instantes equiespaciados).
    """
    mascara = np.asarray(validos, dtype=bool) & np.isfinite(tiempo_rr) & np.isfinite(rr)
    tiempo, rr_validos = tiempo_rr[mascara], rr[mascara]

    if len(rr_validos) < 3:
        raise ValueError(
            "No hay suficientes intervalos RR válidos para calcular el periodograma de Lomb."
        )

    return tiempo - tiempo[0], rr_validos


def calcular_lomb(tiempo, rr, omega):
    """
    Periodograma de Lomb normalizado de los intervalos RR.
    Antes del cálculo se resta la tendencia lineal (y con ella la media): en el registro
    post actividad la frecuencia cardíaca se recupera de forma progresiva, y esa deriva
    taparía las oscilaciones de interés. omega es la frecuencia angular [rad/s].
    """
    tendencia = np.polyval(np.polyfit(tiempo, rr, 1), tiempo)
    rr_sin_tendencia = rr - tendencia
    rr_sin_tendencia -= np.mean(rr_sin_tendencia)

    return signal.lombscargle(tiempo, rr_sin_tendencia, omega, normalize=True)


def potencia_en_banda(frecuencias, potencia, f_min, f_max):
    """Área bajo el periodograma en la banda [f_min, f_max) (regla del trapecio)."""
    mascara = (frecuencias >= f_min) & (frecuencias < f_max)
    if mascara.sum() < 2:
        return np.nan
    return trapezoid(potencia[mascara], frecuencias[mascara])


def resumen_espectral(frecuencias, potencia):
    """
    Frecuencia y período dominantes del periodograma, potencia en las bandas LF y HF
    y relación LF/HF (indicador del balance simpático-vagal).
    """
    indice = np.argmax(potencia)
    lf = potencia_en_banda(frecuencias, potencia, LF_MIN, LF_MAX)
    hf = potencia_en_banda(frecuencias, potencia, HF_MIN, HF_MAX)

    return {
        "Frecuencia_dominante_Hz": frecuencias[indice],
        "Periodo_dominante_s": 1.0 / frecuencias[indice],
        "Potencia_max": potencia[indice],
        "Potencia_LF": lf,
        "Potencia_HF": hf,
        "Relacion_LF_HF": lf / hf if hf > 0 else np.nan,
    }


def informar_lomb(titulo, resumen):
    """Imprime la frecuencia dominante y la distribución de potencia LF/HF."""
    encabezado(f"PERIODOGRAMA DE LOMB - {titulo}")
    print(
        f"Frecuencia dominante: {resumen['Frecuencia_dominante_Hz']:.4f} Hz "
        f"(período {resumen['Periodo_dominante_s']:.2f} s) | "
        f"Potencia normalizada máxima: {resumen['Potencia_max']:.4f}"
    )
    print(
        f"Potencia LF: {resumen['Potencia_LF']:.6f} | "
        f"Potencia HF: {resumen['Potencia_HF']:.6f} | "
        f"LF/HF: {resumen['Relacion_LF_HF']:.4f}"
    )


# ============================================================
# ETAPAS DEL ANÁLISIS
# ============================================================

def etapa_visualizacion(registros, pdf):
    """Etapa 1: estadísticas básicas y gráficos de cada registro adquirido (completo y ampliado)."""
    rango_zoom = (INICIO_ZOOM, INICIO_ZOOM + DURACION_ZOOM)

    for estado, ecg in registros.items():
        titulo = ESTADOS[estado]
        curva = [{"x": vector_tiempo(len(ecg), FS), "y": ecg}]

        mostrar_estadisticas(titulo, ecg, FS)
        graficar(pdf, curva, f"{titulo} - Señal adquirida")
        graficar(pdf, curva, f"{titulo} - Ventana ampliada", rango_x=rango_zoom)


def etapa_preprocesamiento(registros, pdf):
    """
    Etapa 2: quita continua y deriva (pasa-altos), analiza la interferencia de 50 Hz
    (notch opcional) y detecta saturaciones sobre la señal original.
    Devuelve, por estado, la señal preprocesada y la máscara de saturación.
    """
    resultados = {}

    for estado, ecg in registros.items():
        titulo = ESTADOS[estado]
        t = vector_tiempo(len(ecg), FS)

        media = np.mean(ecg)
        ecg_hp = filtro_pasa_altos(ecg, FS, FC_HP, ORDEN_HP)
        f, psd = calcular_psd(ecg_hp, FS)
        analisis_50hz = analizar_50hz(f, psd)
        ecg_final = filtro_notch(ecg_hp, FS, F_NOTCH, Q_NOTCH) if APLICAR_NOTCH else ecg_hp
        saturacion = detectar_saturacion(ecg, LIMITE_INFERIOR, LIMITE_SUPERIOR)

        informar_preprocesamiento(titulo, media, ecg_hp, analisis_50hz, saturacion, FS)
        graficar_preprocesamiento(pdf, titulo, t, ecg, ecg_hp, ecg_final, media, f, psd, saturacion)
        guardar_csv(
            f"ecg_data_{estado}_procesado.csv",
            {
                "Tiempo_s": t,
                "ECG_original_V": ecg,
                "ECG_preprocesado_V": ecg_final,
                "Saturacion": saturacion.astype(int),
            },
        )

        resultados[estado] = {"ecg": ecg_final, "saturacion": saturacion}

    return resultados


def etapa_pan_tompkins(preprocesados, pdf):
    """Etapa 3 (inciso 5): detecta los QRS de cada registro; devuelve el resultado completo por estado."""
    detecciones = {}

    for estado, datos in preprocesados.items():
        titulo = ESTADOS[estado]
        ecg = datos["ecg"]

        res = pan_tompkins(ecg, FS, datos["saturacion"])

        informar_deteccion(titulo, res)
        graficar_etapas_pan_tompkins(pdf, ecg, res, FS, titulo)
        graficar_picos_r(pdf, ecg, res["picos_r"], FS, f"Detección de picos R - {titulo}")
        graficar_frecuencia_cardiaca(
            pdf, res["picos_r"], res["fc"], FS, f"Frecuencia cardíaca instantánea - {titulo}"
        )

        detecciones[estado] = res

    return detecciones


def etapa_tacograma(detecciones, preprocesados, pdf):
    """Etapa 4 (inciso 6): construye el tacograma de cada registro y marca los RR no válidos."""
    tacogramas = {}

    for estado, res in detecciones.items():
        titulo = ESTADOS[estado]

        tiempo, rr, validos = construir_tacograma(
            res["picos_r"], FS, preprocesados[estado]["saturacion"]
        )

        informar_tacograma(titulo, rr, validos)
        graficar_tacograma(pdf, tiempo, rr, validos, f"Tacograma - {titulo}")
        guardar_csv(
            f"tacograma_{estado}.csv",
            {"Tiempo_s": tiempo, "RR_s": rr, "RR_ms": rr * 1000, "Valido": validos.astype(int)},
        )

        tacogramas[estado] = {"tiempo": tiempo, "rr": rr, "validos": validos}

    return tacogramas


def etapa_lomb(tacogramas, pdf):
    """
    Etapa 5 (inciso 7): periodograma de Lomb de cada tacograma sobre una grilla de frecuencias
    común. La frecuencia mínima es 1/T del registro más corto (límite de resolución), para que
    reposo y post actividad se puedan comparar directamente.
    """
    preparados = {
        estado: preparar_tacograma_lomb(tac["tiempo"], tac["rr"], tac["validos"])
        for estado, tac in tacogramas.items()
    }

    duraciones = {estado: t[-1] - t[0] for estado, (t, _) in preparados.items()}
    f_min = 1.0 / min(duraciones.values())

    encabezado("CONFIGURACIÓN DEL PERIODOGRAMA DE LOMB")
    for estado, duracion in duraciones.items():
        print(f"Duración útil {ESTADOS[estado]}: {duracion:.2f} s")
    print(f"Rango de frecuencias analizado: {f_min:.4f} - {F_MAX_LOMB:.2f} Hz")

    frecuencias = np.linspace(f_min, F_MAX_LOMB, N_FRECUENCIAS_LOMB)
    omega = 2 * np.pi * frecuencias  # lombscargle trabaja en rad/s

    potencias, resumenes = {}, []
    for estado, (tiempo, rr) in preparados.items():
        titulo = ESTADOS[estado]
        potencia = calcular_lomb(tiempo, rr, omega)
        resumen = resumen_espectral(frecuencias, potencia)

        informar_lomb(titulo, resumen)
        graficar_lomb(
            pdf, frecuencias, [("Periodograma de Lomb", potencia)],
            f"Periodograma de Lomb - {titulo}", resumen["Frecuencia_dominante_Hz"],
        )

        potencias[estado] = potencia
        resumenes.append({"Estado": titulo, **resumen})

    graficar_lomb(
        pdf, frecuencias, [(ESTADOS[e], p) for e, p in potencias.items()],
        "Comparación de los periodogramas de Lomb",
    )

    guardar_csv(
        "periodograma_lomb.csv",
        {"Frecuencia_Hz": frecuencias, **{f"Lomb_{e.capitalize()}": p for e, p in potencias.items()}},
    )
    guardar_csv("resumen_lomb.csv", resumenes)


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    registros = {estado: cargar_ecg(ruta) for estado, ruta in ARCHIVOS_ECG.items()}

    with PdfPages(ARCHIVO_FIGURAS) as pdf:
        etapa_visualizacion(registros, pdf)
        preprocesados = etapa_preprocesamiento(registros, pdf)
        detecciones = etapa_pan_tompkins(preprocesados, pdf)
        tacogramas = etapa_tacograma(detecciones, preprocesados, pdf)
        etapa_lomb(tacogramas, pdf)

    print(f"\nFiguras guardadas en: {ARCHIVO_FIGURAS}")


if __name__ == "__main__":
    main()