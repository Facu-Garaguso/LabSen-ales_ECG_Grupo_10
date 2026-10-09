import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, welch, iirnotch, filtfilt
from scipy import signal


#
# ============================================================
# CONFIGURACIÓN
# ============================================================

FS = 480  # Frecuencia de muestreo [Hz]

ARCHIVO_PRE = "ecg_data_pre.csv"
ARCHIVO_POST = "ecg_data_post.csv"

CARPETA_IMAGENES = Path("Imagenes")
CARPETA_IMAGENES.mkdir(exist_ok=True)

# Ventana que se mostrará en los gráficos ampliados
INICIO_ZOOM = 0       # segundos
DURACION_ZOOM = 10    # segundos


# ============================================================
# CARGA DE DATOS
# ============================================================

def cargar_ecg(nombre_archivo):
    """
    Carga un archivo CSV de ECG y devuelve la señal como vector NumPy.
    Se espera una columna denominada 'Sample'.
    """

    datos = pd.read_csv(nombre_archivo)

    # Elimina posibles espacios en los nombres de las columnas
    datos.columns = datos.columns.str.strip()

    if "Sample" not in datos.columns:
        raise ValueError(
            f"El archivo {nombre_archivo} no contiene una columna llamada 'Sample'. "
            f"Columnas encontradas: {list(datos.columns)}"
        )

    señal = pd.to_numeric(datos["Sample"], errors="coerce").to_numpy()

    # Elimina posibles valores no numéricos
    señal = señal[np.isfinite(señal)]

    return señal


ecg_pre = cargar_ecg(ARCHIVO_PRE)
ecg_post = cargar_ecg(ARCHIVO_POST)


# ============================================================
# VECTORES DE TIEMPO
# ============================================================

t_pre = np.arange(len(ecg_pre)) / FS
t_post = np.arange(len(ecg_post)) / FS


# ============================================================
# INFORMACIÓN BÁSICA DE LOS REGISTROS
# ============================================================

print("\n================ ECG PRE ================")
print(f"Número de muestras: {len(ecg_pre)}")
print(f"Duración: {len(ecg_pre) / FS:.2f} s")
print(f"Duración: {len(ecg_pre) / FS / 60:.2f} min")
print(f"Valor medio: {np.mean(ecg_pre):.4f} V")
print(f"Valor mínimo: {np.min(ecg_pre):.4f} V")
print(f"Valor máximo: {np.max(ecg_pre):.4f} V")
print(f"Desvío estándar: {np.std(ecg_pre):.4f} V")

print("\n=============== ECG POST ================")
print(f"Número de muestras: {len(ecg_post)}")
print(f"Duración: {len(ecg_post) / FS:.2f} s")
print(f"Duración: {len(ecg_post) / FS / 60:.2f} min")
print(f"Valor medio: {np.mean(ecg_post):.4f} V")
print(f"Valor mínimo: {np.min(ecg_post):.4f} V")
print(f"Valor máximo: {np.max(ecg_post):.4f} V")
print(f"Desvío estándar: {np.std(ecg_post):.4f} V")


# ============================================================
# FUNCIÓN PARA GRAFICAR SEÑAL COMPLETA
# ============================================================

def graficar_completa(t, señal, titulo, nombre_imagen):

    plt.figure(figsize=(14, 5))

    plt.plot(t, señal, linewidth=0.8)

    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [V]")
    plt.title(titulo)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    ruta = CARPETA_IMAGENES / nombre_imagen
    plt.savefig(ruta, dpi=300, bbox_inches="tight")

    plt.show()


# ============================================================
# FUNCIÓN PARA GRAFICAR UNA VENTANA AMPLIADA
# ============================================================

def graficar_zoom(
    t,
    señal,
    titulo,
    nombre_imagen,
    inicio=0,
    duracion=10
):

    muestra_inicial = int(inicio * FS)
    muestra_final = int((inicio + duracion) * FS)

    muestra_final = min(muestra_final, len(señal))

    t_zoom = t[muestra_inicial:muestra_final]
    señal_zoom = señal[muestra_inicial:muestra_final]

    plt.figure(figsize=(14, 5))

    plt.plot(t_zoom, señal_zoom, linewidth=1)

    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [V]")
    plt.title(titulo)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    ruta = CARPETA_IMAGENES / nombre_imagen
    plt.savefig(ruta, dpi=300, bbox_inches="tight")

    plt.show()


# ============================================================
# ECG PRE - SEÑAL COMPLETA
# ============================================================

graficar_completa(
    t_pre,
    ecg_pre,
    "ECG en reposo - Señal adquirida",
    "ECG_Pre_Completo.png"
)


# ============================================================
# ECG POST - SEÑAL COMPLETA
# ============================================================

graficar_completa(
    t_post,
    ecg_post,
    "ECG post actividad física - Señal adquirida",
    "ECG_Post_Completo.png"
)


# ============================================================
# ECG PRE - VENTANA AMPLIADA
# ============================================================

graficar_zoom(
    t_pre,
    ecg_pre,
    "ECG en reposo - Ventana ampliada",
    "ECG_Pre_Zoom.png",
    inicio=INICIO_ZOOM,
    duracion=DURACION_ZOOM
)


# ============================================================
# ECG POST - VENTANA AMPLIADA
# ============================================================

graficar_zoom(
    t_post,
    ecg_post,
    "ECG post actividad física - Ventana ampliada",
    "ECG_Post_Zoom.png",
    inicio=INICIO_ZOOM,
    duracion=DURACION_ZOOM
)

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

FS = 480  # Frecuencia de muestreo [Hz]

ARCHIVO_PRE = "ecg_data_pre.csv"
ARCHIVO_POST = "ecg_data_post.csv"

CARPETA_IMAGENES = Path("Imagenes")
CARPETA_IMAGENES.mkdir(exist_ok=True)


# ============================================================
# PARÁMETROS DEL PREPROCESAMIENTO
# ============================================================

# Filtro pasa-altos para continua y deriva
FC_HP = 0.5       # Hz
ORDEN_HP = 4

# Frecuencia de la red eléctrica
F_NOTCH = 50      # Hz
Q_NOTCH = 30

# IMPORTANTE:
# Primero analizar los espectros.
# Cambiar a True únicamente si se confirma interferencia de 50 Hz.
APLICAR_NOTCH = False


# Límites utilizados para detectar posibles saturaciones
LIMITE_INFERIOR = 0.01
LIMITE_SUPERIOR = 3.29


# Ventana utilizada para los gráficos ampliados
INICIO_ZOOM = 0       # s
DURACION_ZOOM = 10    # s


# ============================================================
# CARGA DE DATOS
# ============================================================

def cargar_ecg(nombre_archivo):
    """
    Carga un archivo CSV que contiene una columna llamada Sample.
    """

    datos = pd.read_csv(nombre_archivo)

    datos.columns = datos.columns.str.strip()

    if "Sample" not in datos.columns:
        raise ValueError(
            f"El archivo {nombre_archivo} no contiene la columna 'Sample'. "
            f"Columnas encontradas: {list(datos.columns)}"
        )

    señal = pd.to_numeric(
        datos["Sample"],
        errors="coerce"
    ).to_numpy()

    señal = señal[np.isfinite(señal)]

    return señal


ecg_pre = cargar_ecg(ARCHIVO_PRE)
ecg_post = cargar_ecg(ARCHIVO_POST)


# ============================================================
# VECTORES DE TIEMPO
# ============================================================

t_pre = np.arange(len(ecg_pre)) / FS
t_post = np.arange(len(ecg_post)) / FS


# ============================================================
# FUNCIÓN PARA GUARDAR FIGURAS
# ============================================================

def guardar_figura(nombre):
    """
    Guarda la figura actual dentro de la carpeta Imagenes.
    """

    ruta = CARPETA_IMAGENES / nombre

    plt.tight_layout()

    plt.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Figura guardada: {ruta}")


# ============================================================
# INFORMACIÓN BÁSICA DE LOS REGISTROS
# ============================================================

def mostrar_estadisticas(nombre, señal):

    print(f"\n================ {nombre} ================")

    print(f"Número de muestras: {len(señal)}")
    print(f"Duración: {len(señal) / FS:.2f} s")
    print(f"Duración: {len(señal) / FS / 60:.2f} min")

    print(f"Media: {np.mean(señal):.6f} V")
    print(f"Mínimo: {np.min(señal):.6f} V")
    print(f"Máximo: {np.max(señal):.6f} V")
    print(f"Desvío estándar: {np.std(señal):.6f} V")


mostrar_estadisticas(
    "ECG PRE",
    ecg_pre
)

mostrar_estadisticas(
    "ECG POST",
    ecg_post
)


# ============================================================
# ETAPA 1
# ANÁLISIS DE LA COMPONENTE CONTINUA
# ============================================================

media_pre = np.mean(ecg_pre)
media_post = np.mean(ecg_post)

print("\n========== COMPONENTE CONTINUA ==========")

print(f"Media PRE:  {media_pre:.6f} V")
print(f"Media POST: {media_post:.6f} V")


def graficar_offset(
    tiempo,
    señal,
    media,
    titulo,
    nombre
):

    plt.figure(figsize=(14, 5))

    plt.plot(
        tiempo,
        señal,
        linewidth=0.7,
        label="ECG adquirido"
    )

    plt.axhline(
        media,
        linestyle="--",
        label=f"Valor medio = {media:.3f} V"
    )

    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [V]")
    plt.title(titulo)

    plt.legend()
    plt.grid(True, alpha=0.3)

    guardar_figura(nombre)

    plt.show()
    plt.close()


graficar_offset(
    t_pre,
    ecg_pre,
    media_pre,
    "ECG en reposo - Componente continua",
    "ECG_Pre_Offset.png"
)

graficar_offset(
    t_post,
    ecg_post,
    media_post,
    "ECG post actividad física - Componente continua",
    "ECG_Post_Offset.png"
)


# ============================================================
# ETAPA 2
# FILTRO PASA-ALTOS PARA CONTINUA Y DERIVA
# ============================================================

def filtro_pasa_altos(
    señal,
    fs,
    fc=0.5,
    orden=4
):
    """
    Diseña un filtro Butterworth IIR pasa-altos y lo aplica
    hacia adelante y hacia atrás mediante filtfilt.

    De esta manera se evita introducir un corrimiento de fase.
    """

    b, a = butter(
        orden,
        fc,
        btype="highpass",
        fs=fs
    )

    señal_filtrada = filtfilt(
        b,
        a,
        señal
    )

    return señal_filtrada


ecg_pre_hp = filtro_pasa_altos(
    ecg_pre,
    FS,
    FC_HP,
    ORDEN_HP
)

ecg_post_hp = filtro_pasa_altos(
    ecg_post,
    FS,
    FC_HP,
    ORDEN_HP
)


print("\n========== FILTRO PASA-ALTOS ==========")

print(
    f"Media PRE después del filtrado: "
    f"{np.mean(ecg_pre_hp):.6f} V"
)

print(
    f"Media POST después del filtrado: "
    f"{np.mean(ecg_post_hp):.6f} V"
)


# ============================================================
# COMPARACIÓN ANTES Y DESPUÉS DEL PASA-ALTOS
# ============================================================

def graficar_pasa_altos(
    tiempo,
    original,
    filtrada,
    media_original,
    titulo,
    nombre
):
    """
    Para poder comparar ambas señales sobre el mismo eje,
    la señal original se centra únicamente para la visualización.

    La señal utilizada realmente para el procesamiento pasa
    directamente por el filtro pasa-altos.
    """

    original_centrada = original - media_original

    plt.figure(figsize=(14, 6))

    plt.plot(
        tiempo,
        original_centrada,
        linewidth=0.6,
        label="Original centrada para comparación"
    )

    plt.plot(
        tiempo,
        filtrada,
        linewidth=0.8,
        label="Pasa-altos 0.5 Hz"
    )

    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [V]")
    plt.title(titulo)

    plt.legend()
    plt.grid(True, alpha=0.3)

    guardar_figura(nombre)

    plt.show()
    plt.close()


graficar_pasa_altos(
    t_pre,
    ecg_pre,
    ecg_pre_hp,
    media_pre,
    "ECG en reposo - Corrección de línea de base",
    "ECG_Pre_PasaAltos.png"
)

graficar_pasa_altos(
    t_post,
    ecg_post,
    ecg_post_hp,
    media_post,
    "ECG post actividad física - Corrección de línea de base",
    "ECG_Post_PasaAltos.png"
)


# ============================================================
# ZOOM DEL EFECTO DEL PASA-ALTOS
# ============================================================

def graficar_zoom_pasa_altos(
    tiempo,
    original,
    filtrada,
    media_original,
    titulo,
    nombre,
    inicio=0,
    duracion=10
):

    i0 = int(inicio * FS)
    i1 = int((inicio + duracion) * FS)

    i1 = min(
        i1,
        len(original)
    )

    original_centrada = original - media_original

    plt.figure(figsize=(14, 6))

    plt.plot(
        tiempo[i0:i1],
        original_centrada[i0:i1],
        linewidth=0.8,
        label="Original centrada"
    )

    plt.plot(
        tiempo[i0:i1],
        filtrada[i0:i1],
        linewidth=1,
        label="Después del pasa-altos"
    )

    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [V]")
    plt.title(titulo)

    plt.legend()
    plt.grid(True, alpha=0.3)

    guardar_figura(nombre)

    plt.show()
    plt.close()


graficar_zoom_pasa_altos(
    t_pre,
    ecg_pre,
    ecg_pre_hp,
    media_pre,
    "ECG en reposo - Efecto del pasa-altos",
    "ECG_Pre_PasaAltos_Zoom.png",
    INICIO_ZOOM,
    DURACION_ZOOM
)

graficar_zoom_pasa_altos(
    t_post,
    ecg_post,
    ecg_post_hp,
    media_post,
    "ECG post actividad física - Efecto del pasa-altos",
    "ECG_Post_PasaAltos_Zoom.png",
    INICIO_ZOOM,
    DURACION_ZOOM
)


# ============================================================
# ETAPA 3
# ANÁLISIS ESPECTRAL
# ============================================================

def calcular_psd(
    señal,
    fs
):
    """
    Estima la densidad espectral de potencia mediante
    el método de Welch.
    """

    frecuencia, psd = welch(
        señal,
        fs=fs,
        nperseg=min(8192, len(señal))
    )

    return frecuencia, psd


f_pre, psd_pre = calcular_psd(
    ecg_pre_hp,
    FS
)

f_post, psd_post = calcular_psd(
    ecg_post_hp,
    FS
)


# ============================================================
# PSD ENTRE 0 Y 100 Hz
# ============================================================

def graficar_psd(
    frecuencia,
    psd,
    titulo,
    nombre
):

    mascara = (
        (frecuencia >= 0)
        &
        (frecuencia <= 100)
    )

    plt.figure(figsize=(12, 5))

    plt.semilogy(
        frecuencia[mascara],
        psd[mascara]
    )

    plt.axvline(
        50,
        linestyle="--",
        label="50 Hz"
    )

    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("PSD [V²/Hz]")
    plt.title(titulo)

    plt.legend()
    plt.grid(True, alpha=0.3)

    guardar_figura(nombre)

    plt.show()
    plt.close()


graficar_psd(
    f_pre,
    psd_pre,
    "ECG en reposo - Densidad espectral de potencia",
    "ECG_Pre_PSD.png"
)

graficar_psd(
    f_post,
    psd_post,
    "ECG post actividad física - Densidad espectral de potencia",
    "ECG_Post_PSD.png"
)


# ============================================================
# ZOOM ESPECTRAL ENTRE 40 Y 60 Hz
# ============================================================

def graficar_50hz(
    frecuencia,
    psd,
    titulo,
    nombre
):

    mascara = (
        (frecuencia >= 40)
        &
        (frecuencia <= 60)
    )

    plt.figure(figsize=(12, 5))

    plt.semilogy(
        frecuencia[mascara],
        psd[mascara]
    )

    plt.axvline(
        50,
        linestyle="--",
        label="50 Hz"
    )

    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("PSD [V²/Hz]")
    plt.title(titulo)

    plt.legend()
    plt.grid(True, alpha=0.3)

    guardar_figura(nombre)

    plt.show()
    plt.close()


graficar_50hz(
    f_pre,
    psd_pre,
    "ECG en reposo - Análisis alrededor de 50 Hz",
    "ECG_Pre_PSD_50Hz.png"
)

graficar_50hz(
    f_post,
    psd_post,
    "ECG post actividad física - Análisis alrededor de 50 Hz",
    "ECG_Post_PSD_50Hz.png"
)


# ============================================================
# CUANTIFICACIÓN DE LA COMPONENTE DE 50 Hz
# ============================================================

def analizar_50hz(
    frecuencia,
    psd
):
    """
    Obtiene la PSD correspondiente al punto más cercano
    a 50 Hz y la compara con el nivel local del espectro.

    Este valor se utiliza únicamente como apoyo para la
    inspección del gráfico. No decide automáticamente
    si debe aplicarse un notch.
    """

    indice_50 = np.argmin(
        np.abs(frecuencia - 50)
    )

    f_real = frecuencia[indice_50]
    potencia_50 = psd[indice_50]

    # Región vecina entre 45 y 55 Hz
    # excluyendo aproximadamente 49-51 Hz
    mascara_vecina = (
        (frecuencia >= 45)
        &
        (frecuencia <= 55)
        &
        (
            (frecuencia < 49)
            |
            (frecuencia > 51)
        )
    )

    nivel_local = np.median(
        psd[mascara_vecina]
    )

    if nivel_local > 0:
        relacion = potencia_50 / nivel_local
    else:
        relacion = np.nan

    return (
        f_real,
        potencia_50,
        nivel_local,
        relacion
    )


(
    f50_pre,
    potencia50_pre,
    nivel50_pre,
    relacion50_pre
) = analizar_50hz(
    f_pre,
    psd_pre
)

(
    f50_post,
    potencia50_post,
    nivel50_post,
    relacion50_post
) = analizar_50hz(
    f_post,
    psd_post
)


print("\n========== ANÁLISIS DE 50 Hz ==========")

print("\nPRE")

print(
    f"Frecuencia evaluada: "
    f"{f50_pre:.3f} Hz"
)

print(
    f"PSD en 50 Hz: "
    f"{potencia50_pre:.6e} V²/Hz"
)

print(
    f"Nivel espectral local: "
    f"{nivel50_pre:.6e} V²/Hz"
)

print(
    f"Relación 50 Hz / nivel local: "
    f"{relacion50_pre:.3f}"
)


print("\nPOST")

print(
    f"Frecuencia evaluada: "
    f"{f50_post:.3f} Hz"
)

print(
    f"PSD en 50 Hz: "
    f"{potencia50_post:.6e} V²/Hz"
)

print(
    f"Nivel espectral local: "
    f"{nivel50_post:.6e} V²/Hz"
)

print(
    f"Relación 50 Hz / nivel local: "
    f"{relacion50_post:.3f}"
)


# ============================================================
# ETAPA 4
# FILTRO NOTCH OPCIONAL
# ============================================================

def filtro_notch(
    señal,
    fs,
    frecuencia=50,
    q=30
):
    """
    Filtro rechaza-banda estrecho centrado en 50 Hz.
    Se aplica con filtfilt para evitar corrimiento de fase.
    """

    b, a = iirnotch(
        frecuencia,
        q,
        fs
    )

    señal_filtrada = filtfilt(
        b,
        a,
        señal
    )

    return señal_filtrada


if APLICAR_NOTCH:

    ecg_pre_final = filtro_notch(
        ecg_pre_hp,
        FS,
        F_NOTCH,
        Q_NOTCH
    )

    ecg_post_final = filtro_notch(
        ecg_post_hp,
        FS,
        F_NOTCH,
        Q_NOTCH
    )

    print("\nSe aplicó filtro notch de 50 Hz")

else:

    ecg_pre_final = ecg_pre_hp.copy()
    ecg_post_final = ecg_post_hp.copy()

    print("\nNo se aplicó filtro notch de 50 Hz")


# ============================================================
# ETAPA 5
# DETECCIÓN DE SATURACIONES
# ============================================================

def detectar_saturacion(
    señal,
    limite_inferior,
    limite_superior
):
    """
    Marca muestras cercanas a los límites de adquisición.
    """

    mascara = (
        (señal <= limite_inferior)
        |
        (señal >= limite_superior)
    )

    return mascara


sat_pre = detectar_saturacion(
    ecg_pre,
    LIMITE_INFERIOR,
    LIMITE_SUPERIOR
)

sat_post = detectar_saturacion(
    ecg_post,
    LIMITE_INFERIOR,
    LIMITE_SUPERIOR
)


# ============================================================
# IDENTIFICACIÓN DE SEGMENTOS CONTIGUOS SATURADOS
# ============================================================

def encontrar_segmentos(
    mascara,
    fs
):
    """
    Agrupa muestras saturadas consecutivas en segmentos.
    """

    indices = np.flatnonzero(mascara)

    if len(indices) == 0:
        return []

    cortes = np.where(
        np.diff(indices) > 1
    )[0]

    inicios = np.r_[
        indices[0],
        indices[cortes + 1]
    ]

    finales = np.r_[
        indices[cortes],
        indices[-1]
    ]

    segmentos = []

    for inicio, final in zip(
        inicios,
        finales
    ):

        cantidad = final - inicio + 1

        duracion = cantidad / fs

        segmentos.append(
            {
                "inicio_muestra": inicio,
                "fin_muestra": final,
                "inicio_s": inicio / fs,
                "fin_s": final / fs,
                "duracion_s": duracion
            }
        )

    return segmentos


segmentos_pre = encontrar_segmentos(
    sat_pre,
    FS
)

segmentos_post = encontrar_segmentos(
    sat_post,
    FS
)


print("\n========== SATURACIÓN ==========")

print("\nPRE")

print(
    f"Muestras próximas a saturación: "
    f"{np.sum(sat_pre)}"
)

print(
    f"Porcentaje: "
    f"{100 * np.mean(sat_pre):.4f} %"
)

print(
    f"Segmentos encontrados: "
    f"{len(segmentos_pre)}"
)


print("\nPOST")

print(
    f"Muestras próximas a saturación: "
    f"{np.sum(sat_post)}"
)

print(
    f"Porcentaje: "
    f"{100 * np.mean(sat_post):.4f} %"
)

print(
    f"Segmentos encontrados: "
    f"{len(segmentos_post)}"
)


if len(segmentos_post) > 0:

    duracion_maxima = max(
        segmento["duracion_s"]
        for segmento in segmentos_post
    )

    print(
        f"Mayor duración continua de saturación POST: "
        f"{duracion_maxima:.4f} s"
    )


# ============================================================
# VISUALIZACIÓN DE SATURACIONES
# ============================================================

def graficar_saturacion(
    tiempo,
    señal,
    mascara,
    titulo,
    nombre
):

    plt.figure(figsize=(14, 5))

    plt.plot(
        tiempo,
        señal,
        linewidth=0.7,
        label="ECG"
    )

    if np.any(mascara):

        plt.scatter(
            tiempo[mascara],
            señal[mascara],
            s=12,
            label="Muestras próximas a saturación"
        )

    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [V]")
    plt.title(titulo)

    plt.legend()
    plt.grid(True, alpha=0.3)

    guardar_figura(nombre)

    plt.show()
    plt.close()


graficar_saturacion(
    t_pre,
    ecg_pre,
    sat_pre,
    "ECG en reposo - Evaluación de saturación",
    "ECG_Pre_Saturacion.png"
)

graficar_saturacion(
    t_post,
    ecg_post,
    sat_post,
    "ECG post actividad física - Evaluación de saturación",
    "ECG_Post_Saturacion.png"
)


# ============================================================
# ETAPA 6
# VISUALIZACIÓN DE LA SEÑAL FINAL
# ============================================================

def graficar_final(
    tiempo,
    señal,
    titulo,
    nombre
):

    plt.figure(figsize=(14, 5))

    plt.plot(
        tiempo,
        señal,
        linewidth=0.8
    )

    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [V]")
    plt.title(titulo)

    plt.grid(True, alpha=0.3)

    guardar_figura(nombre)

    plt.show()
    plt.close()


graficar_final(
    t_pre,
    ecg_pre_final,
    "ECG en reposo - Señal preprocesada",
    "ECG_Pre_Procesado.png"
)

graficar_final(
    t_post,
    ecg_post_final,
    "ECG post actividad física - Señal preprocesada",
    "ECG_Post_Procesado.png"
)


# ============================================================
# ZOOM DE LA SEÑAL FINAL
# ============================================================

def graficar_zoom_final(
    tiempo,
    señal,
    titulo,
    nombre,
    inicio=0,
    duracion=10
):

    i0 = int(inicio * FS)

    i1 = int(
        (inicio + duracion) * FS
    )

    i1 = min(
        i1,
        len(señal)
    )

    plt.figure(figsize=(14, 5))

    plt.plot(
        tiempo[i0:i1],
        señal[i0:i1],
        linewidth=1
    )

    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [V]")
    plt.title(titulo)

    plt.grid(True, alpha=0.3)

    guardar_figura(nombre)

    plt.show()
    plt.close()


graficar_zoom_final(
    t_pre,
    ecg_pre_final,
    "ECG en reposo - Señal preprocesada",
    "ECG_Pre_Procesado_Zoom.png",
    INICIO_ZOOM,
    DURACION_ZOOM
)

graficar_zoom_final(
    t_post,
    ecg_post_final,
    "ECG post actividad física - Señal preprocesada",
    "ECG_Post_Procesado_Zoom.png",
    INICIO_ZOOM,
    DURACION_ZOOM
)


# ============================================================
# GUARDADO DE LOS RESULTADOS
# ============================================================

datos_pre = pd.DataFrame(
    {
        "Tiempo_s": t_pre,
        "ECG_original_V": ecg_pre,
        "ECG_preprocesado_V": ecg_pre_final,
        "Saturacion": sat_pre.astype(int)
    }
)

datos_post = pd.DataFrame(
    {
        "Tiempo_s": t_post,
        "ECG_original_V": ecg_post,
        "ECG_preprocesado_V": ecg_post_final,
        "Saturacion": sat_post.astype(int)
    }
)


datos_pre.to_csv(
    "ecg_data_pre_procesado.csv",
    index=False
)

datos_post.to_csv(
    "ecg_data_post_procesado.csv",
    index=False
)


print("\n========================================")
print("PREPROCESAMIENTO FINALIZADO")
print("========================================")

print("Archivos generados:")
print("ecg_data_pre_procesado.csv")
print("ecg_data_post_procesado.csv")

# ============================================================
# INCISO 5 - DETECCIÓN DE COMPLEJOS QRS MEDIANTE PAN-TOMPKINS
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal


# ============================================================
# CARPETA DE SALIDA DE IMÁGENES
# ============================================================

CARPETA_IMAGENES = Path("Imagenes")
CARPETA_IMAGENES.mkdir(exist_ok=True)


# ============================================================
# FUNCIÓN PAN-TOMPKINS CON UMBRAL ADAPTATIVO Y SEARCH-BACK
# ============================================================

def pan_tompkins(ecg, fs, mascara_saturacion=None):
    """
    Implementación de Pan-Tompkins adaptada al presente trabajo.

    Etapas:
    1) filtrado pasa-banda 5-15 Hz,
    2) derivada,
    3) elevación al cuadrado,
    4) integración por ventana móvil de 150 ms,
    5) detección de máximos candidatos,
    6) umbral adaptativo SPKI/NPKI,
    7) search-back para recuperar latidos de menor amplitud,
    8) localización del pico R sobre la señal pasa-banda,
    9) exclusión de detecciones próximas a saturación.
    """

    ecg = np.asarray(ecg, dtype=float).flatten()

    if mascara_saturacion is not None:
        mascara_saturacion = np.asarray(
            mascara_saturacion,
            dtype=bool
        )

    # --------------------------------------------------------
    # 1. FILTRO PASA-BANDA
    # --------------------------------------------------------

    f_low = 5.0
    f_high = 15.0

    b, a = signal.butter(
        2,
        [f_low, f_high],
        btype="bandpass",
        fs=fs
    )

    ecg_bp = signal.filtfilt(
        b,
        a,
        ecg
    )

    # --------------------------------------------------------
    # 2. FILTRO DERIVATIVO
    # --------------------------------------------------------

    kernel_derivada = np.array(
        [1, 2, 0, -2, -1]
    ) * fs / 8.0

    ecg_derivada = signal.convolve(
        ecg_bp,
        kernel_derivada,
        mode="same"
    )

    # --------------------------------------------------------
    # 3. ELEVACIÓN AL CUADRADO
    # --------------------------------------------------------

    ecg_cuadrado = ecg_derivada ** 2

    # --------------------------------------------------------
    # 4. INTEGRACIÓN POR VENTANA MÓVIL
    # --------------------------------------------------------

    ventana_seg = 0.150
    N = max(1, int(ventana_seg * fs))
    ventana = np.ones(N) / N

    ecg_integrado = signal.convolve(
        ecg_cuadrado,
        ventana,
        mode="same"
    )

    # --------------------------------------------------------
    # 5. PICOS CANDIDATOS
    # --------------------------------------------------------
    # Se usa un período refractario de 200 ms. La decisión de
    # si un máximo pertenece a QRS o ruido se toma luego con
    # el umbral adaptativo.

    periodo_refractario = int(0.200 * fs)

    picos_candidatos, _ = signal.find_peaks(
        ecg_integrado,
        distance=periodo_refractario
    )

    if len(picos_candidatos) == 0:
        return {
            "bandpass": ecg_bp,
            "derivada": ecg_derivada,
            "cuadrado": ecg_cuadrado,
            "integrado": ecg_integrado,
            "picos_candidatos": np.array([], dtype=int),
            "picos_integrados": np.array([], dtype=int),
            "picos_searchback": np.array([], dtype=int),
            "picos_r": np.array([], dtype=int),
            "rr": np.array([]),
            "fc": np.array([]),
            "umbral": np.zeros(len(ecg)),
            "umbral_secundario": np.zeros(len(ecg))
        }

    amplitudes = ecg_integrado[picos_candidatos]

    # --------------------------------------------------------
    # 6. INICIALIZACIÓN DEL UMBRAL ADAPTATIVO
    # --------------------------------------------------------
    # Se utilizan los primeros 2 s para estimar inicialmente
    # el nivel de ruido (NPKI) y el nivel de señal QRS (SPKI).
    # Si hubiera pocos máximos en ese intervalo se emplean los
    # primeros candidatos disponibles.

    fin_inicializacion = int(2.0 * fs)
    mascara_inicial = picos_candidatos <= fin_inicializacion
    amplitudes_iniciales = amplitudes[mascara_inicial]

    if len(amplitudes_iniciales) < 3:
        amplitudes_iniciales = amplitudes[:min(8, len(amplitudes))]

    NPKI = np.percentile(amplitudes_iniciales, 25)
    SPKI = np.percentile(amplitudes_iniciales, 75)

    if SPKI <= NPKI:
        SPKI = np.max(amplitudes_iniciales)

    THRESHOLD_I1 = NPKI + 0.25 * (SPKI - NPKI)
    THRESHOLD_I2 = 0.5 * THRESHOLD_I1

    # Factores clásicos de actualización exponencial.
    alpha = 0.125

    picos_integrados = []
    picos_searchback = []

    # Se guardan candidatos clasificados inicialmente como ruido
    # para que puedan recuperarse mediante search-back.
    candidatos_ruido = []

    # Historial RR en muestras para estimar cuándo falta un latido.
    rr_historial = []
    ultimo_qrs = None

    # Umbral usado en cada máximo candidato, para poder mostrar
    # gráficamente cómo se adapta a lo largo del registro.
    umbral_en_candidatos = []
    umbral2_en_candidatos = []

    for pico, amplitud in zip(picos_candidatos, amplitudes):

        # ----------------------------------------------------
        # SEARCH-BACK
        # ----------------------------------------------------
        # Si el tiempo desde el último QRS supera 1.66 veces el
        # RR de referencia, se revisan los picos que habían sido
        # clasificados como ruido. Se acepta el de mayor amplitud
        # que supere el umbral secundario.

        if ultimo_qrs is not None and len(rr_historial) >= 2:

            rr_referencia = np.mean(rr_historial[-8:])
            rr_limite = 1.66 * rr_referencia

            while pico - ultimo_qrs > rr_limite:

                elegibles = [
                    item for item in candidatos_ruido
                    if (
                        item[0] > ultimo_qrs + periodo_refractario
                        and item[0] < pico - periodo_refractario
                        and item[1] >= THRESHOLD_I2
                    )
                ]

                if len(elegibles) == 0:
                    break

                pico_sb, amplitud_sb = max(
                    elegibles,
                    key=lambda item: item[1]
                )

                picos_integrados.append(pico_sb)
                picos_searchback.append(pico_sb)

                nuevo_rr = pico_sb - ultimo_qrs
                rr_historial.append(nuevo_rr)
                ultimo_qrs = pico_sb

                SPKI = (
                    alpha * amplitud_sb
                    + (1 - alpha) * SPKI
                )

                THRESHOLD_I1 = NPKI + 0.25 * (SPKI - NPKI)
                THRESHOLD_I2 = 0.5 * THRESHOLD_I1

                candidatos_ruido = [
                    item for item in candidatos_ruido
                    if item[0] != pico_sb
                ]

                rr_referencia = np.mean(rr_historial[-8:])
                rr_limite = 1.66 * rr_referencia

        # Guardar el umbral que corresponde a este instante.
        umbral_en_candidatos.append(THRESHOLD_I1)
        umbral2_en_candidatos.append(THRESHOLD_I2)

        # ----------------------------------------------------
        # CLASIFICACIÓN DEL CANDIDATO ACTUAL
        # ----------------------------------------------------

        supera_umbral = amplitud >= THRESHOLD_I1

        respeta_refractario = (
            ultimo_qrs is None
            or pico - ultimo_qrs >= periodo_refractario
        )

        if supera_umbral and respeta_refractario:

            picos_integrados.append(pico)

            if ultimo_qrs is not None:
                rr_historial.append(pico - ultimo_qrs)

            ultimo_qrs = pico

            SPKI = (
                alpha * amplitud
                + (1 - alpha) * SPKI
            )

        else:

            candidatos_ruido.append((pico, amplitud))

            NPKI = (
                alpha * amplitud
                + (1 - alpha) * NPKI
            )

        THRESHOLD_I1 = NPKI + 0.25 * (SPKI - NPKI)
        THRESHOLD_I2 = 0.5 * THRESHOLD_I1

    # Ordenar y eliminar duplicados que pudieran provenir del search-back.
    picos_integrados = np.array(
        sorted(set(picos_integrados)),
        dtype=int
    )

    picos_searchback = np.array(
        sorted(set(picos_searchback)),
        dtype=int
    )

    # --------------------------------------------------------
    # TRAZA TEMPORAL DEL UMBRAL
    # --------------------------------------------------------

    umbral_en_candidatos = np.asarray(
        umbral_en_candidatos,
        dtype=float
    )

    umbral2_en_candidatos = np.asarray(
        umbral2_en_candidatos,
        dtype=float
    )

    indices = np.arange(len(ecg_integrado))

    if len(picos_candidatos) == 1:
        umbral_traza = np.full(
            len(ecg_integrado),
            umbral_en_candidatos[0]
        )
        umbral2_traza = np.full(
            len(ecg_integrado),
            umbral2_en_candidatos[0]
        )
    else:
        umbral_traza = np.interp(
            indices,
            picos_candidatos,
            umbral_en_candidatos,
            left=umbral_en_candidatos[0],
            right=umbral_en_candidatos[-1]
        )

        umbral2_traza = np.interp(
            indices,
            picos_candidatos,
            umbral2_en_candidatos,
            left=umbral2_en_candidatos[0],
            right=umbral2_en_candidatos[-1]
        )

    # --------------------------------------------------------
    # 7. LOCALIZACIÓN DEL PICO R EN LA SEÑAL PASA-BANDA
    # --------------------------------------------------------

    ventana_busqueda = int(0.150 * fs)
    picos_r = []

    for pico in picos_integrados:

        inicio = max(
            0,
            pico - ventana_busqueda
        )

        fin = min(
            len(ecg_bp),
            pico + ventana_busqueda + 1
        )

        segmento = ecg_bp[inicio:fin]

        if len(segmento) == 0:
            continue

        indice_local = np.argmax(
            np.abs(segmento)
        )

        pico_r = inicio + indice_local

        # ----------------------------------------------------
        # EXCLUSIÓN DE ZONAS SATURADAS
        # ----------------------------------------------------
        # Si existe saturación en +/-100 ms alrededor del pico,
        # la detección no se utiliza como pico R válido.

        if mascara_saturacion is not None:

            sat_inicio = max(
                0,
                pico_r - int(0.10 * fs)
            )

            sat_fin = min(
                len(mascara_saturacion),
                pico_r + int(0.10 * fs) + 1
            )

            if np.any(
                mascara_saturacion[sat_inicio:sat_fin]
            ):
                continue

        picos_r.append(pico_r)

    picos_r = np.asarray(
        picos_r,
        dtype=int
    )

    # --------------------------------------------------------
    # 8. ELIMINACIÓN DE DETECCIONES DUPLICADAS
    # --------------------------------------------------------

    if len(picos_r) > 1:

        picos_r = np.sort(picos_r)
        picos_limpios = [picos_r[0]]

        for pico in picos_r[1:]:

            anterior = picos_limpios[-1]

            if pico - anterior >= periodo_refractario:
                picos_limpios.append(pico)

            else:
                # Si dos detecciones quedan demasiado próximas,
                # conservar la de mayor amplitud en el pasa-banda.
                if abs(ecg_bp[pico]) > abs(ecg_bp[anterior]):
                    picos_limpios[-1] = pico

        picos_r = np.asarray(
            picos_limpios,
            dtype=int
        )

    # --------------------------------------------------------
    # 9. INTERVALOS RR Y FRECUENCIA CARDÍACA
    # --------------------------------------------------------

    rr = np.diff(picos_r) / fs

    if len(rr) > 0:
        fc = 60.0 / rr
    else:
        fc = np.array([])

    return {
        "bandpass": ecg_bp,
        "derivada": ecg_derivada,
        "cuadrado": ecg_cuadrado,
        "integrado": ecg_integrado,
        "picos_candidatos": picos_candidatos,
        "picos_integrados": picos_integrados,
        "picos_searchback": picos_searchback,
        "picos_r": picos_r,
        "rr": rr,
        "fc": fc,
        "umbral": umbral_traza,
        "umbral_secundario": umbral2_traza
    }


# ============================================================
# APLICACIÓN A LOS DOS ECG
# ============================================================

resultado_pre = pan_tompkins(
    ecg_pre_final,
    FS,
    mascara_saturacion=sat_pre
)

resultado_post = pan_tompkins(
    ecg_post_final,
    FS,
    mascara_saturacion=sat_post
)


# ============================================================
# EXTRAER RESULTADOS
# ============================================================

picos_r_pre = resultado_pre["picos_r"]
picos_r_post = resultado_post["picos_r"]

rr_pre = resultado_pre["rr"]
rr_post = resultado_post["rr"]

fc_pre = resultado_pre["fc"]
fc_post = resultado_post["fc"]


# ============================================================
# RESULTADOS NUMÉRICOS DE PAN-TOMPKINS
# ============================================================

print("\n==============================")
print("ECG PRE-ACTIVIDAD")
print("==============================")
print(
    "Cantidad de complejos QRS detectados:",
    len(picos_r_pre)
)
print(
    "Detecciones recuperadas por search-back:",
    len(resultado_pre["picos_searchback"])
)

if len(rr_pre) > 0:
    print(f"RR medio: {np.mean(rr_pre):.3f} s")
    print(f"FC media: {np.mean(fc_pre):.2f} lat/min")
    print(f"FC mínima: {np.min(fc_pre):.2f} lat/min")
    print(f"FC máxima: {np.max(fc_pre):.2f} lat/min")

print("\n==============================")
print("ECG POST-ACTIVIDAD")
print("==============================")
print(
    "Cantidad de complejos QRS detectados:",
    len(picos_r_post)
)
print(
    "Detecciones recuperadas por search-back:",
    len(resultado_post["picos_searchback"])
)

if len(rr_post) > 0:
    print(f"RR medio: {np.mean(rr_post):.3f} s")
    print(f"FC media: {np.mean(fc_post):.2f} lat/min")
    print(f"FC mínima: {np.min(fc_post):.2f} lat/min")
    print(f"FC máxima: {np.max(fc_post):.2f} lat/min")


# ============================================================
# FUNCIÓN AUXILIAR PARA GUARDAR IMÁGENES
# ============================================================

def guardar_imagen(nombre_archivo):
    ruta = CARPETA_IMAGENES / nombre_archivo

    plt.tight_layout()

    plt.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Imagen guardada en: {ruta}")


# ============================================================
# GRÁFICO DE LAS ETAPAS DE PAN-TOMPKINS
# ============================================================

def graficar_pan_tompkins(
    ecg,
    resultado,
    fs,
    titulo,
    nombre_archivo
):

    t = np.arange(len(ecg)) / fs

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(14, 11),
        sharex=True
    )

    # ECG preprocesado
    axes[0].plot(
        t,
        ecg,
        linewidth=0.8
    )
    axes[0].set_ylabel("Amplitud")
    axes[0].set_title(f"{titulo} - ECG procesado")
    axes[0].grid(True, alpha=0.3)

    # Pasa-banda
    axes[1].plot(
        t,
        resultado["bandpass"],
        linewidth=0.8
    )
    axes[1].set_ylabel("Amplitud")
    axes[1].set_title("Filtrado pasa-banda 5-15 Hz")
    axes[1].grid(True, alpha=0.3)

    # Derivada
    axes[2].plot(
        t,
        resultado["derivada"],
        linewidth=0.8
    )
    axes[2].set_ylabel("Amplitud")
    axes[2].set_title("Filtro derivativo")
    axes[2].grid(True, alpha=0.3)

    # Cuadrado
    axes[3].plot(
        t,
        resultado["cuadrado"],
        linewidth=0.8
    )
    axes[3].set_ylabel("Amplitud")
    axes[3].set_title("Elevación al cuadrado")
    axes[3].grid(True, alpha=0.3)

    # Integración y umbral adaptativo
    axes[4].plot(
        t,
        resultado["integrado"],
        linewidth=0.8,
        label="Señal integrada"
    )

    axes[4].plot(
        t,
        resultado["umbral"],
        linestyle="--",
        linewidth=1.2,
        label="Umbral adaptativo"
    )

    if len(resultado["picos_integrados"]) > 0:
        picos = resultado["picos_integrados"]

        axes[4].scatter(
            picos / fs,
            resultado["integrado"][picos],
            marker="x",
            s=45,
            label="QRS detectados"
        )

    if len(resultado["picos_searchback"]) > 0:
        picos_sb = resultado["picos_searchback"]

        axes[4].scatter(
            picos_sb / fs,
            resultado["integrado"][picos_sb],
            marker="o",
            facecolors="none",
            s=70,
            label="Recuperados por search-back"
        )

    axes[4].set_ylabel("Amplitud")
    axes[4].set_xlabel("Tiempo [s]")
    axes[4].set_title("Integración por ventana móvil y umbral adaptativo")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()

    guardar_imagen(nombre_archivo)
    plt.show()
    plt.close(fig)


# ============================================================
# GRAFICAR ETAPAS DE PAN-TOMPKINS
# ============================================================

graficar_pan_tompkins(
    ecg_pre_final,
    resultado_pre,
    FS,
    "ECG pre-actividad",
    "PanTompkins_Etapas_Pre.png"
)

graficar_pan_tompkins(
    ecg_post_final,
    resultado_post,
    FS,
    "ECG post-actividad",
    "PanTompkins_Etapas_Post.png"
)


# ============================================================
# FUNCIÓN PARA GRAFICAR PICOS R
# ============================================================

def graficar_picos_r(
    ecg,
    picos_r,
    fs,
    titulo,
    nombre_archivo
):

    t = np.arange(len(ecg)) / fs

    plt.figure(figsize=(14, 5))

    plt.plot(
        t,
        ecg,
        linewidth=0.8,
        label="ECG procesado"
    )

    if len(picos_r) > 0:
        plt.scatter(
            picos_r / fs,
            ecg[picos_r],
            marker="x",
            s=60,
            label="Picos R",
            zorder=3
        )

    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.title(titulo)
    plt.grid(True, alpha=0.3)
    plt.legend()

    guardar_imagen(nombre_archivo)
    plt.show()
    plt.close()


graficar_picos_r(
    ecg_pre_final,
    picos_r_pre,
    FS,
    "Detección de picos R - ECG pre-actividad",
    "PanTompkins_Pre.png"
)

graficar_picos_r(
    ecg_post_final,
    picos_r_post,
    FS,
    "Detección de picos R - ECG post-actividad",
    "PanTompkins_Post.png"
)


# ============================================================
# FUNCIÓN PARA GRAFICAR FRECUENCIA CARDÍACA
# ============================================================

def graficar_frecuencia_cardiaca(
    picos_r,
    fc,
    fs,
    titulo,
    nombre_archivo
):

    if len(fc) == 0:
        print(
            f"No hay suficientes detecciones para generar "
            f"{nombre_archivo}"
        )
        return

    tiempo_fc = picos_r[1:] / fs

    plt.figure(figsize=(12, 4))

    plt.plot(
        tiempo_fc,
        fc,
        marker="o",
        markersize=4,
        linewidth=1
    )

    plt.xlabel("Tiempo [s]")
    plt.ylabel("Frecuencia cardíaca [lat/min]")
    plt.title(titulo)
    plt.grid(True, alpha=0.3)

    guardar_imagen(nombre_archivo)
    plt.show()
    plt.close()


graficar_frecuencia_cardiaca(
    picos_r_pre,
    fc_pre,
    FS,
    "Frecuencia cardíaca instantánea - Pre-actividad",
    "FC_Pre.png"
)

graficar_frecuencia_cardiaca(
    picos_r_post,
    fc_post,
    FS,
    "Frecuencia cardíaca instantánea - Post-actividad",
    "FC_Post.png"
)


# ============================================================
# INCISO 6 - TACOGRAMAS
# ============================================================


def construir_tacograma(
    picos_r,
    fs,
    mascara_saturacion=None
):
    """
    Construye el tacograma a partir de los picos R válidos.

    Cada intervalo RR se asocia temporalmente al segundo pico R
    que lo delimita. Si un intervalo atraviesa una región de
    saturación, se conserva para referencia pero se marca como
    no válido para el análisis fisiológico.
    """

    picos_r = np.asarray(
        picos_r,
        dtype=int
    )

    if len(picos_r) < 2:
        return (
            np.array([]),
            np.array([]),
            np.array([], dtype=bool)
        )

    rr = np.diff(picos_r) / fs
    tiempo_rr = picos_r[1:] / fs

    validos = np.ones(
        len(rr),
        dtype=bool
    )

    if mascara_saturacion is not None:

        mascara_saturacion = np.asarray(
            mascara_saturacion,
            dtype=bool
        )

        for i in range(len(rr)):

            inicio = picos_r[i]
            fin = picos_r[i + 1] + 1

            if np.any(
                mascara_saturacion[inicio:fin]
            ):
                validos[i] = False

    return (
        tiempo_rr,
        rr,
        validos
    )


# ============================================================
# CONSTRUCCIÓN DE LOS TACOGRAMAS
# ============================================================

tiempo_rr_pre, rr_tac_pre, validos_pre = construir_tacograma(
    picos_r_pre,
    FS,
    mascara_saturacion=sat_pre
)

tiempo_rr_post, rr_tac_post, validos_post = construir_tacograma(
    picos_r_post,
    FS,
    mascara_saturacion=sat_post
)


# ============================================================
# INFORMACIÓN DE LOS TACOGRAMAS
# ============================================================

print("\n==============================")
print("TACOGRAMA PRE-ACTIVIDAD")
print("==============================")
print(
    "Cantidad total de intervalos RR:",
    len(rr_tac_pre)
)
print(
    "Intervalos RR válidos:",
    np.sum(validos_pre)
)
print(
    "Intervalos RR descartados por saturación:",
    np.sum(~validos_pre)
)

if np.any(validos_pre):
    rr_pre_validos = rr_tac_pre[validos_pre]
    print(f"RR medio: {np.mean(rr_pre_validos):.3f} s")
    print(f"RR mínimo: {np.min(rr_pre_validos):.3f} s")
    print(f"RR máximo: {np.max(rr_pre_validos):.3f} s")

print("\n==============================")
print("TACOGRAMA POST-ACTIVIDAD")
print("==============================")
print(
    "Cantidad total de intervalos RR:",
    len(rr_tac_post)
)
print(
    "Intervalos RR válidos:",
    np.sum(validos_post)
)
print(
    "Intervalos RR descartados por saturación:",
    np.sum(~validos_post)
)

if np.any(validos_post):
    rr_post_validos = rr_tac_post[validos_post]
    print(f"RR medio: {np.mean(rr_post_validos):.3f} s")
    print(f"RR mínimo: {np.min(rr_post_validos):.3f} s")
    print(f"RR máximo: {np.max(rr_post_validos):.3f} s")


# ============================================================
# FUNCIÓN PARA GRAFICAR TACOGRAMA
# ============================================================

def graficar_tacograma(
    tiempo_rr,
    rr,
    validos,
    titulo,
    nombre_archivo
):

    if len(rr) == 0:
        print(
            f"No hay suficientes picos R para generar "
            f"{nombre_archivo}"
        )
        return

    plt.figure(figsize=(12, 4.5))

    # Para evitar unir con una línea dos puntos separados por un
    # intervalo inválido, se construye una copia con NaN.
    rr_grafico = rr.astype(float).copy()
    rr_grafico[~validos] = np.nan

    plt.plot(
        tiempo_rr,
        rr_grafico * 1000,
        marker="o",
        markersize=4,
        linewidth=1,
        label="Intervalos RR válidos"
    )

    if np.any(~validos):
        plt.scatter(
            tiempo_rr[~validos],
            rr[~validos] * 1000,
            marker="x",
            s=50,
            label="Intervalos afectados por saturación"
        )

    plt.xlabel("Tiempo [s]")
    plt.ylabel("Intervalo RR [ms]")
    plt.title(titulo)
    plt.grid(True, alpha=0.3)
    plt.legend()

    guardar_imagen(nombre_archivo)
    plt.show()
    plt.close()


graficar_tacograma(
    tiempo_rr_pre,
    rr_tac_pre,
    validos_pre,
    "Tacograma - ECG en reposo",
    "Tacograma_Pre.png"
)

graficar_tacograma(
    tiempo_rr_post,
    rr_tac_post,
    validos_post,
    "Tacograma - ECG post actividad física",
    "Tacograma_Post.png"
)


# ============================================================
# GUARDADO DE LOS TACOGRAMAS EN CSV
# ============================================================

tacograma_pre = pd.DataFrame(
    {
        "Tiempo_s": tiempo_rr_pre,
        "RR_s": rr_tac_pre,
        "RR_ms": rr_tac_pre * 1000,
        "Valido": validos_pre.astype(int)
    }
)

tacograma_post = pd.DataFrame(
    {
        "Tiempo_s": tiempo_rr_post,
        "RR_s": rr_tac_post,
        "RR_ms": rr_tac_post * 1000,
        "Valido": validos_post.astype(int)
    }
)

tacograma_pre.to_csv(
    "tacograma_pre.csv",
    index=False
)

tacograma_post.to_csv(
    "tacograma_post.csv",
    index=False
)


# ============================================================
# GUARDADO Y VERIFICACIÓN FINAL DE TODAS LAS IMÁGENES
# ============================================================
# Cada función de graficado guarda la figura en el momento en
# que se genera. Esta sección final verifica que todas las
# imágenes correspondientes a Pan-Tompkins y al tacograma hayan
# quedado efectivamente dentro de la carpeta Imagenes del repo.

IMAGENES_ANALISIS = [
    "PanTompkins_Etapas_Pre.png",
    "PanTompkins_Etapas_Post.png",
    "PanTompkins_Pre.png",
    "PanTompkins_Post.png",
    "FC_Pre.png",
    "FC_Post.png",
    "Tacograma_Pre.png",
    "Tacograma_Post.png"
]

print("\n========================================")
print("IMÁGENES GENERADAS")
print("========================================")

for nombre_imagen in IMAGENES_ANALISIS:

    ruta = CARPETA_IMAGENES / nombre_imagen

    if ruta.exists():
        print(f"OK: {ruta}")
    else:
        print(f"FALTA: {ruta}")

print("\nArchivos CSV generados:")
print("tacograma_pre.csv")
print("tacograma_post.csv")

# ============================================================
# INCISO 7 - PERIODOGRAMA DE LOMB
# ============================================================
#
# El periodograma de Lomb permite estimar el contenido
# frecuencial de una señal muestreada de manera no uniforme.
#
# En este caso la señal analizada está formada por los
# intervalos RR del tacograma. Como los picos R no ocurren
# exactamente en instantes equiespaciados, los RR tampoco
# constituyen una señal uniformemente muestreada.
#
# Por este motivo no es necesario interpolar previamente
# el tacograma para aplicar Lomb.
# ============================================================


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Frecuencia máxima analizada.
#
# Para variabilidad de frecuencia cardíaca resulta suficiente
# estudiar hasta aproximadamente 0.5 Hz.

F_MAX_LOMB = 0.50

# Cantidad de frecuencias utilizadas para evaluar
# el periodograma.

N_FRECUENCIAS_LOMB = 5000


# ============================================================
# BANDAS DE FRECUENCIA DE INTERÉS
# ============================================================
#
# Se incluyen principalmente para facilitar posteriormente
# la comparación entre reposo y post-actividad.
#
# LF: 0.04 - 0.15 Hz
# HF: 0.15 - 0.40 Hz
#
# No se utiliza la banda VLF para realizar una interpretación
# cuantitativa debido a la corta duración del registro en reposo.
# ============================================================

LF_MIN = 0.04
LF_MAX = 0.15

HF_MIN = 0.15
HF_MAX = 0.40


# ============================================================
# PREPARAR DATOS PARA LOMB
# ============================================================

def preparar_tacograma_lomb(
    tiempo_rr,
    rr,
    validos
):
    """
    Selecciona únicamente los intervalos RR válidos.

    También elimina posibles NaN o infinitos.

    Finalmente desplaza el vector temporal para que comience
    en cero.
    """

    tiempo_rr = np.asarray(
        tiempo_rr,
        dtype=float
    )

    rr = np.asarray(
        rr,
        dtype=float
    )

    validos = np.asarray(
        validos,
        dtype=bool
    )


    # --------------------------------------------------------
    # MÁSCARA DE DATOS VÁLIDOS
    # --------------------------------------------------------

    mascara = (
        validos
        &
        np.isfinite(tiempo_rr)
        &
        np.isfinite(rr)
    )


    tiempo = tiempo_rr[
        mascara
    ]

    rr_validos = rr[
        mascara
    ]


    if len(rr_validos) < 3:

        raise ValueError(
            "No hay suficientes intervalos RR válidos "
            "para calcular el periodograma de Lomb."
        )


    # --------------------------------------------------------
    # HACER QUE EL TIEMPO COMIENCE EN CERO
    # --------------------------------------------------------

    tiempo = (
        tiempo
        -
        tiempo[0]
    )


    return (
        tiempo,
        rr_validos
    )


# ============================================================
# PREPARAR TACOGRAMAS PRE Y POST
# ============================================================

tiempo_lomb_pre, rr_lomb_pre = preparar_tacograma_lomb(
    tiempo_rr_pre,
    rr_tac_pre,
    validos_pre
)


tiempo_lomb_post, rr_lomb_post = preparar_tacograma_lomb(
    tiempo_rr_post,
    rr_tac_post,
    validos_post
)


# ============================================================
# FRECUENCIA MÍNIMA DEL ANÁLISIS
# ============================================================
#
# La resolución frecuencial depende de la duración del registro.
#
# Para poder comparar PRE y POST sobre la misma grilla de
# frecuencias se toma como referencia el registro de menor
# duración.
#
# Aproximadamente:
#
#       f_min ~ 1 / T
#
# ============================================================

duracion_lomb_pre = (
    tiempo_lomb_pre[-1]
    -
    tiempo_lomb_pre[0]
)


duracion_lomb_post = (
    tiempo_lomb_post[-1]
    -
    tiempo_lomb_post[0]
)


duracion_minima = min(
    duracion_lomb_pre,
    duracion_lomb_post
)


F_MIN_LOMB = (
    1.0
    /
    duracion_minima
)


print("\n========================================")
print("CONFIGURACIÓN DEL PERIODOGRAMA DE LOMB")
print("========================================")

print(
    f"Duración útil PRE: "
    f"{duracion_lomb_pre:.2f} s"
)

print(
    f"Duración útil POST: "
    f"{duracion_lomb_post:.2f} s"
)

print(
    f"Frecuencia mínima analizada: "
    f"{F_MIN_LOMB:.4f} Hz"
)

print(
    f"Frecuencia máxima analizada: "
    f"{F_MAX_LOMB:.2f} Hz"
)


# ============================================================
# GRILLA COMÚN DE FRECUENCIAS
# ============================================================

frecuencias_lomb = np.linspace(
    F_MIN_LOMB,
    F_MAX_LOMB,
    N_FRECUENCIAS_LOMB
)


# scipy.signal.lombscargle utiliza frecuencia angular [rad/s].

omega_lomb = (
    2
    *
    np.pi
    *
    frecuencias_lomb
)


# ============================================================
# FUNCIÓN PARA CALCULAR EL PERIODOGRAMA
# ============================================================

def calcular_lomb(
    tiempo,
    rr,
    omega
):
    """
    Calcula el periodograma de Lomb de los intervalos RR.

    Antes del cálculo se elimina:

    1) el valor medio;
    2) una posible tendencia lineal lenta.

    La eliminación de la tendencia resulta especialmente
    importante en el registro post-actividad, donde existe
    una recuperación progresiva de la frecuencia cardíaca.

    El periodograma se devuelve normalizado.
    """

    tiempo = np.asarray(
        tiempo,
        dtype=float
    )

    rr = np.asarray(
        rr,
        dtype=float
    )


    # --------------------------------------------------------
    # ELIMINAR TENDENCIA LINEAL
    # --------------------------------------------------------
    #
    # Se ajusta:
    #
    #       RR(t) = a*t + b
    #
    # y se resta esa tendencia.
    #
    # De esta forma el periodograma representa principalmente
    # las oscilaciones alrededor de la tendencia general.
    # --------------------------------------------------------

    coeficientes = np.polyfit(
        tiempo,
        rr,
        1
    )

    tendencia = np.polyval(
        coeficientes,
        tiempo
    )

    rr_sin_tendencia = (
        rr
        -
        tendencia
    )


    # --------------------------------------------------------
    # ELIMINAR POSIBLE MEDIA RESIDUAL
    # --------------------------------------------------------

    rr_sin_tendencia = (
        rr_sin_tendencia
        -
        np.mean(
            rr_sin_tendencia
        )
    )


    # --------------------------------------------------------
    # PERIODOGRAMA DE LOMB
    # --------------------------------------------------------

    potencia = signal.lombscargle(
        tiempo,
        rr_sin_tendencia,
        omega,
        precenter=False,
        normalize=True
    )


    return (
        potencia,
        rr_sin_tendencia,
        tendencia
    )


# ============================================================
# CALCULAR LOMB PRE
# ============================================================

(
    lomb_pre,
    rr_pre_sin_tendencia,
    tendencia_pre
) = calcular_lomb(
    tiempo_lomb_pre,
    rr_lomb_pre,
    omega_lomb
)


# ============================================================
# CALCULAR LOMB POST
# ============================================================

(
    lomb_post,
    rr_post_sin_tendencia,
    tendencia_post
) = calcular_lomb(
    tiempo_lomb_post,
    rr_lomb_post,
    omega_lomb
)


# ============================================================
# FRECUENCIA DOMINANTE
# ============================================================

indice_max_pre = np.argmax(
    lomb_pre
)

indice_max_post = np.argmax(
    lomb_post
)


frecuencia_dominante_pre = (
    frecuencias_lomb[
        indice_max_pre
    ]
)

frecuencia_dominante_post = (
    frecuencias_lomb[
        indice_max_post
    ]
)


potencia_max_pre = (
    lomb_pre[
        indice_max_pre
    ]
)

potencia_max_post = (
    lomb_post[
        indice_max_post
    ]
)


# ============================================================
# PERÍODO ASOCIADO A LA FRECUENCIA DOMINANTE
# ============================================================

periodo_dominante_pre = (
    1.0
    /
    frecuencia_dominante_pre
)

periodo_dominante_post = (
    1.0
    /
    frecuencia_dominante_post
)


print("\n========================================")
print("PERIODOGRAMA DE LOMB - PRE")
print("========================================")

print(
    f"Frecuencia dominante: "
    f"{frecuencia_dominante_pre:.4f} Hz"
)

print(
    f"Período correspondiente: "
    f"{periodo_dominante_pre:.2f} s"
)

print(
    f"Potencia normalizada máxima: "
    f"{potencia_max_pre:.4f}"
)


print("\n========================================")
print("PERIODOGRAMA DE LOMB - POST")
print("========================================")

print(
    f"Frecuencia dominante: "
    f"{frecuencia_dominante_post:.4f} Hz"
)

print(
    f"Período correspondiente: "
    f"{periodo_dominante_post:.2f} s"
)

print(
    f"Potencia normalizada máxima: "
    f"{potencia_max_post:.4f}"
)


# ============================================================
# POTENCIA DENTRO DE UNA BANDA
# ============================================================

def potencia_en_banda(
    frecuencias,
    potencia,
    f_min,
    f_max
):
    """
    Calcula el área bajo el periodograma dentro de
    una determinada banda de frecuencias.
    """

    mascara = (
        (frecuencias >= f_min)
        &
        (frecuencias < f_max)
    )


    if np.sum(
        mascara
    ) < 2:

        return np.nan


    return np.trapezoid(
        potencia[
            mascara
        ],
        frecuencias[
            mascara
        ]
    )


# ============================================================
# POTENCIA LF
# ============================================================

lf_pre = potencia_en_banda(
    frecuencias_lomb,
    lomb_pre,
    LF_MIN,
    LF_MAX
)


lf_post = potencia_en_banda(
    frecuencias_lomb,
    lomb_post,
    LF_MIN,
    LF_MAX
)


# ============================================================
# POTENCIA HF
# ============================================================

hf_pre = potencia_en_banda(
    frecuencias_lomb,
    lomb_pre,
    HF_MIN,
    HF_MAX
)


hf_post = potencia_en_banda(
    frecuencias_lomb,
    lomb_post,
    HF_MIN,
    HF_MAX
)


# ============================================================
# RELACIÓN LF/HF
# ============================================================

if hf_pre > 0:

    relacion_lf_hf_pre = (
        lf_pre
        /
        hf_pre
    )

else:

    relacion_lf_hf_pre = np.nan


if hf_post > 0:

    relacion_lf_hf_post = (
        lf_post
        /
        hf_post
    )

else:

    relacion_lf_hf_post = np.nan


print("\n========================================")
print("DISTRIBUCIÓN ESPECTRAL")
print("========================================")

print("\nPRE")

print(
    f"Potencia LF: "
    f"{lf_pre:.6f}"
)

print(
    f"Potencia HF: "
    f"{hf_pre:.6f}"
)

print(
    f"LF/HF: "
    f"{relacion_lf_hf_pre:.4f}"
)


print("\nPOST")

print(
    f"Potencia LF: "
    f"{lf_post:.6f}"
)

print(
    f"Potencia HF: "
    f"{hf_post:.6f}"
)

print(
    f"LF/HF: "
    f"{relacion_lf_hf_post:.4f}"
)


# ============================================================
# FUNCIÓN PARA GRAFICAR PERIODOGRAMA DE LOMB
# ============================================================

def graficar_lomb(
    frecuencias,
    potencia,
    titulo,
    nombre_archivo,
    frecuencia_dominante=None
):

    plt.figure(
        figsize=(12, 5)
    )


    plt.plot(
        frecuencias,
        potencia,
        linewidth=1.2,
        label="Periodograma de Lomb"
    )


    # --------------------------------------------------------
    # LÍMITES LF / HF
    # --------------------------------------------------------

    plt.axvline(
        LF_MIN,
        linestyle="--",
        linewidth=0.8,
        label="0.04 Hz"
    )

    plt.axvline(
        LF_MAX,
        linestyle="--",
        linewidth=0.8,
        label="0.15 Hz"
    )

    plt.axvline(
        HF_MAX,
        linestyle="--",
        linewidth=0.8,
        label="0.40 Hz"
    )


    # --------------------------------------------------------
    # FRECUENCIA DOMINANTE
    # --------------------------------------------------------

    if frecuencia_dominante is not None:

        plt.axvline(
            frecuencia_dominante,
            linestyle=":",
            linewidth=1.2,
            label=(
                f"Frecuencia dominante = "
                f"{frecuencia_dominante:.3f} Hz"
            )
        )


    plt.xlabel(
        "Frecuencia [Hz]"
    )

    plt.ylabel(
        "Potencia normalizada"
    )

    plt.title(
        titulo
    )

    plt.xlim(
        F_MIN_LOMB,
        F_MAX_LOMB
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()


    guardar_imagen(
        nombre_archivo
    )


    plt.show()

    plt.close()


# ============================================================
# PERIODOGRAMA PRE
# ============================================================

graficar_lomb(
    frecuencias_lomb,
    lomb_pre,
    "Periodograma de Lomb - ECG en reposo",
    "Lomb_Pre.png",
    frecuencia_dominante_pre
)


# ============================================================
# PERIODOGRAMA POST
# ============================================================

graficar_lomb(
    frecuencias_lomb,
    lomb_post,
    "Periodograma de Lomb - ECG post actividad física",
    "Lomb_Post.png",
    frecuencia_dominante_post
)


# ============================================================
# COMPARACIÓN PRE VS POST
# ============================================================

plt.figure(
    figsize=(12, 5)
)


plt.plot(
    frecuencias_lomb,
    lomb_pre,
    linewidth=1.2,
    label="Reposo"
)


plt.plot(
    frecuencias_lomb,
    lomb_post,
    linewidth=1.2,
    label="Post actividad física"
)


# Límites de las bandas

plt.axvline(
    LF_MIN,
    linestyle="--",
    linewidth=0.8
)

plt.axvline(
    LF_MAX,
    linestyle="--",
    linewidth=0.8
)

plt.axvline(
    HF_MAX,
    linestyle="--",
    linewidth=0.8
)


plt.xlabel(
    "Frecuencia [Hz]"
)

plt.ylabel(
    "Potencia normalizada"
)

plt.title(
    "Comparación de los periodogramas de Lomb"
)

plt.xlim(
    F_MIN_LOMB,
    F_MAX_LOMB
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()


guardar_imagen(
    "Lomb_Comparacion.png"
)


plt.show()

plt.close()


# ============================================================
# GUARDAR RESULTADOS EN CSV
# ============================================================

datos_lomb = pd.DataFrame(
    {
        "Frecuencia_Hz":
        frecuencias_lomb,

        "Lomb_Pre":
        lomb_pre,

        "Lomb_Post":
        lomb_post
    }
)


datos_lomb.to_csv(
    "periodograma_lomb.csv",
    index=False
)


# ============================================================
# RESUMEN NUMÉRICO
# ============================================================

resumen_lomb = pd.DataFrame(
    {
        "Estado": [
            "Reposo",
            "Post actividad"
        ],

        "Frecuencia_dominante_Hz": [
            frecuencia_dominante_pre,
            frecuencia_dominante_post
        ],

        "Periodo_dominante_s": [
            periodo_dominante_pre,
            periodo_dominante_post
        ],

        "Potencia_LF": [
            lf_pre,
            lf_post
        ],

        "Potencia_HF": [
            hf_pre,
            hf_post
        ],

        "Relacion_LF_HF": [
            relacion_lf_hf_pre,
            relacion_lf_hf_post
        ]
    }
)


resumen_lomb.to_csv(
    "resumen_lomb.csv",
    index=False
)


# ============================================================
# VERIFICACIÓN DE IMÁGENES DEL INCISO 7
# ============================================================

IMAGENES_LOMB = [

    "Lomb_Pre.png",

    "Lomb_Post.png",

    "Lomb_Comparacion.png"
]


print("\n========================================")
print("IMÁGENES DEL INCISO 7")
print("========================================")


for nombre_imagen in IMAGENES_LOMB:

    ruta = (
        CARPETA_IMAGENES
        /
        nombre_imagen
    )

    if ruta.exists():

        print(
            f"OK: {ruta}"
        )

    else:

        print(
            f"FALTA: {ruta}"
        )


print("\nArchivos generados:")

print(
    "periodograma_lomb.csv"
)

print(
    "resumen_lomb.csv"
)