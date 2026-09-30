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
import matplotlib.pyplot as plt
from scipy import signal


# ============================================================
# CARPETA DE SALIDA DE IMÁGENES
# ============================================================

CARPETA_IMAGENES = "Imagenes"

# Si la carpeta no existe, se crea automáticamente
os.makedirs(CARPETA_IMAGENES, exist_ok=True)


# ============================================================
# FUNCIÓN PAN-TOMPKINS
# ============================================================

def pan_tompkins(ecg, fs, mascara_saturacion=None):

    ecg = np.asarray(ecg, dtype=float).flatten()


    # --------------------------------------------------------
    # 1. FILTRO PASA-BANDA
    # --------------------------------------------------------
    # El complejo QRS concentra gran parte de su contenido
    # energético en este rango de frecuencias.
    #
    # Aunque el ECG ya fue previamente acondicionado,
    # este filtrado forma parte de la etapa de detección
    # del algoritmo Pan-Tompkins.

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
    # Resalta las pendientes rápidas presentes en el QRS.

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
    # Hace positivos todos los valores y aumenta el peso
    # relativo de las pendientes de mayor amplitud.

    ecg_cuadrado = ecg_derivada ** 2


    # --------------------------------------------------------
    # 4. INTEGRACIÓN MEDIANTE VENTANA MÓVIL
    # --------------------------------------------------------
    # Se utiliza una ventana aproximada de 150 ms.

    ventana_seg = 0.150

    N = max(
        1,
        int(ventana_seg * fs)
    )

    ventana = np.ones(N) / N

    ecg_integrado = signal.convolve(
        ecg_cuadrado,
        ventana,
        mode="same"
    )


    # --------------------------------------------------------
    # 5. DETECCIÓN DE PICOS CANDIDATOS
    # --------------------------------------------------------
    # Distancia mínima de 250 ms entre complejos.
    #
    # Esto equivale aproximadamente a una FC máxima de
    # 240 lat/min.

    distancia_minima = int(
        0.25 * fs
    )

    picos_candidatos, propiedades = signal.find_peaks(
        ecg_integrado,
        distance=distancia_minima
    )


    # --------------------------------------------------------
    # CASO SIN DETECCIONES
    # --------------------------------------------------------

    if len(picos_candidatos) == 0:

        return {
            "bandpass": ecg_bp,
            "derivada": ecg_derivada,
            "cuadrado": ecg_cuadrado,
            "integrado": ecg_integrado,
            "picos_integrados": np.array([], dtype=int),
            "picos_r": np.array([], dtype=int),
            "rr": np.array([]),
            "fc": np.array([]),
            "umbral": 0
        }


    # --------------------------------------------------------
    # 6. UMBRAL ADAPTATIVO
    # --------------------------------------------------------
    # Se estima un nivel representativo del ruido y otro
    # asociado a los picos de señal.

    amplitudes = ecg_integrado[
        picos_candidatos
    ]

    nivel_ruido = np.percentile(
        amplitudes,
        25
    )

    nivel_senal = np.percentile(
        amplitudes,
        90
    )

    umbral = nivel_ruido + 0.25 * (
        nivel_senal - nivel_ruido
    )

    picos_integrados = picos_candidatos[
        amplitudes >= umbral
    ]


    # --------------------------------------------------------
    # 7. LOCALIZACIÓN EXACTA DEL PICO R
    # --------------------------------------------------------
    # El máximo de la señal integrada no necesariamente
    # coincide temporalmente con el pico R.
    #
    # Se busca entonces el máximo absoluto del ECG filtrado
    # en una ventana alrededor de cada detección.

    ventana_busqueda = int(
        0.150 * fs
    )

    picos_r = []

    for pico in picos_integrados:

        inicio = max(
            0,
            pico - ventana_busqueda
        )

        fin = min(
            len(ecg_bp),
            pico + ventana_busqueda
        )

        segmento = ecg_bp[
            inicio:fin
        ]

        if len(segmento) == 0:
            continue


        # Se utiliza valor absoluto porque el QRS puede
        # encontrarse orientado positiva o negativamente.

        indice_local = np.argmax(
            np.abs(segmento)
        )

        pico_r = inicio + indice_local


        # ----------------------------------------------------
        # EXCLUSIÓN DE ZONAS SATURADAS
        # ----------------------------------------------------

        if mascara_saturacion is not None:

            sat_inicio = max(
                0,
                pico_r - int(0.10 * fs)
            )

            sat_fin = min(
                len(mascara_saturacion),
                pico_r + int(0.10 * fs)
            )

            if np.any(
                mascara_saturacion[
                    sat_inicio:sat_fin
                ]
            ):
                continue


        picos_r.append(
            pico_r
        )


    picos_r = np.array(
        picos_r,
        dtype=int
    )


    # --------------------------------------------------------
    # 8. ELIMINACIÓN DE DETECCIONES DUPLICADAS
    # --------------------------------------------------------

    if len(picos_r) > 1:

        picos_r = np.sort(
            picos_r
        )

        picos_limpios = [
            picos_r[0]
        ]

        for pico in picos_r[1:]:

            anterior = picos_limpios[-1]

            if (
                pico - anterior
                >= distancia_minima
            ):

                picos_limpios.append(
                    pico
                )

            else:

                # Si aparecen dos detecciones demasiado
                # próximas se conserva la de mayor amplitud.

                if (
                    abs(ecg_bp[pico])
                    >
                    abs(ecg_bp[anterior])
                ):

                    picos_limpios[-1] = pico


        picos_r = np.array(
            picos_limpios,
            dtype=int
        )


    # --------------------------------------------------------
    # 9. INTERVALOS RR
    # --------------------------------------------------------

    rr = np.diff(
        picos_r
    ) / fs


    # --------------------------------------------------------
    # 10. FRECUENCIA CARDÍACA INSTANTÁNEA
    # --------------------------------------------------------

    fc = 60.0 / rr


    # --------------------------------------------------------
    # DEVOLVER RESULTADOS
    # --------------------------------------------------------

    return {

        "bandpass": ecg_bp,

        "derivada": ecg_derivada,

        "cuadrado": ecg_cuadrado,

        "integrado": ecg_integrado,

        "picos_integrados": picos_integrados,

        "picos_r": picos_r,

        "rr": rr,

        "fc": fc,

        "umbral": umbral
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

picos_r_pre = resultado_pre[
    "picos_r"
]

picos_r_post = resultado_post[
    "picos_r"
]

rr_pre = resultado_pre[
    "rr"
]

rr_post = resultado_post[
    "rr"
]

fc_pre = resultado_pre[
    "fc"
]

fc_post = resultado_post[
    "fc"
]


# ============================================================
# MOSTRAR RESULTADOS NUMÉRICOS
# ============================================================

print("\n")
print("==============================")
print("ECG PRE-ACTIVIDAD")
print("==============================")

print(
    "Cantidad de complejos QRS detectados:",
    len(picos_r_pre)
)

if len(rr_pre) > 0:

    print(
        f"RR medio: "
        f"{np.mean(rr_pre):.3f} s"
    )

    print(
        f"FC media: "
        f"{np.mean(fc_pre):.2f} lat/min"
    )

    print(
        f"FC mínima: "
        f"{np.min(fc_pre):.2f} lat/min"
    )

    print(
        f"FC máxima: "
        f"{np.max(fc_pre):.2f} lat/min"
    )


print("\n")
print("==============================")
print("ECG POST-ACTIVIDAD")
print("==============================")

print(
    "Cantidad de complejos QRS detectados:",
    len(picos_r_post)
)

if len(rr_post) > 0:

    print(
        f"RR medio: "
        f"{np.mean(rr_post):.3f} s"
    )

    print(
        f"FC media: "
        f"{np.mean(fc_post):.2f} lat/min"
    )

    print(
        f"FC mínima: "
        f"{np.min(fc_post):.2f} lat/min"
    )

    print(
        f"FC máxima: "
        f"{np.max(fc_post):.2f} lat/min"
    )


# ============================================================
# FUNCIÓN PARA GRAFICAR LAS ETAPAS DE PAN-TOMPKINS
# ============================================================

def graficar_pan_tompkins(
    ecg,
    resultado,
    fs,
    titulo,
    nombre_archivo
):

    t = np.arange(
        len(ecg)
    ) / fs


    fig, axes = plt.subplots(
        5,
        1,
        figsize=(14, 11),
        sharex=True
    )


    # --------------------------------------------------------
    # ECG PROCESADO
    # --------------------------------------------------------

    axes[0].plot(
        t,
        ecg,
        linewidth=0.8
    )

    axes[0].set_ylabel(
        "Amplitud"
    )

    axes[0].set_title(
        f"{titulo} - ECG procesado"
    )

    axes[0].grid(
        True
    )


    # --------------------------------------------------------
    # FILTRO PASA-BANDA
    # --------------------------------------------------------

    axes[1].plot(
        t,
        resultado["bandpass"],
        linewidth=0.8
    )

    axes[1].set_ylabel(
        "Amplitud"
    )

    axes[1].set_title(
        "Filtrado pasa-banda 5-15 Hz"
    )

    axes[1].grid(
        True
    )


    # --------------------------------------------------------
    # DERIVADA
    # --------------------------------------------------------

    axes[2].plot(
        t,
        resultado["derivada"],
        linewidth=0.8
    )

    axes[2].set_ylabel(
        "Amplitud"
    )

    axes[2].set_title(
        "Filtro derivativo"
    )

    axes[2].grid(
        True
    )


    # --------------------------------------------------------
    # CUADRADO
    # --------------------------------------------------------

    axes[3].plot(
        t,
        resultado["cuadrado"],
        linewidth=0.8
    )

    axes[3].set_ylabel(
        "Amplitud"
    )

    axes[3].set_title(
        "Elevación al cuadrado"
    )

    axes[3].grid(
        True
    )


    # --------------------------------------------------------
    # INTEGRACIÓN POR VENTANA MÓVIL
    # --------------------------------------------------------

    axes[4].plot(
        t,
        resultado["integrado"],
        linewidth=0.8,
        label="Señal integrada"
    )

    axes[4].axhline(
        resultado["umbral"],
        linestyle="--",
        label="Umbral de detección"
    )


    if len(
        resultado["picos_integrados"]
    ) > 0:

        axes[4].scatter(

            resultado[
                "picos_integrados"
            ] / fs,

            resultado[
                "integrado"
            ][
                resultado[
                    "picos_integrados"
                ]
            ],

            marker="x",

            s=45,

            label="QRS detectados"
        )


    axes[4].set_ylabel(
        "Amplitud"
    )

    axes[4].set_xlabel(
        "Tiempo [s]"
    )

    axes[4].set_title(
        "Integración por ventana móvil"
    )

    axes[4].grid(
        True
    )

    axes[4].legend()


    plt.tight_layout()


    # --------------------------------------------------------
    # GUARDAR IMAGEN
    # --------------------------------------------------------

    ruta = os.path.join(
        CARPETA_IMAGENES,
        nombre_archivo
    )

    plt.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight"
    )


    print(
        f"Imagen guardada en: {ruta}"
    )


    plt.show()

    plt.close()


# ============================================================
# GRAFICAR Y GUARDAR ETAPAS DE PAN-TOMPKINS
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

    t = np.arange(
        len(ecg)
    ) / fs


    plt.figure(
        figsize=(14, 5)
    )


    plt.plot(
        t,
        ecg,
        linewidth=0.8,
        label="ECG procesado"
    )


    if len(picos_r) > 0:

        plt.scatter(

            picos_r / fs,

            ecg[
                picos_r
            ],

            marker="x",

            s=60,

            label="Picos R",

            zorder=3
        )


    plt.xlabel(
        "Tiempo [s]"
    )

    plt.ylabel(
        "Amplitud"
    )

    plt.title(
        titulo
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()


    # --------------------------------------------------------
    # GUARDAR IMAGEN
    # --------------------------------------------------------

    ruta = os.path.join(
        CARPETA_IMAGENES,
        nombre_archivo
    )

    plt.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight"
    )


    print(
        f"Imagen guardada en: {ruta}"
    )


    plt.show()

    plt.close()


# ============================================================
# GRÁFICOS FINALES CON PICOS R
# ============================================================

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
            f"No hay suficientes detecciones "
            f"para generar {nombre_archivo}"
        )

        return


    # Cada valor de FC corresponde al intervalo entre
    # dos picos R consecutivos.

    tiempo_fc = (
        picos_r[1:] / fs
    )


    plt.figure(
        figsize=(12, 4)
    )


    plt.plot(
        tiempo_fc,
        fc,
        marker="o",
        linewidth=1
    )


    plt.xlabel(
        "Tiempo [s]"
    )

    plt.ylabel(
        "Frecuencia cardíaca [lat/min]"
    )

    plt.title(
        titulo
    )

    plt.grid(
        True
    )

    plt.tight_layout()


    # --------------------------------------------------------
    # GUARDAR IMAGEN
    # --------------------------------------------------------

    ruta = os.path.join(
        CARPETA_IMAGENES,
        nombre_archivo
    )

    plt.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight"
    )


    print(
        f"Imagen guardada en: {ruta}"
    )


    plt.show()

    plt.close()


# ============================================================
# FRECUENCIA CARDÍACA PRE-ACTIVIDAD
# ============================================================

graficar_frecuencia_cardiaca(
    picos_r_pre,
    fc_pre,
    FS,
    "Frecuencia cardíaca instantánea - Pre-actividad",
    "FC_Pre.png"
)


# ============================================================
# FRECUENCIA CARDÍACA POST-ACTIVIDAD
# ============================================================

graficar_frecuencia_cardiaca(
    picos_r_post,
    fc_post,
    FS,
    "Frecuencia cardíaca instantánea - Post-actividad",
    "FC_Post.png"
)