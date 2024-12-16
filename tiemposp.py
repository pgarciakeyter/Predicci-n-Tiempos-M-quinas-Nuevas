import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def convertir_a_excel(archivo):
    # Verificar si el archivo es un CSV o alguna otra fuente
    if archivo.name.endswith('.csv'):
        # Si el archivo es CSV, leerlo y guardarlo como Excel
        df = pd.read_csv(archivo)
        archivo_convertido = "archivo_convertido.xlsx"
        df.to_excel(archivo_convertido, index=False)
        return archivo_convertido
    elif archivo.name.endswith('.sql'):
        # Aquí podrías tener una lógica para ejecutar la consulta SQL y convertirla en un DataFrame
        # Por ejemplo, usando un conector de base de datos. Este es un caso más avanzado.
        # Para el propósito de este ejemplo, supongamos que el archivo SQL se convierte de alguna forma en DataFrame.
        # Simulamos la conversión a DataFrame:
        df = pd.read_sql("SELECT * FROM some_table", con)  # Esto depende de cómo te conectes a tu base de datos.
        archivo_convertido = "archivo_convertido.xlsx"
        df.to_excel(archivo_convertido, index=False)
        return archivo_convertido
    else:
        return archivo.name  # Si ya es Excel, devolver el nombre tal cual

def generar_diccionario_valores(df, columna):
    # Obtener valores únicos y ordenarlos para consistencia
    valores_unicos = sorted(map(str, df[columna].unique()))
    # Crear el diccionario con valores consecutivos
    valores_diccionario = {valor: indice + 1 for indice, valor in enumerate(valores_unicos)}
    return {columna: valores_diccionario}

codcent_nombre = {
    5405: "ALMACENEROS",
    5410: "CONFORMADOCHAPA",
    5415: "AISLAMIENTOCHAPA",
    5420: "CONFORMADOCOBRE/TUBERÍAS",
    5425: "HERRERÍAHIDRÁULICA",
    5430: "CUADROSELÉCTRICOS",
    5435: "ENSAMBLAJE",
    5445: "FRIGORISTASUNIONMAQUINA",
    5455: "FUGASVACIOYPRECARGA",
    5460: "CONEXIONESHIDRÁULICAS",
    5465: "AISLAMIENTOCOQUILLA",
    5470: "CONEXIONESELECTRICAS",
    5480: "PRUEBAS",
    5490: "CONTROLFINALYEMBALAJE"
}



# Funciones
def traducir_valor_streamlit(campo, valor, valores_traduccion):
    diccionario = valores_traduccion.get(campo)
    if diccionario and valor in diccionario:
        return diccionario[valor]
    else:
        st.error(f"El valor '{valor}' no es válido para el campo '{campo}'. Por favor, revisa los datos ingresados.")
        raise ValueError(f"El valor '{valor}' no es válido para el campo '{campo}'.")

def solicitar_datos_usuario_streamlit(entrada):
    if st.button("Procesar datos"):
        if len(entrada) < 10:
            st.error("Error: La cadena es demasiado corta para contener todos los valores.")
            return None

        try:
            # Extraer valores de la entrada
            familia = entrada[:-9]  # Todo excepto los últimos 8 caracteres
            chasis = entrada[-9]  # 4 caracteres antes del final
            funcionamiento = entrada[-5]  # Penúltimo carácter
            version = entrada[-4]  # Antepenúltimo carácter
            versionhidr = entrada[-3]  # Penúltimo carácter

            # Traducir valores
            familia_tradu = traducir_valor_streamlit("FAMILIA", familia, valores_traduccion)
              # Validar si chasis es un número
            try:
                chasis_tradu = float(chasis)  # Intentar convertir chasis a float
            except ValueError:
                st.error(f"Error: El valor de CHASIS '{chasis}' no es un número válido.")
                return None
            funcionamiento_tradu = traducir_valor_streamlit("FUNCIONAMIENTO", funcionamiento, valores_traduccion)
            version_tradu = traducir_valor_streamlit("VERSION", version, valores_traduccion)
            versionhidr_tradu = traducir_valor_streamlit("VERSIONHIDR", versionhidr, valores_traduccion)

            # Crear DataFrame
            datos = np.array([[familia_tradu, chasis_tradu, funcionamiento_tradu, version_tradu, versionhidr_tradu]], dtype=float)
            nueva_maquina = pd.DataFrame(
                datos,
                columns=["FAMILIA", "CHASIS", "FUNCIONAMIENTO", "VERSION", "VERSIONHIDR"]
            )

        
            return nueva_maquina

        except ValueError as e:
            st.error(f"Error: {e}")
            return None

def predecir_tiempos_streamlit(maquina, grouped, codcent_nombre):
    st.write("### Predicción de Tiempos por Máquina")
    total_predicciones = 0
    
    for codcent, grupo in grouped:
        st.write(f"Centro {codcent}: {len(grupo)} registros")
        nombre_codcent = codcent_nombre.get(codcent, "Desconocido")
        

        # Separar características y etiquetas
        x = grupo.drop(columns=['PROMEDIOHORAS', 'CODCENT'])
        y = grupo['PROMEDIOHORAS']

        # Dividir los datos en entrenamiento y prueba
        #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)

        # Entrenar modelo Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(x, y)

        # Predicción
        rf_predictions = rf_model.predict(maquina)

        st.write(f"Predicciones para el grupo {int(codcent)} - {nombre_codcent}: {rf_predictions} horas")
        total_predicciones += sum(rf_predictions)  # Sumar las predicciones del grupo actual

        st.write("---")
        
        

    # Mostrar la suma total de las predicciones
    st.write(f"Total de horas empleadas en la máquina: {total_predicciones} horas")

# Streamlit App
st.title("Predicción de Tiempos para Nuevas Máquinas")

# Cargar archivo
uploaded_file = st.file_uploader("Sube el archivo Excel con los datos", type=["xlsx"])

if uploaded_file is not None:
    basededatos = pd.read_excel(uploaded_file)
    #Si el archivo es una consulta SQL o CSV, convertirlo a Excel
    archivo_final = convertir_a_excel(uploaded_file)
    
    # Leer el archivo convertido (si fue necesario)
    basededatos = pd.read_excel(archivo_final)
    st.write("Vista previa de los datos:")
    st.write(basededatos.head())

    # Preprocesamiento
    basededatos = basededatos.drop(columns=["CENTRO"])
    basededatos = basededatos.drop(columns=["MAXIMOHORAS"])
    basededatos = basededatos.drop(columns=["MINIMOHORAS"])
    basededatos = basededatos.drop(columns=["NUMREGISTROS"])
    basededatos = basededatos.drop(columns=["SUMAHORAS"])
    group_sizes = basededatos.groupby('CODCENT').size()
    basededatos = basededatos.loc[basededatos["DESVTIPICAHORAS"] < 40, :]
    basededatos = basededatos.drop(columns=["DESVTIPICAHORAS"])
    valid_groups = group_sizes[group_sizes >= 10].index
    basededatos = basededatos[basededatos['CODCENT'].isin(valid_groups)]
    basededatos = basededatos[basededatos['CODCENT'] != 5440]
    basededatos = basededatos.dropna(how='any')
    basededatos["CHASIS"] = pd.to_numeric(basededatos["CHASIS"], errors="coerce")
    
    # Generar el diccionario
    valores_familia = generar_diccionario_valores(basededatos, "FAMILIA")
    valores_funcionamiento = generar_diccionario_valores(basededatos, "FUNCIONAMIENTO")
    valores_version = generar_diccionario_valores(basededatos, "VERSION")
    valores_versionhidr = generar_diccionario_valores(basededatos, "VERSIONHIDR")
    basededatos.replace(valores_familia, inplace = True)
    basededatos.replace(valores_funcionamiento, inplace = True)
    basededatos.replace(valores_version, inplace = True)
    basededatos.replace(valores_versionhidr, inplace = True)

    #---------------------
    valores_traduccion = {**valores_familia, **valores_funcionamiento, **valores_version, **valores_versionhidr}
   
    # Agrupar datos por CODCENT
    grouped = basededatos.groupby('CODCENT')
    # st.write("Tipos de datos de las columnas:", grouped.dtypes)

    st.write("Introduce el nombre de la máquina")

    # Entrada del usuario
    entrada = st.text_input("Introduce la cadena de datos:", "")
    nueva_maquina = solicitar_datos_usuario_streamlit(entrada)

    if nueva_maquina is not None:
        predecir_tiempos_streamlit(nueva_maquina, grouped, codcent_nombre)
