import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Mostrar la imagen al inicio
url_imagen = "https://github.com/pgarciakeyter/Predicci-n-Tiempos-M-quinas-Nuevas/blob/main/assets/keyter%20logo.png?raw=true"

col1, col2, col3 = st.columns([1, 2, 1])  # Columnas para centrar la imagen

with col2:  # Columna del medio
    st.image(url_imagen, width=500) 
    
# Pie con identificación
st.markdown("---")
#st.markdown("**Creado por [Paula García Chacón](https://www.tuwebsite.com)**") 
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
# Funciones
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
    mensaje_mostrado = False
    
    for codcent, grupo in grouped:
        nombre_codcent = codcent_nombre.get(codcent, "Desconocido")

        # Separar características y etiquetas
        x = grupo.drop(columns=['PROMEDIOHORAS', 'CODCENT', 'NUMREGISTROS'])
        y = grupo['PROMEDIOHORAS']
        z = grupo['NUMREGISTROS']

        # Verificar si los datos de la máquina existen en los registros
        existe = x.isin(maquina.to_dict(orient="list")).all(axis=1)
        if existe.any():  # Si existe un valor exacto en los registros
            if not mensaje_mostrado:                
                st.write(f"**El equipo {entrada} se ha fabricado anteriormente**")
                mensaje_mostrado=True
            valores_reales = y[existe].values
            num_registros = z[existe].values
            st.write(f"Nº de horas imputadas del centro  **{int(codcent)} - {nombre_codcent}** en este equipo: **{valores_reales[0]} horas**")
            #st.write(f"Este centro ha registrado {len(grupo)} datos")
            if num_registros == 1 :
                st.write(f"Este equipo ha pasado por este centro {num_registros[0]} vez")
            else:
                st.write(f"Este equipo ha pasado por este centro {num_registros[0]} veces")                
            total_predicciones += sum(valores_reales)
        else:
            # Entrenar modelo Random Forest si no se encuentran datos exactos
            if not mensaje_mostrado:                
                st.write(f"**El equipo {entrada} nunca se ha fabricado. Se mostrarán predicciones de tiempos:**")
                mensaje_mostrado=True
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(x, y)

            # Predicción
            rf_predictions = rf_model.predict(maquina)
            rf_predictions_redondeados = list(map(lambda x: round(x, 2), rf_predictions))
            st.write(f"Predicción de horas imputadas del centro **{int(codcent)} - {nombre_codcent}** en este equipo: **{rf_predictions_redondeados[0]} horas**")
            #st.write(f"El centro {codcent} ha registrado {len(grupo)} datos")
            st.write(f"Este equipo ha pasado por este centro 0 veces")   
            total_predicciones += sum(rf_predictions)

        st.write("---")

    # Mostrar la suma total de las predicciones o valores reales
    st.write(f"Total de horas empleadas en la máquina: **{round(total_predicciones, 2)} horas**")   

# Streamlit App
st.title("Predicción de Tiempos para Nuevas Máquinas")

# Cargar archivo
uploaded_file = st.file_uploader("Sube el archivo Excel con los datos", type=["xlsx"])

if uploaded_file is not None:
    basededatos = pd.read_excel(uploaded_file, sheet_name=1)
    st.write("Vista previa de los datos:")
    st.write(basededatos.head())

    # Preprocesamiento
    basededatos = basededatos.drop(columns=["CENTRO"])
    basededatos = basededatos.drop(columns=["MAXIMOHORAS"])
    basededatos = basededatos.drop(columns=["MINIMOHORAS"])
    basededatos = basededatos.drop(columns=["SUMAHORAS"])
    group_sizes = basededatos.groupby('CODCENT').size()
    basededatos = basededatos.loc[basededatos["DESVTIPICAHORAS"] < 40, :]
    basededatos = basededatos.drop(columns=["DESVTIPICAHORAS"])
    valid_groups = group_sizes[group_sizes >= 10].index
    basededatos = basededatos[basededatos['CODCENT'].isin(valid_groups)]
    basededatos = basededatos[basededatos['CODCENT'] != 5440]
    basededatos = basededatos.dropna(how='any')
    basededatos["CHASIS"] = pd.to_numeric(basededatos["CHASIS"], errors="coerce")
    basededatos["NUMREGISTROS"] = pd.to_numeric(basededatos["NUMREGISTROS"], errors="coerce")
    
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

    
    # Entrada del usuario
    entrada = st.text_input("Introduce el nombre de la máquina:", "")
    nueva_maquina = solicitar_datos_usuario_streamlit(entrada)

    if nueva_maquina is not None:
        predecir_tiempos_streamlit(nueva_maquina, grouped, codcent_nombre)
