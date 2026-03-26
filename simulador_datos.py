import pandas as pd 
import numpy as np
#creo ese simulador aparte los rangos de datos intentando acercarce lo posible a simular pacientes
#   
def generar_datos(numero_registros: int, datos_con_ruido: int, semilla: int) -> pd.DataFrame:
    np.random.seed(semilla)
    datos= pd.DataFrame({"Edad": np.random.randint(18, 90, size=numero_registros),
                        "Peso" : np.random.normal(loc=70, scale=15 , size=numero_registros).round(2),
                        "Altura" :np.random.normal(loc=1.70, scale=0.15, size=numero_registros).round(2),
                        "Presion_Arterial": np.random.normal(loc=125, scale=15, size=numero_registros).round(2),
                        "Glucosa": np.random.normal(loc=95, scale=15, size=numero_registros).round(2)
                        })
    indices_con_error=np.random.choice(datos.index, size=datos_con_ruido, replace=False)
    for indice in indices_con_error:
        columna=np.random.choice(["Edad","Peso","Altura","Presion_Arterial","Glucosa"])
        
        ingresar_nan=np.random.choice([True,False])
        if ingresar_nan:
            datos.loc[indice,columna] = np.nan 
        else:
            
            if columna == "Edad":
                datos.loc[indice, columna] = np.random.randint(-100 , 0)
            elif columna == "Peso":
                datos.loc[indice, columna] = np.random.randint(-40, -25)
            elif columna == "Altura":
                datos.loc[indice, columna] = np.random.randint(0,7)
            elif columna == "Presion_Arterial":
                datos.loc[indice, columna] = np.random.randint(-70, 500)
            elif columna == "Glucosa":
                datos.loc[indice, columna] = np.random.randint(-40, 1000)
    
    return datos 
#Aqui clasifica si el paciente es hospitalizado o retornando 0 si no y 1 si es hospitalizado
#esto pretende clasificar un paciente intentando acercarse lo que se hace en un hospital.    
def evaluar_hospitalizacion(paciente:pd.Series ) -> int:
    imc=paciente["Peso"] / (paciente["Altura"]**2)
    
    malas_condiciones=(paciente["Edad"] > 80 or
            imc > 40 or imc < 16 or
            paciente["Presion_Arterial"] > 180 or 
            paciente["Glucosa"] > 200 or paciente["Glucosa"] < 50 
    
            )
    if malas_condiciones:
        return 1
    else:
        return 0 
    
def registros_pacientes(num_registros: int, datos_con_ruido: int = 0, semilla: int = 42) -> pd.DataFrame:
    datos=generar_datos(num_registros,datos_con_ruido, semilla)
    datos["Hospitalizacion"] =datos.apply(evaluar_hospitalizacion, axis=1)
    datos["Edad"] = datos["Edad"].astype("Int64")
    return datos 


