import pandas as pd
import numpy as np
import simulador_datos
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score,roc_auc_score,roc_curve,precision_score,f1_score
import matplotlib.pyplot as plt
import seaborn as sns
semilla_random=np.random.randint(0,1000) 
#Extrae los datos de un simulador personalizado
#---Parametros---
#num_registros: La cantidad de registros que genera 
#datos_con_ruido: la cantidad de datos corrompidos dentro de la cantidad num_registros  
datos=simulador_datos.registros_pacientes(num_registros=1000, datos_con_ruido=50,)
len_inicial=len(datos)
mensaje="Presione ENTER para Continuar..."
divicion="="* len(mensaje)
print(divicion)
print("Datos generados Correctamente..")
print(datos.head(4))
input(mensaje)
print(divicion)

#Detecta duplicados y los elimina.
duplicados=datos.duplicated().sum()
print(f"Registros Duplicados: {duplicados}")
datos = datos.drop_duplicates()
print(f"Registros Concervados: ({len(datos)}/{len_inicial})")
input(mensaje)
print(divicion)

#Detecta Registros Con valores Nulos y los elimina.
nulos=datos.isnull().sum().sum()
print(f"Registros con valores Nulos: {nulos}")
datos = datos.dropna()
print(f"Registros Concervados: ({len(datos)}/{len_inicial})")
input(mensaje)
print(divicion)

#Detectar si hay numeros negativos 
#Deteccion de outlier y procede a eliminarlos 

columnas_numericas= datos.select_dtypes(include=["int64","float64"]).columns
indices_valor_negativo=set()
indices_con_outlier= set()

for columna in columnas_numericas:
    negativo_detec= set(datos.index[datos[columna] < 0])
    indices_valor_negativo.update(negativo_detec)
print(f"Registros con valores negativos: {len(indices_valor_negativo)}")
if indices_valor_negativo:
    print(datos.loc[list(indices_valor_negativo)].head(4))
else:
    print("No se han detectado valores negativos")
#Aqui elimino los datos negativos con los indices encontrados
datos=datos.drop(index=indices_valor_negativo)
print(f"Registros Concervados: ({len(datos)}/{len_inicial})")
input(mensaje)
print(divicion)
for columna in columnas_numericas:
    #como son datos generados y no me detecte muchos outlier cambio parametros a 40 vecinos de comparacion y el 2% de contaminados
    modelo= LocalOutlierFactor(n_neighbors=40, contamination=0.02)
    resultado=modelo.fit_predict(datos[[columna]])

    outlier_indice= set(datos.index[resultado == -1])
    indices_con_outlier.update(outlier_indice)
 
print(f"Registros con valores Outlier: {len(indices_con_outlier)}")    
if indices_con_outlier:
    print(datos.loc[list(indices_con_outlier)].head(4))
else:
     ("No se ha encontrado registro Outlier")
datos=datos.drop(index=indices_con_outlier)
print(f"Registros Concervados: ({len(datos)}/{len_inicial})")
input(mensaje)
print(divicion)
etiqueta_objetivo_conteo=datos["Hospitalizacion"].value_counts()
por_hospitalizados=(datos["Hospitalizacion"].sum() / len(datos)) * 100
print(etiqueta_objetivo_conteo)

print(f"El {por_hospitalizados:.2f}% de los registros destinados al entrenar corresponden a  pacientes Hospitalizados")
input(mensaje)
print(divicion)

#Divicion de datos de entrenamiento/prueba 
x=datos.drop(["Hospitalizacion"], axis=1).values
y=datos["Hospitalizacion"].values

escala=StandardScaler()
x=escala.fit_transform(x)

x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = train_test_split(x,y, test_size=0.3, random_state=42)

modelo=RandomForestClassifier()

modelo.fit(x_entrenamiento, y_entrenamiento)
print("Entrenando...")
prediccion = modelo.predict(x_prueba)
y_proba= modelo.predict_proba(x_prueba)[:,1]

print("----------Muestras----------")
#Semillas aleatorias para que no de las miasmas 4 muestras
muestra = pd.DataFrame({"Reales":y_prueba,"Prediccion":prediccion}).sample(4,random_state=semilla_random)
print(muestra)

# Evaluacion del modelo 
print("----------Evaluacion el modelo---------")
accuracy=accuracy_score(y_prueba, prediccion)
print(f"Accuracy(presicion): {accuracy:.3f}")

presicion=precision_score(y_prueba, prediccion)
print(f"Presicion: {presicion:.3f}")

recall=recall_score(y_prueba, prediccion)
print(f"Recall(Sensibilidad): {recall:.3f}")

f1=f1_score(y_prueba, prediccion)
print(f"F1_Score: {f1:.3f}")

roc_auc= roc_auc_score(y_prueba, y_proba)
print(f"ROC AUC: {roc_auc:.3}")

#Bloque de graficado
sns.set_theme(style="darkgrid")
falsos_pos, verdaderos_pos, umbral = roc_curve(y_prueba,y_proba) 

fig, (g1,g2) = plt.subplots(2,1, figsize=(8,6))
fig.suptitle("Evaluacion del Modelo.",fontsize=14)

g1.plot(falsos_pos, verdaderos_pos, label=f"Curva ROC /AUC:{roc_auc:.3f}", color="Orange") 
g1.plot([0.1],[0.1], color="navy", linestyle="--")
g1.set_xlim([0.0, 1.0])
g1.set_ylim([0.0, 1.05])

g1.set_xlabel(" Tasa de Falsos positivos(FPR)")
g1.set_ylabel("Tasa de Verdaderos Positivos(Recall)")
g1.set_title("Curva ROC")
g1.legend()
g1.grid(True, linestyle="--", alpha=0.5)

metricas_nom=["Acuracy","Presicion","Recall","F1-score"]
metricas_val=[accuracy,presicion,recall,f1]
colores=["blue","orange","green","purple"]

g2.bar(metricas_nom, metricas_val,color=colores)
g2.set_ylim(0,1)
g2.set_ylabel("Valor")
g2.set_title("Metricas de Evaluacion")
g2.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
