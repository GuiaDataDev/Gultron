# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 13:48:30 2022

@author: José Eduardo Mendoza Portillo DataScientistSr
Actualización del codigo a funciones para optimizar las lineas de codigo,
resumen de los resultados
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
#import cv2
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
#from IPython.display import display
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import mysql.connector
from datetime import datetime

"Query to import the data set"

def run_query(query='', DB_HOST = 'mysqlhistorico.ct6m0mh6y7qj.us-east-2.rds.amazonaws.com', DB_USER = 'admin', DB_PASS = 'AfV!PtA8', DB_NAME = 'avi2'): 
    datos = datos = [DB_HOST, DB_USER, DB_PASS, DB_NAME] 
    conn = mysql.connector.connect( host=datos[0],
                                     user=datos[1],
                                     passwd=datos[2],
                                     database=datos[3]
                                     ) # Conectar a la base de datos 
    cursor = conn.cursor()         # Crear un cursor 
    cursor.execute(query)          # Ejecutar una consulta 

    if query.upper().startswith('SELECT'): 
        data = cursor.fetchall()   # Traer los resultados de un select 
    else: 
        conn.commit()              # Hacer efectiva la escritura de datos 
        data = None 
    
    cursor.close()                 # Cerrar el cursor 
    conn.close()  
    return pd.DataFrame(data)


consulta="""select operacion, inventario, cast(fecha_negocio as float) as 'fecha_negocio',
tipo_cliente_neg, presupuesto, intencion_compra, precio_cierre, cast(fecha_cierre as float) as 'fecha_cierre',
lat, lon, gender, auto_canal, prospeccion_canal, punto_ventas_canal, face_canal, face_medio, prospeccion_medio, auto_medio
from data_mart.vw_ventas
where fecha_negocio between '2022-01-01' and '2022-09-01';"""

consulta1="""select idhubspot, operacion, 4 as 'etapa_operacion',inventario, cast(fecha_negocio as float) as 'fecha_negocio',lat, lon,
gender, auto_canal, prospeccion_canal, punto_ventas_canal, face_canal, face_medio, prospeccion_medio, auto_medio
from data_mart.vw_ventas
where fecha_negocio between '2022-01-01' and '2022-07-01'
union all
select idhubspot, operacion, etapa_operacion, inventario, cast(fecha_negocio as float) as 'fecha_negocio', lat, lon,
gender, auto_canal, prospeccion_canal, punto_ventas_canal, face_canal, face_medio, prospeccion_medio, auto_medio
from data_mart.vw_leads
where fecha_negocio between '2022-01-01' and '2022-07-01';"""

consulta2="""select operacion, inventario, cast(fecha_negocio as float) as fecha_negocio, tipo_cliente_neg, cast(presupuesto as decimal(18,2)) as presupuesto, intencion_compra, precio_cierre, cast(fecha_cierre as float) as fecha_cierre,
 cast(lat as decimal(18,2)) as lat, cast(lon as decimal(18,2)) as lon,gender, auto_canal, prospeccion_canal, punto_ventas_canal, face_canal, face_medio, prospeccion_medio, auto_medio
from data_mart.vw_ventas
where fecha_negocio between '2022-01-01' and '2022-06-30';"""

inicio = datetime.utcnow()
print("Inicio de ejecucion: {0}".format(inicio))

clientes_df1 = run_query(query=consulta2, DB_HOST ='dbguia.crjnuawidyj0.us-east-2.rds.amazonaws.com', DB_USER='dataSciencie', DB_PASS='N99^p3R2p8', DB_NAME='data_mart')
names = ['operacion', 'inventario', 'fecha_negocio', 'tipo_cliente_neg', 'presupuesto', 'intencion_compra',\
         'precio_cierre', 'fecha_cierre', 'lat', 'lon','gender', 'auto_canal', 'prospeccion_canal','punto_ventas_canal',\
             'face_canal', 'face_medio', 'prospeccion_medio', 'auto_medio']


    
clientes_df=clientes_df1
clientes_df.columns = names

clientes_df = pd.read_csv("dostercios.csv")
clientes_df = clientes_df.iloc[:,1:len(clientes_df.columns)]

################################################
## Global variables
n_col = len(clientes_df.columns)
max_clusters = 15
level_encoder = 10
batch_size = 128
epochs = 500
random_state = 456
n_components = 3

colors = {"color": ["blue","gold","green","darkred","red","magenta","grey","purple","orange","pink","brown"]}
colors = pd.DataFrame(colors)

def barplot_visualization(x):
  fig = plt.Figure(figsize = (12, 6))
  fig = px.bar(x = clientes_df[x].value_counts().index, y = clientes_df[x].value_counts(), color = clientes_df[x].value_counts().index, height = 600)
  fig.show()

"""
for i in range(1,6):
    barplot_visualization(i)
"""

def plot_similarity(labels, features, rotation):
  corr = features.corr()# np.inner(clientes_df, clientes_df)
  sns.set(font_scale=1.2)
  g = sns.heatmap(
      corr,
      xticklabels=labels,
      yticklabels=labels,
      vmin=0,
      vmax=1,
      cmap="YlGnBu") # YLGnBu, YlOrRd
  g.set_xticklabels(labels, rotation=rotation)
  g.set_title("Features Correlations")
  plt.show()
  print(corr)

# Feature Scaling
scaler = StandardScaler()
clientes_df_scaled = scaler.fit_transform(clientes_df)

def kmeans_plot(dataset):
    range_values = range(1, max_clusters+1)
    scores = []
    for i in range_values:
      kmeans = KMeans(n_clusters = i)
      kmeans.fit(dataset)
      scores.append(kmeans.inertia_) # la inercia es la suma de los cuadrados de las distancias de las observaciones al centro del cluster más cercano
    
    plt.plot(range_values, scores, 'bx-')
    plt.title('Encontrar el número correcto de clusters')
    plt.xlabel('Nº Clusters')
    plt.ylabel('WCSS') 
    plt.show()

# Reducir los datos originales a 3 dimensiones usando PCA para visualizar los clústeres
def pca(dataset,n_components):
    columns = []
    for i in range(n_components):
        comp = 'pca'+str(i+1)
        columns.append(comp)
    pca = PCA(n_components = n_components)
    principal_comp = pca.fit_transform(dataset)
    pca_df = pd.DataFrame(data = principal_comp, columns = columns)
    return pca_df,pca

def kmeans_fit(n_cluster,dataset):
    kmeans = KMeans(n_cluster, random_state = random_state )
    kmeans.fit(dataset)
    labels = kmeans.labels_
    y_kmeans = kmeans.fit_predict(clientes_df_scaled)
    pca_df, pca_model = pca(clientes_df_scaled,n_components)
    pca_df = pd.concat([pca_df, pd.DataFrame({'cluster':labels})], axis = 1)
    cluster_centers = pd.DataFrame(data = kmeans.cluster_centers_, columns = [clientes_df.columns])
    cluster_centers = scaler.inverse_transform(cluster_centers)
    cluster_centers = pd.DataFrame(data = cluster_centers, columns = [clientes_df.columns])
    clientes_df_cluster = pd.concat([clientes_df, pd.DataFrame({'cluster':labels})], axis = 1)
    return pca_df,pca_model,n_cluster,kmeans,labels,cluster_centers,clientes_df_cluster,y_kmeans


def plot_hist(kmeans_dict):
    for i in kmeans_dict['clientes_df'].columns:
      plt.figure(figsize = (30, kmeans_dict['n_clusters']))
      for j in range(kmeans_dict['n_clusters']):
        plt.subplot(1, kmeans_dict['n_clusters'], j+1)
        cluster = kmeans_dict['clientes_df_cluster'][clientes_df_cluster['cluster'] == j]
        cluster[i].hist()
        plt.title('{}    \nCluster - {} '.format(i,j))
      plt.show()

    # Visualizar los clústeres con 3D-Scatterplot
def plot_pca_3d(pca_df):
    fig = px.scatter_3d(pca_df, x = 'pca1', y = 'pca2', z = 'pca3', 
                  color = 'cluster', symbol = 'cluster', size_max = 18, opacity = 0.7)
    fig.update_layout(margin = dict(l = 0, r = 0, b = 0, t = 0))
    fig.show()
  
    # Visualizar los clústeres con 2D-Scatterplot    
def plot_pca_2d(pca_df):
    dim = pca_df["cluster"].nunique()
    palette = list(colors.iloc[0:dim,0])
    ax = sns.scatterplot(x = "pca1", y = "pca2", hue = "cluster", data = pca_df, palette = palette)
    plt.show()

def summary_cluster(data_cluster):
    summary = pd.DataFrame()
    for i in range(0,data_cluster["cluster"].nunique()):
        resumen = data_cluster[data_cluster["cluster"] == i].describe()
        summary = pd.concat([summary, resumen], axis = 0)
    return summary

def cluster_percentage(data_cluster):
    summary_1 = {}
    for cluster  in data_cluster["cluster"].unique():
        summary_2 = {}
        total = data_cluster[data_cluster["cluster"] == cluster].count()[0]
        for col in data_cluster.columns:
             porcentaje = data_cluster[data_cluster["cluster"] == cluster].groupby([col]).count().iloc[:,0]/total*100
             summary_2[col] = porcentaje
        summary_1["cluster_"+str(cluster)] = summary_2  
    return summary_1  
    
# PCA for original data scaled
n_cluster = 6
# Assingn the k-means model and create the dataframe adding the clusters
pca_df,pca_model,n_cluster,kmeans,labels,cluster_centers,clientes_df_cluster,y_kmeans = kmeans_fit(n_cluster,pd.DataFrame(clientes_df_scaled) )
summary = cluster_percentage(clientes_df_cluster) 
summary["cluster_dist"] = summary_cluster(clientes_df_cluster)
# Create a dictionaty to add the models before and after the encoder
kmeans_dict ={} 
kmeans_dict['kmeans_std']= {"pca_df":pca_df,"pca_model":pca_model,"n_clusters":n_cluster, "clientes_df":clientes_df,"clientes_df_scaled":clientes_df_scaled,"kmeans": kmeans,"labels": labels,"cluster_centers":cluster_centers,"clientes_df_cluster":clientes_df_cluster,"y_kmeans":y_kmeans,"summary":summary}

# plot similarity to check correlations
plot_similarity(clientes_df.columns, pd.DataFrame(clientes_df_scaled), 90)

# plot histograms for the std model
kmeans_plot(clientes_df_scaled)
plot_hist(kmeans_dict['kmeans_std'])
plot_pca_2d(kmeans_dict['kmeans_std']["pca_df"][["pca1","pca2","cluster"]])


def nn_encoder(dataset,level_encoder,batch_size,epochs):
    input_df = Input(shape = (n_col,))
    x = Dense(50, activation = 'relu')(input_df)
    x = Dense(500, activation = 'relu', kernel_initializer = 'glorot_uniform')(x)
    x = Dense(500, activation = 'relu', kernel_initializer = 'glorot_uniform')(x)
    x = Dense(2000, activation = 'relu', kernel_initializer = 'glorot_uniform')(x)
    encoded = Dense(level_encoder, activation = 'relu', kernel_initializer = 'glorot_uniform')(x)
    x = Dense(2000, activation = 'relu', kernel_initializer = 'glorot_uniform')(encoded)
    x = Dense(500, activation = 'relu', kernel_initializer = 'glorot_uniform')(x)
    decoded = Dense(n_col, kernel_initializer = 'glorot_uniform')(x)
    
    # autoencoder
    autoencoder = Model(input_df, decoded)
    
    # encoder - utilizado para reducir la dimensión
    encoder = Model(input_df, encoded)
    
    autoencoder.compile(optimizer = 'adam', loss='mean_squared_error')
    
    autoencoder.fit(dataset, dataset, batch_size = batch_size, epochs = epochs, verbose = 3)
    
    autoencoder.save_weights('autoencoder_1.h5')
    
    pred = encoder.predict(clientes_df_scaled)
    return pred,encoder,autoencoder

clientes_encoder,encoder,autoencoder = nn_encoder(clientes_df_scaled,level_encoder,batch_size,epochs)

n_cluster = 2 #sugerencia de gultron
pca_df,pca_model,n_cluster,kmeans,labels,cluster_centers,clientes_df_cluster,y_kmeans  = kmeans_fit(n_cluster,pd.DataFrame(clientes_encoder))
summary = cluster_percentage(clientes_df_cluster) 
summary["cluster_dist"] = summary_cluster(clientes_df_cluster)
kmeans_dict['kmeans_encoded_2']= {"pca_df":pca_df,"pca_model":pca_model,"n_clusters":n_cluster, "clientes_df":clientes_df,"clientes_df_scaled":clientes_df_scaled, "kmeans": kmeans,"labels": labels,"cluster_centers":cluster_centers,"clientes_df_cluster":clientes_df_cluster,"y_kmeans":y_kmeans,"summary":summary}

kmeans_plot(clientes_encoder)
plot_hist(kmeans_dict['kmeans_encoded_2'])
plot_pca_2d(kmeans_dict['kmeans_encoded_2']["pca_df"][["pca1","pca2","cluster"]])

n_cluster = 3 #sugerencia de gultron
pca_df,pca_model,n_cluster,kmeans,labels,cluster_centers,clientes_df_cluster,y_kmeans  = kmeans_fit(n_cluster,pd.DataFrame(clientes_encoder))
summary = cluster_percentage(clientes_df_cluster) 
summary["cluster_dist"] = summary_cluster(clientes_df_cluster)
kmeans_dict['kmeans_encoded_3']= {"pca_df":pca_df,"pca_model":pca_model,"n_clusters":n_cluster, "clientes_df":clientes_df,"clientes_df_scaled":clientes_df_scaled, "kmeans": kmeans,"labels": labels,"cluster_centers":cluster_centers,"clientes_df_cluster":clientes_df_cluster,"y_kmeans":y_kmeans,"summary":summary}

kmeans_plot(clientes_encoder)
plot_hist(kmeans_dict['kmeans_encoded_3'])
plot_pca_2d(kmeans_dict['kmeans_encoded_3']["pca_df"][["pca1","pca2","cluster"]])

# saving the dictionary with the complete model analysis
import pickle
# save dictionary to pickle file
with open('Resumen_gultron_v2.pickle', 'wb') as file:
    pickle.dump(kmeans_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

'''# laod a pickle file
with open("Resumen_gultron v1.pickle", "rb") as file:
    loaded_dict = pickle.load(file)
'''
fin = datetime.utcnow()
print("Fin de ejecucion: {0}".format(fin))    
#np.save('file.npy', kmeans_dict)
#new_dict = np.load('file.npy', allow_pickle='TRUE')

#pca_df.to_csv("pca_df.csv", index = False)
#gultron_2_cluster = kmeans_dict["kmeans_encoded_2"]["clientes_df_cluster"]
#gultron_2_cluster.to_csv("gultron_two_cluster.csv", index = False)
#gultron_3_cluster = kmeans_dict["kmeans_encoded_3"]["clientes_df_cluster"]
#gultron_3_cluster.to_csv("gultron_three_cluster.csv", index = False)

