import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def load_data():
    """
    Carga datos
    Regresa 3 datasets:
    - Matriz de suarios, Películas y raitings. 
    - Arreglo de id de péliculas y nombres
    - Arreglo de nombres (id_pelicula)
    """
    
    # ----------- Carga de datos para la matriz de raitings -------------------
    links_small = pd.read_csv('links_small.csv')
    ratings_small = pd.read_csv('ratings_small.csv')
    
    # Construyendo la matriz Y_ai
    y_ia = links_small.set_index('movieId').join(ratings_small.set_index('movieId'))
    y_ia = y_ia.reset_index()
    #y_ia.pivot(index="userId", columns="movieId", values="rating") 
    y_ia = pd.DataFrame(y_ia.pivot(index='userId', columns='movieId', values='rating'))
    y_ia = pd.DataFrame(y_ia.to_records())
    # Eliminando usuario Nan
    y_ia = y_ia[pd.notnull(y_ia['userId'])]
    # Borrando columna 1 con user_id
    y_ia = y_ia.drop(['userId'], axis=1)
    # Cambiando Nan por zeros
    Y_0 = y_ia.copy()
    Y_0[np.isnan(Y_0)] = 0
    
    # -------------------- Catálogo de películas ---------------------------
    # Carga de datos
    movies_metadata = pd.read_csv('movies_metadata.csv',low_memory=False)
    
    # Procesamiento del archivo
    names_mov = pd.DataFrame(movies_metadata[['id','title']])
    df_mov = pd.DataFrame(links_small[['tmdbId','movieId']])
    
    # Renombramos Id por tmdbId
    names_mov = names_mov.rename(columns = {'id':'tmdbId'})
    
    # Quitando identificadores incorrectos
    malos = names_mov[names_mov['tmdbId'].str.contains('-')].index
    names_mov = names_mov.drop(index=malos)
    
    # Hacemos flotantes los identificadores
    names_mov['tmdbId'] = names_mov['tmdbId'].apply(lambda x: float(x))
    
    # Quitamos Nas de identificadores
    df_mov=df_mov.dropna(0)
    
    # Ordenamos titulos e identificadores
    id_base = df_mov.sort_values(by=['tmdbId'])
    titles_id = names_mov.sort_values(by=['tmdbId'])
    
    # Hacemos merge de ambas bases
    bases_nombres_id = id_base.merge(titles_id, on=['tmdbId'],how='left')
    
    # -------------------- Catálogo id películas ---------------------------
    id_movies = y_ia.columns.tolist()
    id_movies = np.array(id_movies).astype(int)
    
    return Y_0, bases_nombres_id, id_movies


def create_train_test(ratings):
    """
    separar conjuntos de test y training
    remueve 10 ratings de cada usuario
    y lo asigma a cada conjunto del test
    """
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        test_index = np.random.choice(
            np.flatnonzero(ratings[user]), size = 10, replace = False)

        train[user, test_index] = 0.0
        test[user, test_index] = ratings[user, test_index]
        
    # assert that training and testing set are truly disjoint
    assert np.all(train * test == 0)
    return train, test


def fun_ECM(actual, predicted):
    """Función de Error Cuadrático Medio"""
    suma_error = 0.0
    # loop sobre todos los valores
    for i in range(len(actual)):
        # el error es la suma de (actual - prediction)^2
        prediction_error =  actual[i] - predicted[i]
        suma_error += (prediction_error ** 2)
    # Normalizamos
    mean_error = suma_error / float(len(actual))
    return (mean_error)


