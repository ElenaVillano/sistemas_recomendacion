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
    
    # ----------- Carga de datos para la matriz de ratings -------------------
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
    Separar conjuntos de test y training
    remueve 10 ratings de cada usuario y lo asigna a cada conjunto del test
    :param ratings: Dataframe de los ratings en matriz
    :return train, test: Datasets partidos para entrenamiento y prueba
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
    """
    Función de Error Cuadrático Medio
    :param actual: Valor real
    :param predicted: Valor predicho
    :return mean_error: Error cuadrático medio
    """
    suma_error = 0.0
    # loop sobre todos los valores
    for i in range(len(actual)):
        # el error es la suma de (actual - prediction)^2
        prediction_error =  actual[i] - predicted[i]
        suma_error += (prediction_error ** 2)
    # Normalizamos
    mean_error = suma_error / float(len(actual))
    return (mean_error)


def recomendaciones_id(Y, MCompleta, array_movies, bases_nombres_id, user, top=5):
    """
    Regresa id de las películas reomendadas dado un usuario
    :param Y: Recibe matriz con Nan/ceros
    :param MCompleta: Matriz predicha
    :param array_movies: arreglo de las películas
    :param bases_nombres_id: los nombres de las películas con su id,
    :param user: id del usuario
    :param top: default 5 para obtener el top 5 de películas recomendadas.
    :return recomendaciones: recomendaciones de películas que no ha calificado
    :return recomendaciones_t: recomendaciones de películas (considerando las que ya calificó
    """
    
    # ------------------------------------ Para nuevas recomendaciones------------------------------
    # Se multiplica boleano con Y para poner en cero los ratings dados por el usuario
    nvos = []
    # Agregando índice
    for i in range (len(MCompleta)):
        # Se concatena el indice de la pélicula
        nvos.append([array_movies[i],MCompleta[i]*(~Y[i].any())])
    
    # nvos es un arreglo se pasa a dataframe para mejor manejabilidad
    nvos = pd.DataFrame(nvos)
    # se colocan nombres a las columnas
    nvos.columns = ['id_movie','rating_recom']
    # se ordenan de forma descendente
    nvos = nvos.sort_values(by=['rating_recom'], ascending=False)
    # Borrando 0
    nvos = nvos[(nvos[['rating_recom']] != 0).all(axis=1)]
    # Obteniendo el top
    recomendaciones = nvos['id_movie'].head(5).to_numpy()
    
    # ------------------------------------ Para recomendaciones incluyendo las existentes ----------
    # Incluyendo los datos calificados
    todos = []
    # Agregando índice
    for i in range (len(MCompleta)):
        # Se concatena el indice de la pélicula
        todos.append([array_movies[i], MCompleta[i]])
    # todos es un arreglo se pasa a dataframe para mejor manejabilidad
    todos = pd.DataFrame(todos)
    # se colocan nombres a las columnas
    todos.columns = ['id_movie','rating_recom']
    # se ordenan de forma descendente
    todos = todos.sort_values(by=['rating_recom'], ascending=False)
    # Obteniendo el top
    recomendaciones_t = todos['id_movie'].head(5).to_numpy()  
    
    # Por si se quiere imprimir desde aqui las recomendaciones
    #print("Estás son las nuevas películas que te recomendamos usuario no. ", user)
    #print(bases_nombres_id[bases_nombres_id['movieId'].isin(recomendaciones)][["movieId", "title"]])
    
    #print("Estás películas te pueden interesar usuario no. ", user)
    #print(bases_nombres_id[bases_nombres_id['movieId'].isin(recomendaciones_t)][["movieId", "title"]])
    
    return recomendaciones, recomendaciones_t


def desempenio_NDCG(ratings, user):
    """
    Evaluar el desempeño para un usuario
    :param ratings: matriz a evaluar
    :param user: usuario a evaluar
    :return ndcg: normalized discounted cumulative gain
    """
    
    rating_1 = ratings.loc[user]
    rating_user = rating_1[rating_1!=0]
    suma_dcg = 0
    suma_idcg = 0
    for i in range(0, len(rating_user)):
        suma_dcg += rating_user[i] / np.log2(i+1 + 1)
        suma_idcg += (pow(2,rating_user[i]) - 1) / np.log2(i+1 + 1)

    ndcg = round(suma_dcg / suma_idcg,2)
    
    return ndcg

def obtain_ndcg_all_users(ratings):
    """
    Obtener el normalized discounted cumulative gain para todos los usuarios en la matriz
    :param ratings: Matriz a evaluar
    :return ndcg: Diccionario de todos los desempeños de cada usuario
    """
    ndcg = {}
    
    for user in ratings.index:
        ndcg.update({user:desempenio_NDCG(ratings, user)})
    return ndcg
