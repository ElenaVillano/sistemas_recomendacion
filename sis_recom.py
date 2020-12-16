import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def load_data():
    """
    Carga datos
    Regresa dataframe de Usuarios, Películas y raitings
    """
    
    # Carga de datos
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

    
    return Y_0  


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





