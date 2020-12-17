"""
Módulo principal para correr el sistema de recomendación
"""

import numpy as np
import pandas as pd
import pickle
import os

# Importación de paqueterías creadas por nosotras
from sis_recom import load_data
from sis_recom import create_train_test, fun_ECM
from sis_recom import recomendaciones_id
from sis_recom import desempenio_NDCG, obtain_ndcg_all_users
from matrix_factorization import Matrix_Factorization

import click


@click.command()
@click.option('--crear_matriz', default=False, is_flag=True, help="Creará la matriz de nuevo")
@click.option('--k', default=671, help="K con la que se quiere construir la matriz")
@click.option('--usuario', default=1, help='Usuario del que quieres recomendacion')

def main(usuario, crear_matriz, k):
    # Cargamos datos ya limpios
    ratings, bases_nombres_id, arr_movies = load_data()
    n_users, n_items = ratings.shape

    if crear_matriz:
        # Separando datos en train y test
        train, test = create_train_test(np.array(ratings))

        # Entrenamos
        als = Matrix_Factorization(n_iters=15, k=k, _lambda=0.01)
        als.fit(train, test)


        # Esta es la matriz completa para poder hacer recomendaciones
        MCompleta = als.predict()
        MCompleta = pd.DataFrame(MCompleta,
                                 columns=ratings.columns,
                                 index=ratings.index)

        # Guardamos la matriz en un pickle
        output = os.path.join(os.getcwd(),'matriz_recomendaciones.pkl')
        pickle.dump(MCompleta, open(output, "wb"))

    # Recuperar el pickle
    output = os.path.join(os.getcwd(),'matriz_recomendaciones.pkl')
    MCompleta = pickle.load(open(output, "rb"))
    
    id_user = usuario
    rec_nvas, rec_todas = recomendaciones_id(np.array(ratings)[id_user - 1],
                                             np.array(MCompleta)[id_user - 1],
                                             arr_movies,
                                             bases_nombres_id,
                                             id_user)


    # Películas recomendadas que no ha visto
    print("Estás son las nuevas películas que te recomendamos usuario no. ", id_user)
    print(bases_nombres_id[bases_nombres_id['movieId'].isin(rec_nvas)][["movieId", "title"]])


    # Películas recomendadas que probablemente ya vio
    print("Estás películas te pueden interesar usuario no. ", id_user)
    print(bases_nombres_id[bases_nombres_id['movieId'].isin(rec_todas)][["movieId", "title"]])


__version__ = '0.1.0'