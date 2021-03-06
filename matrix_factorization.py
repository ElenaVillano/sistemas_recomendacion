"""
Módulo de la clase Matrix_Factorization
"""
import numpy as np
from sis_recom import fun_ECM

class Matrix_Factorization:
    """
    Factorización de matrices para predecir entradas vacías en una matriz
    
    Parametros
    ----------
    n_iters : numero de iteraciones para entregar algoritmo
        
    k : rank de la matriz
        
    _lambda : regularización
    """
    
    def __init__(self, n_iters, k, _lambda):
        self._lambda = _lambda
        self.n_iters = n_iters
        self.k = k  
        
    def fit(self, train, test):
        """
        partición de entrenamiento y test
        y selección de vectores aleatorios
        """
        self.n, self.m = train.shape
        
        self.Xu = np.random.random((self.n, self.k))
        self.Yi = np.random.random((self.m, self.k))
        
        self.test_ecm_record  = []
        self.train_ecm_record = []
        
        for _ in range(self.n_iters):
            self.Xu = self._als_step(train, self.Xu, self.Yi)
            self.Yi = self._als_step(train.T, self.Yi, self.Xu) 
            predictions = self.predict()
            test_ecm = self.compute_ecm(test, predictions)
            train_ecm = self.compute_ecm(train, predictions)
            self.train_ecm_record.append(train_ecm)
            self.test_ecm_record.append(test_ecm)
        
        return self    
    
    def _als_step(self, ratings, vec_res, vec_fij):
        """
        función que obtiene los vectores X_u ó Y_i
        """
        A = vec_fij.T.dot(vec_fij) + np.eye(self.k) * self._lambda
        b = ratings.dot(vec_fij)
        A_inv = np.linalg.inv(A)
        vec_res = b.dot(A_inv)
        return vec_res
    
    def predict(self):
        """Predicción de matriz completa XuYi (UV)"""
        pred = self.Xu.dot(self.Yi.T)
        return pred
    
    @staticmethod
    def compute_ecm(y_true, y_pred):
        mask = np.nonzero(y_true)
        ecm = fun_ECM(y_true[mask], y_pred[mask])

        return ecm
