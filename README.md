# Proyecto_Final_ML
# Proyecto Final de Machine Learning

**Integrantes del equipo 07**

| Nombre | Clave |
| :------- | :------: |
|Elena Villalobos Nolasco|000197730|
|Carolina Acosta Tovany|000197627|
|Aide Jazmín González Cruz|000197900|

### Descripción

**Implementación del algoritmo de filtrado colaborativo**

### Paquetes necesarios

#### Paqueterías globales de python

- pandas
- numpy
- random
- matplotlib

#### Paqueterías realizadas

- sis_recom
- cli(recomiendame)


### Instrucciones para correr el programa

1.- Instalar paqueterías de python que se encuentran en el requirements.txt

`pip install -r requirements.txt`

2.- Colocar al mismo nivel de ejecución del notebook y de los .py, los .csv a analizar:

- links_small.csv
- ratings_small.csv
- movies_metadata.csv

Hay 2 formas de correr el programa:

- Con jupyter notebook
- En consola

Para correr con **jupyter notebook**

1.- Abrir el notebook **sistema_recomendacion.ipynb** y ejecutar cada celda. En el notebook se encontrará lo siguiente:

- Carga de datos usando la función ***load_data*** del paquete *sis_recom.py*, con la que obtenemos la matriz de usuarios, películas y *raitings* correspondientes (extraidos de los archivos *links_small.csv* y *ratings_small.csv*), un arreglo de nombres de las películas (que se extrae del archivo *movies_metadata.csv*) y un arreglo de asociación de posición del id de la película (para realizar las recomendaciones).
- Se realiza la separación del *set* de entrenamiento y de prueba
- Se manda llamar a la clase *Matrix_Factorization* que es la que se encarga de la factorización de matrices (U y V)
- Se gráfican los errores cuadráticos medios de cada iteración para ver el comportamiento del algoritmo.
- Después de realiza el cross validation
- En la parte de prueba del usuario se puede sustituir el ***id_user*** por otro id y obtener sus recomendaciones. En este caso sacamos 2 listas de recomendación. La primera incluye sólo las nuevas películas calificadas que el usuario no ha calificado (no ha visto las películas). Y el segundo saca la lista incluyendo las películas que ya vio (para comparar lo que calificó vs lo que se obtuvo)
- También se puede sacar el desempeño NDCG para un usuario en partícular y **el promedio de todos los usuarios.**


Para correr en **consola**

1. Abrir terminal, posicionarse en el repositorio donde se encuentren los archivos .py y ejectuar:

`recomiendame --help`

Este comando te dará información sobre las opciones con las que puede correr el módulo `recomiendame`

2. Ejecutar

`recomiendame --usuario 2`

Este comando cargará la matriz que se predijo durante nuestro entrenamiento con k=671 y desplegará las recomendaciones para el usuario 2.

3. Ejecutar

`recomiendame --crear_matriz --k 280 --usuario 10`

Este comando realizará el entrenamiento con k=280 y obtendremos una nueva matriz de recomendaciones, y desplegará las recomendaciones para el usuario 10.

**Ojo:** Si se ejecuta el comando con las opciones `--crear_matriz` y `--k` y después se ejecuta el comando para sólo recomendaciones, sin realizar el entrenamiento, estas recomendaciones se darán con la úlltima matriz entrenada.

## Estructura del proyecto

| Nombre archivo | Contenido|
|----------------|----------|
| **cli.py** |  Código para ejecución en consola |
| **matrix_factorization.py** | Clase de Factorización de matrices para predecir entradas vacías en una matriz |
| **matriz_recomendaciones.pkl** | Matriz obtenida de la predicción |
| **requirements.txt** | Paqueterías utilizadas |
| **setup.py** | Set up para ejecución en consola |
| **sis_recom.py** | Módulos necesarios para cargar datos, entrenar, predecir y obtener desempeño ncdg |
| **sistema_recomendacion.ipynb** | Jupyter notebook de ejecución para entrenar, predecir, cross validation y  obtención de desempeño con ncdg |


