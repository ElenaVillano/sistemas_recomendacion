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


### Instrucciones para correr el programa

1.- Se deberán instalar las paqueterías globales de python descritas arriba y se deberan bajar los siguientes archivos loc cuales deberán estar al mismo nivel. A continuación se muestra estra estructura:

- el_bueno.ipynb **-> cambiar nombre**
- sis_recom.py
- links_small.csv
- ratings_small.csv
- movies_metadata.csv

2.- Abrir el notebook **el_bueno.ipynb** y ejecutar cada celda. En el notebook se ejecuta lo siguiente:

- Carga de datos usando la función load_data del paquete *sis_recom.py*, con la que obtenemos la matriz de usuarios x películas y los raitings correspondientes, así como los nombres de las películas y un arreglo de asociación de posición del id de la película (para realizar las recomendaciones).
- Se realiza la separación del set de entrenamiento y de prueba
- Se manda llamar a la clase Matrix_Factorization que es la que se encarga de la factorización de matrices (U y V)
- Se gráfican los errores cuadráticos medios de cada iteración para ver el comportamiento del algoritmo.
- Después de realiza el cross validation
- En la parte de prueba del usuario se puede sustituir el id_user por otro id y obtener sus recomendaciones. En este caso sacamos 2 listas de recomendación. La primera incluye sólo las nuevas películas calificadas que el usuario no ha calificado (mo ha visto las películas). Y el segundo saca la lista incluyendo las películas que ya vio (para comparar lo que calificó vs lo que se obtuvo)
- También se puede sacar el desempeño NDCG para un suario en partícular y **el promedio de todos los usuarios.**


