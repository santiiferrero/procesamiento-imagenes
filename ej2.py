import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys


formulario = 'formulario_vacio.png'
img = cv2.imread(formulario,cv2.IMREAD_GRAYSCALE)

def umbralizar(imagen):
  '''Recibe una imagen y le aplica
  un umbral binario invertido'''
  T = (img.min() + img.max())/2
  flag = False
  while ~flag:
    g = img >= T
    Tnext = 0.5*(np.mean(img[g]) + np.mean(img[~g]))
    flag = np.abs(T-Tnext) < 0.5
    T = Tnext
  _, img_th  = cv2.threshold(imagen, thresh=T, maxval=255, type=cv2.THRESH_BINARY_INV)
  return img_th


def hallar_filas(imagen):
  '''Dada una imagen, devuelve un array de índices que representan en dónde se encuentra cada renglón'''
  #Umbralizamos la imagen
  img_th = imagen<170
  img_rows = np.sum(img_th,1) #sumar valor de pixels en c/fila
  #Definimos un umbral
  umbral_rows = 800
  #Aplicamos umbral
  img_rows_th = img_rows > umbral_rows
  y = np.diff(img_rows_th)
  y_indxs = np.argwhere(y)

  # Reshapeo el array para ir seleccionando donde empieza y termina cada renglón
  y_indxs_unpaired = y_indxs.reshape((1,-1)).tolist()[0]
  # y_indxs marca donde empieza y termina la línea, quiero lo mismo pero para los renglones
  y_indxs_unpaired = y_indxs_unpaired[1:-1]
  renglones_ind = np.array(y_indxs_unpaired).reshape((-1,2))

  return renglones_ind

def hallar_columnas(imagen):
  '''Dada una imagen, devuelve un array de índices que representan donde se encuentra cada columna'''
  #Umbralizamos la imagen
  img_th = imagen<170#umbralizar(imagen)
  img_cols = np.sum(img_th,0) #sumar valor de pixels en c/columna
  #Definimos un umbral
  umbral_cols = 30
  #aplicamos umbral
  img_cols_th = img_cols > umbral_cols

  x= np.diff(img_cols_th)
  x_indxs= np.argwhere(x)

  # Reshapeo el array para ir seleccionando donde empieza y termina cada renglón
  x_indxs_unpaired = x_indxs.reshape((1,-1)).tolist()[0]
  # y_indxs marca donde empieza y termina la línea, quiero lo mismo pero para los renglones
  x_indxs_unpaired = x_indxs_unpaired[1:-1]
  columnas_ind = np.array(x_indxs_unpaired).reshape((-1,2))

  return columnas_ind


#funcion para contar letras.
#basicamente se utiliza todo de cv2.connectedComponentsWithStats

#lo mas importante son los argumentos que recibe la funcion
#th_area = 10  # Umbral del area de componentes conectadas, para que no se cuenten los bordes


def contar_letras(binary_image, th_area):
    # Realizar la segmentación de componentes conectados
    comp = cv2.connectedComponentsWithStats(binary_image, 8, cv2.CV_32S)


    # Inicializar una lista para contar letras
    letter_counts = []

    #  Umbralado para que no se cuente como componentes lo que haya quedado de los bordes del campo. Estaba en la ayuda de los profes
    ix_area = comp[2][:, -1] > th_area
    stats = comp[2][ix_area, :]

    # Iterar a través de las estadísticas para analizar los componentes conectados restantes
    for i in range(1, len(stats)):  # Comenzamos desde 1 para omitir el fondo

        # Area del componente conectado
        area = stats[i, cv2.CC_STAT_AREA]

        # Definimos el area del que tendra que ser mayor
        if area > 5:
            letter_counts.append(1)

    return letter_counts


#funcion para contar palabras.
#basicamente se utiliza todo de cv2.connectedComponentsWithStats, ahi estan todas las estadisitcas de las distancias

#lo mas importante son los argumentos que recibe la funcion
#th_area = 10  # Umbral del area de componentes conectadas
# umbral_distancia = 6  # Umbral de distancia entre palabras

def contar_palabras(binary_image, th_area, umbral_distancia):

    # Componentes conectados
    comp = cv2.connectedComponentsWithStats(binary_image, 8, cv2.CV_32S)

    # Lista para contar la cantidad de palabras
    palabras = []

    # Umbralado para que no se cuente como componentes lo que haya quedado de los bordes del campo. Estaba en la ayuda de los profes
    ix_area = comp[2][:, -1] > th_area
    stats = comp[2][ix_area, :]

    # Ordenar las estadísticas por coordenada X
    indices_ordenados = np.argsort(stats[:, 0])
    stats = stats[indices_ordenados]

    # Inicializar variables para rastrear las posiciones de inicio y fin de una palabra
    inicio_palabra = stats[0, 0]
    fin_palabra = inicio_palabra + stats[0, 2]

    # Iterar a través de las estadísticas para analizar los componentes conectados restantes
    for i in range(1, len(stats)):  # Comenzamos desde 1 para omitir el fondo
        x = stats[i, 0]

        # Calcular la distancia entre el componente actual y el anterior
        distancia = x - fin_palabra

        if distancia > umbral_distancia:
            # Se considera una nueva palabra si la distancia es mayor que el umbral
            palabras.append(1)

        fin_palabra = x + stats[i, 2]

    # Agregar la última palabra
    palabras.append(1)

    return len(palabras)

def validar_requerimientos(renglones_filtrados):
  resultados = {}

  #NOMBRE Y APELLIDO
  if (renglones_filtrados[0][0]['letras'] > 25 or renglones_filtrados[0][0]['palabras'] < 2):
    resultados["Nombre y apellido: "] = "MAL"
  else:
      resultados["Nombre y apellido: "] = "OK"
    
    #EDAD
  if renglones_filtrados[1][0]['letras'] < 2 or renglones_filtrados[1][0]['letras'] > 3:
    resultados["Edad: "] = "MAL"
  else:
      resultados["Edad: "] = "OK"
  
    #MAIL
  if renglones_filtrados[2][0]['letras'] > 25 or renglones_filtrados[2][0]['palabras'] != 1:
    resultados["Mail: "] = "MAL"
  else:
    resultados["Mail: "] = "OK"

    #LEGAJO
  if renglones_filtrados[3][0]['letras'] != 8 or renglones_filtrados[3][0]['palabras'] != 1:
    resultados["Legajo: "] = "MAL"
  else:
    resultados["Legajo: "] = "OK"

  #PREGUNTA 1
  if (renglones_filtrados[4][0]['letras'] == 1 and renglones_filtrados[4][1]['letras'] == 0):
      resultados["Pregunta 1: "] = "OK"
  elif (renglones_filtrados[4][0]['letras'] == 0 and renglones_filtrados[4][1]['letras'] == 1):
      resultados["Pregunta 1: "] = "OK"
  else:
      resultados["Pregunta 1: "] = "MAL"
  #PREGUNTA 2
  if (renglones_filtrados[5][0]['letras'] == 1 and renglones_filtrados[5][1]['letras'] == 0):
      resultados["Pregunta 2: "] = "OK"
  elif (renglones_filtrados[5][0]['letras'] == 0 and renglones_filtrados[5][1]['letras'] == 1):
      resultados["Pregunta 2: "] = "OK"
  else:
      resultados["Pregunta 2: "] = "MAL"
  #PREGUNTA 3
  if (renglones_filtrados[6][0]['letras'] == 1 and renglones_filtrados[6][1]['letras'] == 0):
      resultados["Pregunta 3: "] = "OK"
  elif (renglones_filtrados[6][0]['letras'] == 0 and renglones_filtrados[6][1]['letras'] == 1):
      resultados["Pregunta 3: "] = "OK"
  else:
      resultados["Pregunta 3: "] = "MAL"
  #COMENTARIOS
  if renglones_filtrados[7][0]['letras'] > 25:
    resultados["Comentarios: "] = "MAL"
  else:
    resultados["Comentarios: "] = "OK"

  return resultados
     

def validar(formulario):
  #Guarda formulario
  img = cv2.imread(formulario,cv2.IMREAD_GRAYSCALE)

  #Umbralizamos la imagen
  img_th = umbralizar(img)

  img_cols = np.sum(img,0) #sumar valor de pixels en c/columna

  img_rows = np.sum(img,1) #sumar valor de pixels en c/fila

  rows_indxs = hallar_filas(img)

  #columnas_ind = hallar_columnas(img_th)

  renglones = []


  for ir, renglon_ids in enumerate(rows_indxs):
    imagen_renglon = img[renglon_ids[0]:renglon_ids[1],:]
    cols_indxs = hallar_columnas(imagen_renglon)
    renglon = []
    #print(cols_indxs)
    for ic, columna_ids in enumerate(cols_indxs):

      renglon.append({
          'fila': ir+1,
          'columna': ic+1,
          'cord_x': columna_ids,
          'cord_y': renglon_ids, # coordenadas de la seccion VERIFICAR SI ES ASI EL ORDEN, CAPAZ LOS PUSE AL REVES
          'img': imagen_renglon[:,columna_ids[0]:columna_ids[1]]
      })
    renglones.append(renglon)

  renglones.pop(5)

    # filtraremos solo los renglones de las columnas 2 y 3
  renglones_filtrados = []

  for renglon in renglones:
    # filtramos las columnas 2 y 3 que son las que nos interesan
    elementos_filtrados = [elemento for elemento in renglon if elemento['columna'] in [2, 3]]

    if elementos_filtrados:
        renglones_filtrados.append(elementos_filtrados)

  #DETECTAMOS LETRAS
  for ii, renglon in enumerate(renglones_filtrados):
    for ic, columna in enumerate(renglon):
        binary_image = umbralizar(columna["img"])

        # Contar letras en la celda actual
        th_area = 10  # Umbral de área para eliminar componentes conectadas pequeñas
        letter_counts = contar_letras(binary_image, th_area)

        # Agregar al diccionario de cada celda la cantidad de letras
        columna['letras'] = len(letter_counts) - 1 # Restar 1 ya que cuenta el área de la celda

#DETECTAMOS PALABRAS
  for ii, renglon in enumerate(renglones_filtrados):
    for ic, columna in enumerate(renglon):
        binary_image = umbralizar(columna["img"])

        th_area = 10  # Umbral del area de componentes conectadas
        umbral_distancia = 6  # Umbral de distancia entre palabras
        cant_palabras = contar_palabras(binary_image, th_area, umbral_distancia)

        # Agregar al diccionario de cada celda la cantidad de palabras
        columna['palabras'] = cant_palabras


  resultados = validar_requerimientos(renglones_filtrados)
    

  return(resultados)


formulario = sys.argv[1]   #'formulario_01.png'
resultado = validar(formulario)

print(resultado)