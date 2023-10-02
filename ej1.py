import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
plt.show(block=True)

#hist, bins = np.histogram(img.flatten(),256, range=[0,256])

#img_expand = cv2.copyMakeBorder(img, 10,10,10,10, borderType=cv2.BORDER_REFLECT)
#plt.imshow(img_expand,cmap='gray')


def transformar_imagen(imagen, ventana_x = 10, ventana_y = 10, tipo_borde=cv2.BORDER_REFLECT):
  # Creamos la imagen a volcar el resultado
  imagen_final = np.zeros_like(imagen, dtype = np.uint8)
  # Creamos la imagen expandida
  alto, ancho = imagen.shape[:2]
  y_dif = ventana_y // 2
  x_dif = ventana_x // 2
  imagen_exp = cv2.copyMakeBorder(imagen, y_dif , y_dif,x_dif, x_dif, borderType = tipo_borde)
  for y in range(y_dif, alto + y_dif):
    for x in range(x_dif, ancho + x_dif):
      sub_im = imagen_exp[y-y_dif:y+y_dif+1, x-x_dif:x+x_dif+1] # Recortar la ventana
      sub_im_eq = cv2.equalizeHist(sub_im) # Ecualizar la ventana
      imagen_final[y-y_dif,x-x_dif] = sub_im_eq[y_dif,x_dif]
  return imagen_final


imagen_final = transformar_imagen(img, ventana_x=30, ventana_y=30, tipo_borde=cv2.BORDER_REPLICATE)
plt.imshow(imagen_final, cmap='gray')
plt.show(block=True)

