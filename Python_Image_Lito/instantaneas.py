#from turtle import position
import cv2 as cv
#import numpy as np
from PIL import ImageFont, ImageDraw, Image  

from Kivy.MrGadget3D import Litofanias
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
print(filename)
#imagen = 'Imagenes_PhotoBox/Instantanea1.jpg'
#imagen1 = 'Imagenes/Fan5.jpg'
imagen1 = 'PyJ.jpg'
imagen5 = 'C:/Users/Jvele/Pictures/Imagenes/F1.jpg'
imagen2 = 'C:/Users/Jvele/Pictures/Imagenes/Fanu2.jpg'
imagen3 = 'C:/Users/Jvele/Pictures/Imagenes/Fan3.jpg'
imagen4 = 'C:/Users/Jvele/Pictures/Fanu4.png'
#PhotoBox =Litofanias(imagen3)
#cv.imshow('imagen',PhotoBox.Write_Scrable('Fanu','V',(10,30)))


PhotoBox1 =Litofanias(filename)
cv.imshow('imagen',PhotoBox1.Image_instantanea())
cv.imshow('image1',PhotoBox1.Image_instantanea_txt('te amo',(125, 406)))

PhotoBox2 =Litofanias(filename)
cv.imshow('imagen1',PhotoBox2.Corazon_Mensaje('Fa & Je'))
cv.imshow('imagen1',PhotoBox2.Imagen_txt('Te amo',(20,20)))
cv.imshow('image1',PhotoBox1.Image_instantanea_txt('te amo',(125, 406)))
cv.imwrite('litofania_instantanea_texto.jpg',PhotoBox1.Image_instantanea_txt('Feliz dia de san Valentin',(10, 416)))
cv.imwrite('litofania_instantanea.jpg',PhotoBox1.Image_instantanea())
cv.imwrite('litofania_box.jpg',PhotoBox1.Imagen_txt('Feliz 14 de 14 meses juntos',(190,20)))

PhotoBox3 =Litofanias(imagen1)
cv.imshow('imagen1',PhotoBox3.Mosaico(imagen2,imagen3,imagen4))
cv.waitKey(0)
cv.destroyAllWindows()
