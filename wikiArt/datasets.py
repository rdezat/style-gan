# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:51:08 2020

@author: ricard.deza.tripiana
"""

#Descarrega d'imatges de wikiart

from wikiArtApi import *
import numpy as np
import os

#Definim les claus d'acces a la API
api_access_key = '4beeaa873b844ad4'
api_secret_key = 'b81d0b303e64e1cd'

wka = wikiArtApi(api_access_key, api_secret_key)

#Recuperem la sessió [No es necesari per utilitzar la API]
session_key = wka.getSession(api_access_key, api_secret_key)

#Busquem les obres de Picasso
searchType = 'PaintingsByArtist'
picassoId = '57726d84edc2cb3880b48c4d'
searchParam = {'id': picassoId}
picassoPaintings = wka.searchApi(searchType, searchParam)

#Recuperem els id's i url de les imatges de Picasso
picassoImagesDict = wka.getImages(picassoPaintings)

#Recollim els estils de les imatges
for i in range(0,len(picassoImagesDict)):
    picassoImagesDict[i] = wka.getImageStyle(picassoImagesDict[i])
for i in range(320,len(picassoImagesDict)):
    picassoImagesDict[i] = wka.getImageStyle(picassoImagesDict[i])
for i in range(594,len(picassoImagesDict)):
    picassoImagesDict[i] = wka.getImageStyle(picassoImagesDict[i])

#Filtrem les imatges per estil
picassoCubism = wka.filterImages(picassoImagesDict, 'Cubism')

# picassoImagesDictV0 = []
# picassoImagesDictV1 = []
# picassoImagesDictV2 = []
# picassoImagesDictV3 = []
# picassoImagesDictV4 = []
# picassoImagesDictV5 = []
# for i in range(0,200):
#     picassoImagesDictV0.append(picassoImagesDict[i])
# for i in range(200,400):
#     picassoImagesDictV1.append(picassoImagesDict[i])
# for i in range(400,600):
#     picassoImagesDictV2.append(picassoImagesDict[i])
# for i in range(600,800):
#     picassoImagesDictV3.append(picassoImagesDict[i])
# for i in range(800,1000):
#     picassoImagesDictV4.append(picassoImagesDict[i])
# for i in range(1000,len(picassoImagesDict)):
#     picassoImagesDictV5.append(picassoImagesDict[i])

# #Recollim els estils de les imatges
# picassoImagesStyleV0 = wka.getStyles(picassoImagesDictV0)
# picassoImagesStyleV1 = wka.getStyles(picassoImagesDictV1)
# picassoImagesStyleV2 = wka.getStyles(picassoImagesDictV2)
# picassoImagesStyleV3 = wka.getStyles(picassoImagesDictV3)
# picassoImagesStyleV4 = wka.getStyles(picassoImagesDictV4)
# picassoImagesStyleV5 = wka.getStyles(picassoImagesDictV5)

# picassoImagesStyle = []
# picassoImagesStyle.extend(picassoImagesStyleV0)
# picassoImagesStyle.extend(picassoImagesStyleV1)
# picassoImagesStyle.extend(picassoImagesStyleV2)
# picassoImagesStyle.extend(picassoImagesStyleV3)
# picassoImagesStyle.extend(picassoImagesStyleV4)
# picassoImagesStyle.extend(picassoImagesStyleV5)

#Filtrem les imatges per estil
# picassoCubism = wka.filterImages(picassoImagesStyle, 'Cubism')

#Busquem les obres de Rubens
searchType = 'PaintingsByArtist'
rubensId = '57726d84edc2cb3880b48b01'
searchParam = {'id': rubensId}
rubensPaintings = wka.searchApi(searchType, searchParam)

#Recuperem els id's i url de les imatges de Rubens
rubensImagesDict = wka.getImages(rubensPaintings)

#Recollim els estils de les imatges
for i in range(0,len(rubensImagesDict)):
    rubensImagesDict[i] = wka.getImageStyle(rubensImagesDict[i])

#Filtrem les imatges per estil
rubensBaroque = wka.filterImages(rubensImagesDict, 'Baroque')

#Busquem les obres de Pollock
searchType = 'PaintingsByArtist'
pollockId = '57726d7aedc2cb3880b478b9'
searchParam = {'id': pollockId}
pollockPaintings = wka.searchApi(searchType, searchParam)

#Recuperem els id's i url de les imatges de Rubens
pollockImagesDict = wka.getImages(pollockPaintings)

#Recollim els estils de les imatges
for i in range(0,len(pollockImagesDict)):
    pollockImagesDict[i] = wka.getImageStyle(pollockImagesDict[i])
for i in range(0,7):
    pollockImagesDict[i] = wka.getImageStyle(pollockImagesDict[i])

#Filtrem les imatges per estil
pollockAbstract = []
for img in pollockImagesDict:
    if 'styles' in img:
        if 'Expressionism' not in img['styles']:
            pollockAbstract.append(img)

wka.downloadImageToLocal(picassoCubism,'picasso')
wka.downloadImageToLocal(picassoCubism[173:],'picasso')
wka.downloadImageToLocal(rubensBaroque,'rubens')
wka.downloadImageToLocal(pollockAbstract,'pollock')

