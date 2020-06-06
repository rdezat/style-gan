# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 20:10:25 2020

@author: ricard.deza.tripiana
"""

import ssl
import requests
import os

class wikiArtApi:
    
    def __init__(self, api_access_key, api_secret_key):
        self.api_access_key = api_access_key
        self.api_secret_key = api_secret_key
        
    def getSession(self,accessKey, secretKey):
        urlsession = 'https://www.wikiart.org/en/Api/2/login?accessCode='+accessKey+'&secretCode='+secretKey
        response = requests.get(urlsession)
        if response.status_code == 200: 
            sessionKey = response.json()['SessionKey']
        else:
            print(response.text)
        return sessionKey

    def createUrl(self,searchType,parameters):
        urlRoot = 'https://www.wikiart.org/en/api/2/'
        urlSearchType = urlRoot + searchType
        url = urlSearchType + '?'
        i = 0
        for parm in parameters:
            if i != 0:
                url = url + '&'
            url = url + parm + '=' + parameters[parm]
            i = i + 1
        return url
    
    def searchApi(self,searchType,parameters):
        morePages = True
        dataJSON = []
        while morePages == True:
            url = self.createUrl(searchType, parameters)
            response = requests.get(url)
            if response.status_code == 200:
                dataJSON.append(response.json()['data'])
                if response.json()['hasMore'] == True:
                    paginationToken = response.json()['paginationToken']
                    parameters['paginationToken'] = paginationToken
                else:
                    morePages = False
        return (dataJSON)

    def getImages(self,data):
        imagesDict = []
        for page in data:
            for p in page:
                # paintingId = p['id']
                # imageUrl = p['image']
                imagesDict.append([{'id': p['id'], 'url': p['image'], 'artist': p['artistUrl'], 'name': p['url']}])
        imagesDictList = [item for sl in imagesDict for item in sl]
        return imagesDictList
    
    def getStyles(self,images):
        searchType = 'Painting'
        for img in images:
            searchParam = {'id': img['id']}
            url = self.createUrl(searchType, searchParam)
            response = requests.get(url)
            if response.status_code == 200:
                img['styles'] = response.json()['styles']
        return images

    def getImageStyle(self,image):
        searchType = 'Painting'
        searchParam = {'id': image['id']}
        url = self.createUrl(searchType, searchParam)
        response = requests.get(url)
        if response.status_code == 200:
            image['styles'] = response.json()['styles']
        else:
            print(image['id'])
        return image
    
    def filterImages(self,images, style):
        result = []
        for img in images:
            if 'styles' in img:
                if style in img['styles']:
                    result.append(img)
        return result
    
    def downloadImageToLocal(self,images,artist):
        if not os.path.exists(artist):
            os.mkdir(artist)
            print("Directory " , artist ,  " Created ")
        else:    
            print("Directory " , artist ,  " already exists")
        for img in images:
            img_data = requests.get(img['url']).content
            with open('dataset/' + artist + '/' + img['name']+'.jpg', 'wb') as handler:
                handler.write(img_data)
        