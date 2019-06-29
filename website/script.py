#!/usr/bin/env python
import sys
import os
import scipy
import numpy as np
import skimage

from skimage import feature, exposure, io, color
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.externals import joblib

import warnings
warnings.filterwarnings('ignore')

def calcula_histograma_cor(image):
	hist_r = exposure.histogram(image[:,:,0]) #red
	hist_g = exposure.histogram(image[:,:,1]) #blue
	hist_b = exposure.histogram(image[:,:,2]) #green
	x = np.append(hist_r[0], hist_g[0])
	x = np.append(x, hist_b[0])
	return x

def calcula_histograma_lbp(image):
	#image, vizinhança, raio do circulo, metodo
	lbp = feature.texture.local_binary_pattern(image, 8, 2, method='uniform')
	histogram = scipy.stats.itemfreq(lbp)
	x = histogram[:,1]
	norm_hist = normalize(x[:,np.newaxis], axis=0).ravel()
	return x

def calcula_haralick_features(image):
	x = []

	f = feature.texture.greycomatrix(image, [1,2], [0,  np.pi/2], 256, symmetric=True, normed=True)
	x.append(feature.texture.greycoprops(f, prop = 'contrast')[0,0])
	x.append(feature.texture.greycoprops(f, prop = 'contrast')[0,1])
	x.append(feature.texture.greycoprops(f, prop = 'contrast')[1,0])
	x.append(feature.texture.greycoprops(f, prop = 'contrast')[1,1])

	x.append(feature.texture.greycoprops(f, prop = 'dissimilarity')[0,0])
	x.append(feature.texture.greycoprops(f, prop = 'dissimilarity')[0,1])
	x.append(feature.texture.greycoprops(f, prop = 'dissimilarity')[1,0])
	x.append(feature.texture.greycoprops(f, prop = 'dissimilarity')[1,1])

	x.append(feature.texture.greycoprops(f, prop = 'homogeneity')[0,0])
	x.append(feature.texture.greycoprops(f, prop = 'homogeneity')[0,1])
	x.append(feature.texture.greycoprops(f, prop = 'homogeneity')[1,0])
	x.append(feature.texture.greycoprops(f, prop = 'homogeneity')[1,1])

	x.append(feature.texture.greycoprops(f, prop = 'ASM')[0,0])
	x.append(feature.texture.greycoprops(f, prop = 'ASM')[0,1])
	x.append(feature.texture.greycoprops(f, prop = 'ASM')[1,0])
	x.append(feature.texture.greycoprops(f, prop = 'ASM')[1,1])

	x.append(feature.texture.greycoprops(f, prop = 'energy')[0,0])
	x.append(feature.texture.greycoprops(f, prop = 'energy')[0,1])
	x.append(feature.texture.greycoprops(f, prop = 'energy')[1,0])
	x.append(feature.texture.greycoprops(f, prop = 'energy')[1,1])

	return x

def get_class(encoded, y, num):
    for i in range(len(encoded)):
        if(encoded[i] == num):
            break
    return y[i]

def classify_lbp(image, encoded, y):
    
    knn = joblib.load("../extracted_features/knn_lbp.joblib")
    im = color.rgb2gray(io.imread(image))
    x = calcula_histograma_lbp(im)
    a = knn.predict(x.reshape(1,-1))
    print(get_class(encoded, y, a))

def classify_hist(image, encoded, y):
    
    knn = joblib.load('../extracted_features/knn_hist.joblib')
    im = skimage.img_as_float(io.imread(image))
    x = calcula_histograma_cor(im)
    a = knn.predict(x.reshape(1,-1))
    print(get_class(encoded, y, a))

def classify_har(image, encoded, y):
    
    knn = joblib.load('../extracted_features/knn_har.joblib')
    im = skimage.img_as_ubyte(color.rgb2gray(io.imread(image)))
    x = calcula_haralick_features(im)
    a = knn.predict(np.array(x).reshape(1,-1))
    print(get_class(encoded, y, a))


if __name__ == "__main__":
    image = sys.argv[1]
    tecnica = sys.argv[2]

    # print(tecnica)
    	
    MAIN_DIR = "../frutas_dataset_train"
    y = [name for name in os.listdir(MAIN_DIR) if os.path.isdir(os.path.join(MAIN_DIR, name))]
    result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(MAIN_DIR) for f in filenames if os.path.splitext(f)[1] == '.jpg']
    #transformar as labels de string para ints
    le = preprocessing.LabelEncoder()
    encoded = le.fit_transform(y)
    
    if tecnica == "lbp":
        classify_lbp(image, encoded, y)

    if tecnica == "hist":
        classify_hist(image, encoded, y)

    if tecnica == "har":
        classify_har(image, encoded, y)

