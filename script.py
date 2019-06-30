# extrair as imagens junto com suas classes

# X_test_lbp
# X_test_hist
# X_test_har

# k_test

import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
import skimage

from skimage import exposure
from skimage import feature
from skimage import io
from skimage import color
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn import datasets
from sklearn import preprocessing

import warnings
warnings.filterwarnings('ignore')

def find_files(MAIN_DIR):
	#path_to_files
	y = [name for name in os.listdir(MAIN_DIR) if os.path.isdir(os.path.join(MAIN_DIR, name))]
	result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(MAIN_DIR) for f in filenames if os.path.splitext(f)[1] == '.jpg']

	#transformar as labels de string para ints
	le = preprocessing.LabelEncoder()
	encoded = le.fit_transform(y)

	#qtde de arquivos por pasta
	num_files = []
	for i in y:
		DIR = MAIN_DIR + "/" + i + "/"
		num_files.append(len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))

	#criando o target label == classes das frutas
	k = []
	for i in range(len(y)):
	    for j in range(num_files[i]):
	        k.append(encoded[i])
	return result, num_files, k, y, encoded

MAIN_DIR = "frutas_dataset_test"

result, num_files, k_test, y, encoded = find_files(MAIN_DIR)

def calcula_histograma_lbp(image):
	#image, vizinhança, raio do circulo, metodo
	lbp = feature.texture.local_binary_pattern(image, 8, 2, method='uniform')
	histogram = scipy.stats.itemfreq(lbp)
	x = histogram[:,1]
	norm_hist = normalize(x[:,np.newaxis], axis=0).ravel()
	return x

def calcula_histograma_cor(image):
	hist_r = exposure.histogram(image[:,:,0]) #red
	hist_g = exposure.histogram(image[:,:,1]) #blue
	hist_b = exposure.histogram(image[:,:,2]) #green
	x = np.append(hist_r[0], hist_g[0])
	x = np.append(x, hist_b[0])
	return x

def calcula_haralick_features(image):
	x = []

	f = feature.texture.greycomatrix(im, [1,2], [0,  np.pi/2], 256, symmetric=True, normed=True)

	for i in range(2):
		for j in range(2):
			x.append(feature.texture.greycoprops(f, prop = 'contrast')[i,j])

	for i in range(2):
		for j in range(2):
			x.append(feature.texture.greycoprops(f, prop = 'dissimilarity')[i,j])

	for i in range(2):
		for j in range(2):
			x.append(feature.texture.greycoprops(f, prop = 'homogeneity')[i,j])

	for i in range(2):
		for j in range(2):
			x.append(feature.texture.greycoprops(f, prop = 'ASM')[i,j])

	for i in range(2):
		for j in range(2):
			x.append(feature.texture.greycoprops(f, prop = 'energy')[i,j])
	return x

X_lbp = np.zeros((len(result), 10))
for i in range(len(result)):
    im = color.rgb2gray(io.imread(result[i]))
    x = calcula_histograma_lbp(im)
    X_lbp[i,:] = x

X_hist = np.zeros((len(result), 768))
for i in range(len(result)):
    im = skimage.img_as_float(io.imread(result[i]))
    x = calcula_histograma_cor(im)
    X_hist[i,:] = x

X_har = np.zeros((len(result), 20))
for i in range(len(result)):
	im = skimage.img_as_ubyte(color.rgb2gray(io.imread(result[i])))
	x = calcula_haralick_features(im)
	X_har[i,:] = x

from sklearn.externals import joblib
knn_lbp = joblib.load('knn_lbp.joblib')
knn_hist = joblib.load('knn_hist.joblib')
knn_har = joblib.load('knn_har.joblib')


k_lbp_pred = knn_lbp.predict(X_lbp)
k_hist_pred = knn_hist.predict(X_hist)
k_har_pred = knn_har.predict(X_har)

from sklearn import metrics
print("Taxa de acerto LBP:", metrics.accuracy_score(k_test, k_lbp_pred))
print("Taxa de acerto Histograma de Cor:", metrics.accuracy_score(k_test, k_hist_pred))
print("Taxa de acerto Haralick Features:", metrics.accuracy_score(k_test, k_har_pred))