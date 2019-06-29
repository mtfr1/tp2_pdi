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

#funcao basica para mostrar imagens
def show(img):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.set_axis_off()
    plt.show()



#retorna um vetor com o path para todos os arquivos, outro vetor com o numero de arquivos dentro de cada classe
#e o vetor target k
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
		DIR = "frutas_dataset_train/" + i + "/"
		num_files.append(len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))

	#criando o target label == classes das frutas
	k = []
	for i in range(len(y)):
	    for j in range(num_files[i]):
	        k.append(encoded[i])
	return result, num_files, k

MAIN_DIR = "frutas_dataset_train"

result, num_files, k = find_files(MAIN_DIR)

def calcula_histograma_lbp(image):
	#image, vizinhança, raio do circulo, metodo
	lbp = feature.texture.local_binary_pattern(image, 8, 2, method='uniform')
	histogram = scipy.stats.itemfreq(lbp)
	x = histogram[:,1]
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
	f = feature.texture.greycomatrix(image, [1], [0, np.pi/2])
	arr = feature.texture.greycoprops(f, prop = 'contrast').ravel()
	x.append(arr[0])
	x.append(arr[1])
	arr = (feature.texture.greycoprops(f, prop = 'dissimilarity').ravel())
	x.append(arr[0])
	x.append(arr[1])
	arr = (feature.texture.greycoprops(f, prop = 'homogeneity').ravel())
	x.append(arr[0])
	x.append(arr[1])
	arr = (feature.texture.greycoprops(f, prop = 'ASM').ravel())
	x.append(arr[0])
	x.append(arr[1])
	arr = (feature.texture.greycoprops(f, prop = 'energy').ravel())
	x.append(arr[0])
	x.append(arr[1])
	return x

# X_lbp = np.zeros((len(result), 10))
# for i in range(len(result)):
#     im = color.rgb2gray(io.imread(result[i]))
#     x = calcula_histograma_lbp
#     X[i,:] = x

# X_hist = np.zeros((len(result), 768))
# for i in range(len(result)):
#     im = skimage.img_as_float(io.imread(result[i]))
#     x = calcula_histograma_cor(im)
#     X_hist[i,:] = x

# X_har = np.zeros((len(result), 10))
# for i in range(len(result)):
# 	im = skimage.img_as_ubyte(color.rgb2gray(io.imread(result[i])))
# 	x = calcula_haralick_features(im)
# 	X_har[i,:] = x


def read_matrix(PATH):
	X = np.genfromtxt(PATH, delimiter=';')
	return X

X_lbp = read_matrix("extracted_features/x_lbp.txt")
X_hist = read_matrix("extracted_features/x_hist.txt")
X_har = read_matrix("extracted_features/x_har.txt")

from sklearn.model_selection import train_test_split

def treinar_knn(X, k):
	X_train, X_test, k_train, k_test = train_test_split(X_hist, k, test_size=0.5)
	knn = KNeighborsClassifier(n_neighbors=3)
	knn.fit(X_train, k_train)
	return knn

knn_lbp = treinar_knn(X_lbp, k)
knn_hist = treinar_knn(X_hist, k)
knn_har = treinar_knn(X_har, k)

from sklearn.externals import joblib

joblib.dump(knn_lbp, 'knn_lbp.joblib.pkl')
joblib.dump(knn_hist, 'knn_hist.joblib.pkl')
joblib.dump(knn_har, 'knn_har.joblib.pkl')


#para ler
# knn_lbp = joblib.load('knn_lbp.joblib.pkl')
# knn_hist = joblib.load('knn_hist.joblib.pkl')
# knn_har = joblib.load('knn_har.joblib.pkl')