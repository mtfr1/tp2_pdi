# TP2 - Processamento Digital de Imagens

## Classificação de Imagens

### Extração de Features

- Local Binary Pattern (LBP)
  - LBP uniforme, 10 features por imagem.
  - taxa de acerto de aproximadamente 78%.
  - utilizando KNN com k = 3.
- Haralick features
  - offset (1,2), ângulos (0º, 90º)
  - 5 métricas (contraste, dissimilaridade, homogeneidade, ASM e energia).
  - taxa de acerto de aproximadamente 60%.
  - utilizando KNN com k = 3.
- Histograma de cores
  - 256 bins totalizando 768 features (3 canais de cor). 
  - taxa de acerto de aproximadamente 62%.
  - utilizando KNN com k = 3.

## Interface Web
- Rodando em PHP7.2
- Instruções de uso
	- Clone o repositório na pasta /var/www/html
	- Na pasta /var/www/html rode sudo chmod go+rwx /tp2_pdi
	- É necessário que o dataset esteja na pasta raiz (tp2_pdi)
	- Rode o script fruit_classifier.py. 

<https://scikit-image.org/docs/0.7.0/api/skimage.feature.texture.html>

<https://scikit-learn.org/stable/modules/neighbors.html>

<https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.histogram>

https://en.wikipedia.org/wiki/Local_binary_patterns

https://en.wikipedia.org/wiki/Co-occurrence_matrix
