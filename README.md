# TP2 - Processamento Digital de Imagens

## Classificação de Imagens

### Extração de Features

- Local Binary Pattern (LBP)
  - LBP uniforme, 10 features por imagem.
  - taxa de acerto de aproximadamente 78%.
  - utilizando KNN com k = 3.
- Haralick features
- Histograma de cores
  - 256 bins totalizando 768 features (3 canais de cor). 
  - taxa de acerto de aproximadamente 60%.
  - utilizando KNN com k = 3.

<https://scikit-image.org/docs/0.7.0/api/skimage.feature.texture.html>
<https://scikit-learn.org/stable/modules/neighbors.html>
<https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.histogram>