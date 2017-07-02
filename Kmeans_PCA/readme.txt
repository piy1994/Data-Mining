Python Version : Python 2.7.12 

Libraries used :

numpy 
pandas
from scipy.spatial.distance import cdist
math
sys
random
SCIPY

1.The main file is PCA_Kmeans_main.py
2.It needs 2 inputs to run : "digits-embedding file path" "K" where "digits-embedding file path" is the Tsne reduced with 2 dimensiona and K is the required number of clusters
3.The raw file is "digits-raw.csv"
4.I have also included separate functions for Kmeans and PCA.
5.PCA function requires the input file and number of dimensions to be reduced
6.For evaluation, I included my own implementation of Silhoutte coefficient, Normalized Mutual Information gain(NMI) and Weighted sum of squares of distances(wssd)
