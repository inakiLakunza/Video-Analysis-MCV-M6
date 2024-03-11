
import numpy as np



def Euclidean_match(matrix1, matrix2):
    np.sum(matrix1.astype(np.float32)-matrix2.astype(np.float32))**2