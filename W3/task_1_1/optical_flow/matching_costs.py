
import numpy as np
import cv2


def MSE(matrix1, matrix2):
    return ((matrix1.astype(np.float32)-matrix2.astype(np.float32))**2).mean()


def MAE(matrix1, matrix2):
    return ((matrix1.astype(np.float32)-matrix2.astype(np.float32))).mean()



"""
See "Learning OpenCV 3: Computer Vision in C++ with the OpenCV Library" By Adrian Kaehler, Gary Bradski

https://books.google.com.au/books?id=SKy3DQAAQBAJ&lpg=PT607&ots=XGg5zrJXPp&dq=TM_CCOEFF&pg=PT606#v=onepage&q=TM_CCOEFF&f=false

According to the book:

TM_CCORR = Cross correlation

TM_CCOEFF = Correlation coefficient

FWIW: The −1/(w⋅h)⋅∑x″,y″T(x″,y″) in the TM_CCOEFF method is simply used to a) make the template and image zero mean and b) make the dark parts of the image negative values and the bright parts of the image positive values.

This means that when bright parts of the template and image overlap you'll get a positive value in the dot product, as well as when dark parts overlap with dark parts (-ve value x -ve value gives +ve value). That means you get a +ve score for both bright parts matching and dark parts matching.

When you have dark on template (-ve) and bright on image (+ve) you get a -ve value. And when you have bright on template (+ve) and dark on image (-ve) you also get a -ve value. This means you get a negative score on mismatches.

On the other hand if you don't have the −1/(w⋅h)⋅∑x″,y″T(x″,y″) term, i.e. in TM_CCORR method, then you don't get any penalty when there are mismatches between the template and the image. Effectively this method is measuring where you get the brightest set of pixels in the image that are the same shape to the template. (That's why the logo, the soccer ball and area above Messi's leg have high intensity in the matching result).

"""

def NCCORR():
    return cv2.TM_CCORR_NORMED
    

def NCCOEFF():
    return cv2.TM_CCOEFF_NORMED