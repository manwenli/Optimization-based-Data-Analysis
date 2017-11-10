from mnist_tools import *
from plot_tools import *
import numpy as np
import scipy.ndimage
"""
Given an image img perform the following steps:
1) Convolve with the edge detection kernel
2) Let M denote the maximum value over all
values in the resulting convolved image.
3) Threshold the resulting convolution so that all values in the
image smaller than .25*M are set to 0, and all values larger than 
.25*M are set to 1.
4) Return the resulting thresholded convolved image.
"""
def edge_detect(img) :
    x = np.matrix([[-1/8,-1/8,-1/8],[-1/8,1,-1/8],[-1/8,-1/8,-1/8]])
    result = scipy.ndimage.convolve(img,x)
    M = np.max(img)
    result[result<=0.25*np.max(result)]=0
    result[result>0.25*np.max(result)]=1
    
    return result
    
def main() :
    test,testLabels = load_test_data("mnist_all.mat")
    imgs = []
    for im in test :
        img = im.reshape((28,28))
        imgs.extend([img,edge_detect(img)])
    plot_image_grid(imgs,"MNistConvolve",bycol=True,
                    row_titles=['True','Convolved'])

if __name__ == "__main__" :
    main()
