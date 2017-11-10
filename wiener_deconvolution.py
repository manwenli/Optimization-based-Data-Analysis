import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal
from matplotlib.colors import LogNorm
from plot_tools import *
import scipy.ndimage

"""
Given an image and a filter, returns the 2d circular convolution.
Arguments
img: 2d numpy array of the image of shape (R,C)
fil: 2d numpy array of the filter of shape (R,C)
Returns a 2d numpy array of shape (R,C) containing the circular convolution.  The 
returned value must be real (so return the real part if you do a complex calculation).
"""
def convolve(img, fil) :
    
    DFT_image = np.fft.fft2(img)
    DFT_fill = np.fft.fft2(fil)
    
    DFT_conv = np.zeros((64,64))*(0.+0.j)
    
    for i in range(64):
        for j in range(64):
            DFT_conv[i,j]=DFT_image[i,j]*DFT_fill[i,j]
            
    conv = np.fft.ifft2(DFT_conv)
    conv = conv.real
    
    return conv

"""
Given a convolved image and a filter, returns the 2d circular deconvolution.
Arguments
img: 2d numpy array of the convolved image of shape (R,C)
fil: 2d numpy array of the filter of shape (R,C)
Returns a 2d numpy array of shape (R,C) containing the 2d circular deconvolution.  The 
returned value must be real (so return the real part if you do a complex calculation).
"""
def deconvolve(img, fil) :
    DFT_conv = np.fft.fft2(img)
    DFT_fil = np.fft.fft2(fil)
    DFT_fil[DFT_fil == 0] = 0.00001
    DFT_fil_dec = 1/DFT_fil
    
    DFT_img = np.zeros((64,64))*(0.+0.j)
    for i in range(64):
        for j in range(64):
            DFT_img[i,j] = DFT_fil_dec[i,j]*DFT_conv[i,j]
    deconv = np.fft.ifft2(DFT_img)
    
    return deconv.real
    
"""
Given a convolved image, a filter, a list of images, and the variance
of the added Gaussian noise, returns the Wiener deconvolution.
Arguments
img: 2d numpy array of the convolved image of shape (R,C)
fil: 2d numpy array of the filter of shape (R,C)
imgs: 3d numpy array of n images from which to extract mean and variance info; shape (n,R,C)
v: the variance of each entry of the iid additive Gaussian noise
Returns a 2d numpy array of shape (R,C) containing the wiener deconvolution.  The 
returned value must be real (so return the real part if you do a complex calculation).
"""
def wiener_deconvolve(img, fil, imgs, v) :
    #step1
   
    varZ = v*(64**2)
    #step2
    DFT_imgdb = np.zeros((imgs.shape[0],img.shape[0],img.shape[1]))*(0.+0.j)
    for i in range(imgs.shape[0]):
        DFT_imgdb[i,:,:] = np.fft.fft2(imgs[i,:,:])
        
    uX = np.zeros((img.shape[0],img.shape[1]))*(0.+0.j)
    varX = np.zeros((img.shape[0],img.shape[1]))*(0.+0.j)
    for i in range(DFT_imgdb.shape[1]):
        for j in range(DFT_imgdb.shape[2]):
            uX[i,j] = np.mean(DFT_imgdb[:,i,j])
            varX[i,j] = np.var(DFT_imgdb[:,i,j])
    
    #step3
    B = np.fft.fft2(img)
    K = np.fft.fft2(fil)
    
    #step 4
    Bc = np.zeros((img.shape[0],img.shape[1]))*(0.+0.j)
    W = np.zeros((img.shape[0],img.shape[1]))*(0.+0.j)
    Xw = np.zeros((img.shape[0],img.shape[1]))*(0.+0.j)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            Bc[i,j] = B[i,j]-K[i,j]*uX[i,j]
            W[i,j] = np.conj(K[i,j])*varX[i,j]/(np.absolute(K[i,j])*np.absolute(K[i,j])*varX[i,j]+varZ)
            Xw[i,j] = uX[i,j]+W[i,j]*Bc[i,j]
            
    #Bc = B - np.multiply(K,uX)
    #W = np.divide(np.multiply(np.conj(K),varX),(np.multiply(np.multiply(np.absolute(K),np.absolute(K)),varX)+varZ))
    #Xw = uX+np.multiply(W,Bc)
    
    #step 5
    result = np.fft.ifft2(Xw)
    return result.real
            
        
            

def main() :
    data = scipy.io.loadmat('olivettifaces.mat')['faces'].T.astype('float64')
    imgdb = np.array([im.reshape(64,64).T for im in data])
    imgs = []
    imgs2 = []
    for i in range(6,10) :        
        image = data[10*i+6,:].astype('float64')
        image = image.reshape(64,64).T
        fil = np.outer(signal.gaussian(64,1),signal.gaussian(64,1))
        sfil = np.fft.ifftshift(fil)
        conimg = convolve(image,sfil)
        deconimg = deconvolve(conimg,sfil)
        ran = np.amax(conimg)-np.amin(conimg)
        s = ran/50.0
        noise = s*np.random.randn(*image.shape)
        noisycon = conimg + noise
        noisydecon = deconvolve(noisycon,sfil)
        wienerdecon = wiener_deconvolve(noisycon,sfil,imgdb,s**2)
        imgs.extend([image,fil,conimg,deconimg])
        imgs2.extend([noisycon,noisydecon,wienerdecon])

    coltitles = ['Image','Filter','Convolve','Deconvolve']
    coltitles2 = ['NoisyConvolve','NoisyDeconvolve','WienerDeconvolve']
    plot_image_grid(imgs,'Noisefree',(64,64),len(coltitles),4,col_titles=coltitles)
    plot_image_grid(imgs2,'Noisy',(64,64),len(coltitles2),4,col_titles=coltitles2)
    plot_fft_image_grid(imgs,'NoisefreeFFT',(64,64),len(coltitles),4,col_titles=coltitles)

if __name__ == "__main__" :
    main()
