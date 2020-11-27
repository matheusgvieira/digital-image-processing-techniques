import numpy as np;
import cv2 as cv;
from matplotlib import pyplot as plt


def magnitude_spectrum(img):
    # np.uint8 to np.float32
    img_float32 = np.copy(img).astype(np.float32);
    img_float32 = np.where(img_float32 == 0, np.exp(0), img_float32) 
    
    # ln
    img_log = np.log(img_float32);
    # Calc DFT
    img_dft = np.fft.fft2(img_log)
    fshift = np.fft.fftshift(img_dft)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    
    return magnitude_spectrum;

def filtragem_homomorfica(img):
    # Get with and height of image
    m, n = img.shape;
    
    # Define P e Q
    P = np.floor(2*m)
    Q = np.floor(2*n)
    
    # np.uint8 to np.float32
    img_float32 = np.copy(img).astype(np.float32);
    img_float32 = np.where(img_float32 == 0, np.exp(0), img_float32) 
    
    # ln
    img_log = np.log(img_float32);
    
    # Calc DFT
    img_dft = np.fft.fft2(img_log)
    fshift = np.fft.fftshift(img_dft)
    
    # Create H(u,v) and D(u,v)
    D = np.copy(img).astype(np.float64)
    H = np.copy(img).astype(np.float64)    
    
    # Set D(u,v)
    u, v = np.meshgrid(range(m), range(n), sparse=False, indexing='ij');
    D = ((u-P/2)**2+(v-Q/2)**2)**(0.5);
    
    
    # Define params      
    gamaL = 0.5;
    gamaH = 2;
    c = 1;
    Do = 80;
     
    # Set H(u,v)
    H = (gamaH - gamaL)*(1-np.exp(-c*((D**2)/(Do**2)))) + gamaL;   
    
    plt.plot(20*np.log(np.abs(H)));
    plt.title("Filtro no domínio da frequência.")
    plt.show();
    
    # Calc DFT inv
    img_back = np.fft.ifft2(H*img_dft);
    
    # e
    g = np.real(np.exp(img_back));
    
    # np.uint8 + stagger
    img_filtered = cv.normalize(g, None, -1, 255, cv.NORM_MINMAX, cv.CV_8U);
    return(img_filtered)
    
    
img = cv.imread('../images/highlight/inputs/statue.jpeg',0)
img_hom = filtragem_homomorfica(img);

# Print images
fig, axes = plt.subplots(2, 2,figsize=(10, 10));
ax = axes.ravel()

ax[0].imshow(img, cmap=plt.cm.gray);
ax[0].set_title('(a) Image Original');
ax[1].imshow(img_hom, cmap=plt.cm.gray)
ax[1].set_title('(b) Image Filtrada');
ax[2].imshow(magnitude_spectrum(img), cmap=plt.cm.gray)
ax[2].set_title('(c) Aspectro da Imagem Original');
ax[3].imshow(magnitude_spectrum(img_hom), cmap=plt.cm.gray)
ax[3].set_title('(d) Aspectro da Imagem Filtrada');

for a in ax:
    a.axis('off')
fig.tight_layout()
plt.show() 

# cv.imwrite('../images/highlight/inputs/statue2.png',img)
# cv.imwrite('../images/highlight/outputs/statue2.png',img_hom)
   



        








        




        








        

