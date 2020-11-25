# Página 211 - Exemplo e equação
import numpy as np;
import cv2 as cv;
from matplotlib import pyplot as plt

def Dk(uk, vk, u, v):
    return np.sqrt((u - M/2 - uk)**2 + (v - N/2 - vk)**2)

def D_k(uk, vk, u, v):
    return np.sqrt((u - M/2 + uk)**2 + (v - N/2 + vk)**2)

img2 = cv.imread('../images/filter_noise/inputs/men.jpg', 0);

M, N = img2.shape;
n = 4;
Do = 10;

# Calc DFT
img = np.fft.fft2(img2)
fshift = np.fft.fftshift(img)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

# u, v = np.meshgrid(np.arange(0,N), np.arange(0,M))
u, v = np.meshgrid(range(M), range(N), sparse=False, indexing='ij');

# uk = np.array([58,120,145]);
# vk = np.array([0,0,0]);

# uk = np.array([25]);
# vk = np.array([40]);

# H = np.ones(img2.shape);

# for i in range(len(uk)):
#     H *= (1/(1 + (Do/Dk(uk[i],vk[i],u,v))**(2*n)))*(1/(1 + (Do/D_k(uk[i],vk[i],u,v))**(2*n)));
    
# fig, axes = plt.subplots( nrows=1, 
#                           ncols=2,
#                           sharex=True, 
#                           sharey=True, 
#                           figsize=(10, 10))
# ax = axes.ravel()

# ax[0].imshow(magnitude_spectrum, cmap=plt.cm.gray);
# ax[0].set_title('Espectro da Imagem');
# ax[1].imshow(H*magnitude_spectrum, cmap=plt.cm.gray)
# ax[1].set_title('Espectro do Filtro');

# for a in ax:
#     a.axis('off')
# fig.tight_layout()
# plt.show()   

# # Calc DFT inv
# img_back = np.real(np.fft.ifft2(np.fft.ifftshift(H*fshift)));

# # np.uint8 + stagger
# img_filtered = cv.normalize(img_back, None, -1, 255, cv.NORM_MINMAX, cv.CV_8U);

# # Print images
# # cv.imwrite('./images/filtered1.png',img_filtered)
# cv.imshow('Filtered',img_filtered)
# # cv.imwrite('../images/filter_noise/outputs/indian.png', img_filtered)
# cv.imshow('Normal', img2)

# # Wait for can be see
# cv.waitKey(0)
# cv.destroyAllWindows()