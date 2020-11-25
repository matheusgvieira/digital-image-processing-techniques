import cv2 as cv;
import numpy as np;
import matplotlib.pyplot as plt;

img = cv.imread('../images/highlight/inputs/dogs.jpg', 0);
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# img = cv.normalize(img, None, -1, 255, cv.NORM_MINMAX, cv.CV_8U);

# plt.figure()
# # plt.axis('off')
# plt.imshow(img, cmap=plt.cm.gray) 
# # plt.title("Image Original")
# plt.show()


def highlight(img):  
    
    # Calc DFT
    Img = np.fft.fftshift(np.fft.fft2(img));
    Img_m = np.absolute(Img);
    
    # plt.imshow(np.log(Img_m), cmap = 'gray')
    # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    # plt.show()
    
    # Create H(u,v) and D(u,v)
    D = np.copy(img).astype(np.float64)
    Hhp = np.copy(img).astype(np.float64)    
    
    # Set D(u,v)
    M, N = img.shape; # Get with and height of image
    u, v = np.meshgrid(range(M), range(N), sparse=False, indexing='ij');
    D = np.sqrt((u-M/2)**2+(v-N/2)**2)
    
    
    # Define params   
    Do = 5;
    # n = 20;
     
    # Set Hhp(u,v)
    # Hlp = np.exp((-(D**2))/(2*Do**2));
    # Hhp = np.ones(img.shape) - Hlp;
    # Hhp = 1 - np.exp((-(D**2))/(2*Do**2));
    Hhp = -4*(np.pi**2)*(D**2);
    # Hhp = 1/(1 + (D/Do)**(2*n));
      
    plt.plot(Hhp);
    plt.title("Filtro no domínio da frequência.")
    plt.show();
    
    plt.plot(np.log(np.absolute(Hhp)));
    plt.title("Aspectro do filtro no domínio da frequência.")
    plt.show();
    
    # laplacian = cv.Laplacian(img,cv.CV_64F)
    # laplacian1 = laplacian/laplacian.max()
    # output = img - laplacian1
    # Output = np.fft.fftshift(np.fft.fft2(output));
    # Output_m = np.absolute(Output);
    
    # Calc DFT inv
    G = Img*Hhp;
    f2 = np.fft.ifft2(np.fft.ifftshift(G));
    gp = np.real(f2)*((-1)**(M+N));
    out = img - gp;
    Out = np.fft.fftshift(np.fft.fft2(out));
    Out_m = np.absolute(Out);
    img_filtered = cv.normalize(out, None, -1, 255, cv.NORM_MINMAX, cv.CV_8U)
    
    # Print images
    fig, axes = plt.subplots(2, 2,figsize=(10, 10));
    ax = axes.ravel()
    
    ax[0].imshow(img, cmap=plt.cm.gray);
    ax[0].set_title('Image Original');
    ax[1].imshow(img - img_filtered, cmap=plt.cm.gray)
    ax[1].set_title('Image Filtrada');
    ax[2].imshow(np.log(Img_m), cmap=plt.cm.gray)
    ax[2].set_title('Magnitude Spectrum');
    ax[3].imshow(np.log(Out_m), cmap=plt.cm.gray)
    ax[3].set_title('Magnitude Spectrum Output');
    
    for a in ax:
        a.axis('off')
    fig.tight_layout()
    plt.show() 


highlight(img);


# laplacian = cv.Laplacian(img,cv.CV_64F)
# # laplacian = cv.normalize(laplacian, None, -1, 255, cv.NORM_MINMAX, cv.CV_8U);
# # But this tends to localize the edge towards the brighter side.
# laplacian1 = laplacian/laplacian.max()

# fig, axes = plt.subplots(1, 2,figsize=(10, 10));
# ax = axes.ravel()
    
# ax[0].imshow(img, cmap=plt.cm.gray);
# ax[0].set_title('Image Original');
# ax[1].imshow(img - laplacian1, cmap=plt.cm.gray)
# ax[1].set_title('Image Filtrada');

# for a in ax:
#     a.axis('off')
# fig.tight_layout()
# plt.show() 