import cv2 as cv;
import numpy as np;
import matplotlib.pyplot as plt;
import scipy.ndimage as ndimage
import collections


img = cv.imread('../images/granulometry/inputs/coin.jpg', 0);
ret, thresh = cv.threshold(img, 120, 255, cv.THRESH_BINARY_INV); 
imFillHole = ndimage.binary_fill_holes(thresh).astype(np.uint8);
imFillHole = cv.normalize(imFillHole, None, -1, 1, cv.NORM_MINMAX, cv.CV_8U);
# imFillHole = cv.normalize(img, None, -1, 1, cv.NORM_MINMAX, cv.CV_8U);

# kernel = np.ones((50,50),np.uint8)
kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))

# cv.imshow('erosion', erosion);

dilation = cv.dilate(imFillHole,kernel,iterations = 1)

plt.figure()
# plt.axis('off')
plt.imshow(dilation, cmap=plt.cm.gray) 
# plt.title("Image Original")
plt.show()

print(dilation);

num_labels, labels = cv.connectedComponents(dilation)
print(num_labels - 1)

plt.figure()
# plt.axis('off')
plt.imshow(labels, cmap=plt.cm.gray) 
# plt.title("Image Original")
plt.show()


def adj8(img, x, y):
    arr = np.zeros(8);
    arrIndexX = np.zeros(8);
    arrIndexY = np.zeros(8);
    k = 8;
    cx = 0;
    cy = 0;
    cyy = 0;
    negX = True;
    overX = True;
    negY = True;
    overY = True;
    w, h = img.shape;
    
    if(x == 0):
        negX = False;
        k -= 1;
        cx += 1;
    if(x == w-1):
        overX = False;
        k -= 1;
    if(y == 0):
        negY = False;
        k -= 1;
        cy += 1;
    if(y == h-1):
        overY = False;
        k -= 1;
        cyy = 1;
        
    for i in range(8):
        if(i == 0 and negY):
            arr[i] = img[x][y-1];
            arrIndexX[i] = x;
            arrIndexY[i] = y-1;
        if(i == (1 - cy) and negX):
            arr[i] = img[x-1][y-1];
            arrIndexX[i] = x-1;
            arrIndexY[i] = y-1;
        if(i == (2 - cy) and negX):
            arr[i] = img[x-1][y];
            arrIndexX[i] = x-1;
            arrIndexY[i] = y;
        if(i == (3 - cx - cy) and overY):
            arr[i] = img[x-1][y+1];
            arrIndexX[i] = x-1;
            arrIndexY[i] = y+1;
        if(i == (4 - cx - cy) and overY):
            arr[i] = img[x][y+1];
            arrIndexX[i] = x;
            arrIndexY[i] = y+1;
        if(i == (5 - cx - cy) and overY):
            arr[i] = img[x+1][y+1];
            arrIndexX[i] = x+1;
            arrIndexY[i] = y+1;
        if(i == (6 - cx - cy - cyy) and overX):
            arr[i] = img[x+1][y];
            arrIndexX[i] = x+1;
            arrIndexY[i] = y;
        if(i == (7 - cx - cy - cyy) and overX):
            arr[i] = img[x+1][y-1];
            arrIndexX[i] = x+1;
            arrIndexY[i] = y-1;
    return arr[:k], arrIndexX[:k], arrIndexY[:k];



def counterMancha(img2):
    w, h = img2.shape;    
    jmg = np.zeros(img2.shape);
    label = 1;
    Q = 0;
    q = 0;
    
    for j in range(h):
        for i in range(w):
            if(img2[i][j] != 0 and jmg[i][j] == 0):
                ad, x, y = adj8(jmg, i, j);
                if(1 in ad):
                    label = label - 1;
                jmg[i][j] = label;
                Q = img2[i][j];
                q = (i,j);
                
                while(Q != 0):
                    
                    Q = 0;
                    ad, x, y = adj8(img2, q[0], q[1]);
                    
                    for k in range(len(ad)): 
                        if(img2[q[0]][q[1]] == ad[k]): 
                            if(jmg[int(x[k])][int(y[k])] == 0 ):
                                jmg[int(x[k])][int(y[k])] = jmg[i][j];
                                q = (int(x[k]),int(y[k])); 
                                Q = img2[int(x[k])][int(y[k])];
                                break;
                    
                label += 1;
               
                
    return jmg;


imgLabeled = counterMancha(dilation);
print('-------- FINAL --------')
print(imgLabeled)

aux = np.asarray(imgLabeled).reshape(-1)

# Find pixel values and quantity of an image
counter=collections.Counter(aux)

print(f'labels = {list(counter.keys())}')
print(f'labels = {len(list(counter.keys())) - 1}')

# plt.figure()
# plt.axis("off")
# plt.imshow(imFillHole, cmap=plt.cm.gray) 
# plt.title(f'Vetor das chaves = {list(counter.keys())} \n labels = {len(list(counter.keys())) - 1}')
# plt.show()