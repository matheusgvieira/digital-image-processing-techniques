import cv2 as cv;
import numpy as np;
import matplotlib.pyplot as plt;
import scipy.ndimage as ndimage
import collections


img = cv.imread('../images/granulometry/inputs/coin.jpg', 0);
ret, thresh = cv.threshold(img, 120, 255, cv.THRESH_BINARY_INV); 
imFillHole = ndimage.binary_fill_holes(thresh).astype(np.uint8);
imFillHole = cv.normalize(imFillHole, None, -1, 1, cv.NORM_MINMAX, cv.CV_8U);

kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
dilation = cv.dilate(imFillHole,kernel,iterations = 1)

# plt.figure()
# # plt.axis('off')
# plt.imshow(dilation, cmap=plt.cm.gray) 
# # plt.title("Image Original")
# plt.show()

num_labels, labels = cv.connectedComponents(dilation)
print(f'Número de componentes rotulados = {num_labels - 1}')

# plt.figure()
# # plt.axis('off')
# plt.imshow(labels, cmap="jet") 
# # plt.title("Image Original")
# plt.show()

aux = np.asarray(labels).reshape(-1)

# Find pixel values and quantity of an image
counter=collections.Counter(aux)
counterKeys = list(counter.keys());
counterValues = list(counter.values());

print(f'counter = {counter}')
print(f'keys = {counterKeys}')
print(f'values = {counterValues}')

porcentagem = [round((x*100)/sum(counterValues),2) for x in counterValues];

print(f'porcentagem = {porcentagem}')

# Print images
fig, axes = plt.subplots(3, 2,figsize=(10, 10));
ax = axes.ravel()

ax[0].imshow(img, cmap=plt.cm.gray);
ax[0].set_title('Image Original');
ax[1].imshow(thresh, cmap=plt.cm.gray)
ax[1].set_title('Binary Image Inverted');
ax[2].imshow(imFillHole, cmap=plt.cm.gray)
ax[2].set_title('Binary Image Normalized');
ax[3].imshow(dilation, cmap=plt.cm.gray)
ax[3].set_title('Binary Image After Dilation');
ax[4].imshow(labels, cmap="jet")
ax[4].set_title('Labeled Image ');

for a in ax:
    a.axis('off')
fig.tight_layout()
plt.show() 

    


