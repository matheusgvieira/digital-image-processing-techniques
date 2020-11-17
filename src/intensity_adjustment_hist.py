import cv2 as cv;
import numpy as np;
import matplotlib.pyplot as plt;
import collections;


# Function for to find values nearests in array 
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    # return array[idx]
    return idx;

# Function for to find indexs of repeated values 
def numberRepeated(s):
    oc_set = set() 
    index = [] 
    for idx, val in enumerate(s): 
        if val not in oc_set: 
            oc_set.add(val)          
        else: 
            index.append(idx)
    return index;


# Read and show image
img= cv.imread('../images/intensity_adjustment/inputs/eye.jpg', 0);
# img= cv.imread('../images/intensity_adjustment/inputs/woman.jpg', 0);

w, h = img.shape;


# Matrix to vector
aux = np.asarray(img).reshape(-1)

# Find pixel values and quantity of an image
counter=collections.Counter(aux)

# Get values (nk) 
nk = list(counter.values()); # number of pixels with rk intensity
rk = list(counter.keys()); # intensity level of a pixel

# hist() or imhist()
fig = plt.figure(figsize = (10, 7)) 
plt.bar(rk, nk, color ='blue', width = 0.5) 
plt.xlim([0, 255]) 
plt.title("Histograma da Imagem woman.jpg")
plt.xlabel("rk");
plt.ylabel("h(rk) = nk")
plt.show();

# p(r_k) = n_k/64
nk = np.asarray(nk);
p_rk = nk/(w*h);

plt.stem(rk, p_rk);
plt.xlim([0, 255]) 
plt.title("Histograma da Imagem woman.jpg")
plt.xlabel("rk");
plt.ylabel("p(rk) = nk/MN")
plt.show();

# Histograma Equalizado
sk = np.zeros(len(p_rk));
for i in range(len(p_rk)-1, -1, -1):   
    if(i == 0):
        sk[len(p_rk)-1-i] = round(np.sum(p_rk[:]*255))
    else:
        sk[len(p_rk)-1-i] = round(np.sum(p_rk[:-i]*255))

s = sk;
index = numberRepeated(sk);

for i in range(len(index)):    
    index = numberRepeated(sk);    
    p_rk[index[0]-1] = p_rk[index[0]-1] + p_rk[index[0]];
    nk[index[0]-1] = nk[index[0]-1] + nk[index[0]];
    p_rk = np.delete(p_rk, index[0])
    nk = np.delete(nk, index[0])
    sk = np.delete(sk, index[0])

plt.stem(sk, p_rk);
plt.xlim([0, 270]) 
plt.title("Histograma Equalizado da Imagem woman.jpg")
plt.xlabel("sk");
plt.ylabel("p(sk)")
plt.show();

new_img = np.zeros(img.shape);
w, h = new_img.shape;
for i in range(w):
    for j in range(h):
      new_img[i][j] = s[img[i][j]]
      
img = cv.normalize(img, None, -1, 255, cv.NORM_MINMAX, cv.CV_8U);
new_img = cv.normalize(new_img, None, -1, 255, cv.NORM_MINMAX, cv.CV_8U);

cv.imwrite('../images/intensity_adjustment/woman_eq.png',new_img)

fig, axes = plt.subplots( nrows=1, 
                          ncols=2,
                          sharex=True, 
                          sharey=True, 
                          figsize=(10, 10))
ax = axes.ravel()

ax[0].imshow(img, cmap=plt.cm.gray);
ax[0].set_title('Image Original');
ax[1].imshow(new_img, cmap=plt.cm.gray)
ax[1].set_title('Image Equalizada');

for a in ax:
    a.axis('off')
fig.tight_layout()
plt.show() 






