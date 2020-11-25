import matplotlib.pyplot as plt
import numpy as np
import cv2

def my_histogram(image, plot = True, amax = 256, norm = False):
    if(len(image.shape) > 2):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get number of lines and columns
    qntI, qntJ = image.shape
    
    # Creating Histogram manually
    histogram = np.zeros(amax)
    color = 0
    for i in range(qntI):
        for j in range(qntJ):
            color = image[i][j]
            # print(color)
            histogram[color] += 1
            
    if(norm):
        histogram = (histogram - np.amin(histogram)) /  (np.amax(histogram) - np.amin(histogram))
    
    if(plot):
        plt.figure()
        plt.stem(histogram, use_line_collection = True)
        plt.title('Original Image Histogram $p_r(r)$')
        plt.savefig('hist_original_fig')
        plt.show()
    return histogram

def cdf_pdf(pdf, plot = True):
    cdf = np.zeros(len(pdf))
    for h in range(len(pdf)):
        cdf[h] = np.sum(pdf[0: h + 1])
    if(plot):
        plt.figure()
        plt.stem(pdf, use_line_collection = True)
        plt.title('Probability Distribuition function (PDF) $p_z(z)$')
        plt.show()

        plt.figure()
        plt.stem(cdf, use_line_collection = True)
        plt.plot(cdf, 'k')
        plt.title('Cumulative Distribution Function (CDF) $G(z)$')
        plt.show()
    return cdf

def cdf_2D(img, plot = True, amax = 256, norm = False):
    cdf = np.zeros(amax)
    # sdf = np.zeros(amax)
    if(len(img.shape) > 2):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Get number of lines and columns
    qntI, qntJ = img.shape
    # Number of pixels
    qnt_pixels = qntI * qntJ
   
    histogram = my_histogram(img, amax = amax, norm = norm, plot = False)
    # print(len(histogram))
    for h in range(amax):
        cdf[h] = np.sum(histogram[0: h + 1]) / qnt_pixels 
        # sdf[h] = histogram[h] / qnt_pixels 
        
    
    if(plot):
        plt.figure()
        plt.stem(cdf, use_line_collection = True)
        plt.plot(cdf, 'k')
        plt.title('Cumulative Distribution Function (CDF) $G(s)$')
        plt.show()
    return cdf

def inv_cdf(img, required_pdf = None, amax = 256):
    if(len(img.shape) > 2):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x, y = img.shape
    new_img = np.copy(img)

    if(required_pdf is None):
        required_pdf = np.divide(np.ones(amax), amax)
    s = np.round(np.multiply((amax - 1), cdf_2D(img, plot = False)))
    G = np.round(np.multiply((amax - 1), cdf_pdf(required_pdf, plot = False)))

    # Example
    # original_pdf = [0.19, 0.25, 0.21, 0.16, 0.08, 0.06, 0.03, 0.02]
    # required_pdf = [0.0, 0.0, 0.0, 0.15, 0.20, 0.30, 0.20, 0.15]
    # amax = 8
    # s = np.round(np.multiply((amax - 1), cdf_pdf(original_pdf, plot = False)))
    # G = np.round(np.multiply((amax - 1), cdf_pdf(required_pdf, plot = False)))

    s = s.astype(np.uint8)
    new_z = np.zeros(amax)
    G_s = np.zeros(amax)
    diffs = []

    for k in range(amax):
        diffs = np.abs(np.subtract(G, s[k]))
        new_z[k] = np.argmin(diffs)
        G_s[s[k]] = np.argmin(diffs)

    plt.figure()
    plt.stem(G, linefmt = 'k', use_line_collection = True)
    plt.plot(s, '-or')
    plt.legend(['$s_k$', '$G(z_k)$'])
    plt.savefig('cdf_images')
    plt.show()

    plt.figure()
    markerline, stemlines, baseline = plt.stem(s, G_s[s], linefmt = 'k', markerfmt = '-oc', use_line_collection = True)
    plt.setp(baseline, color='k', linewidth=2)
    plt.setp(markerline, linewidth=3)
    plt.title('Mapping s in z')
    plt.xlabel('original values (s)')
    plt.ylabel('new values ($z = G^{-1}(s)$)')
    plt.savefig('transform_graph')
    plt.show()

    for i in range(x):
        for j in range(y):
            new_img[i][j] = new_z[img[i][j]]
    new_img = new_img.astype(np.uint8)
    return new_img

img = cv2.imread("../images/intensity_adjustment/inputs/woman.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
new_img = inv_cdf(img)

fig, ax = plt.subplots(1, 2, figsize = [15, 15])
ax[0].imshow(img, cmap='gray', vmin = 0, vmax = 255)
ax[1].imshow(new_img, cmap='gray', vmin = 0, vmax = 255)