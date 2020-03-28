import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Images/house_c.jpg')
H = np.float32([[1,0,100],[0,1,50]])   # define an translate matrix
rows,cols = img.shape[:2]
res = cv2.warpAffine(img, H, (rows,cols)) # input image, matrix, size after affine
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(res)
