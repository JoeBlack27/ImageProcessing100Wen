import cv2
import numpy as np

# gaussian filter
def gaussian_filter(img, k_size=3, sigma=1.3):
    if len(img.shape) == 3:
        H, W, C =  img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape

    ## zero padding
    pad = k_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out[pad:pad+H, pad:pad+W] = img.copy().astype(np.float)

    ## prepare kernel
    K = np.zeros((k_size, k_size), dtype=np.float)
    for x in range(-pad, -pad + k_size):
        for y in range(-pad, -pad + k_size):
            K[y+pad, x+pad] = np.exp(-(x**2 + y**2) / (2*(sigma**2)))
    K /= (2 * np.pi * sigma**2)
    K /= K.sum()

    tmp = out.copy()

    ## filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad+y, pad+x, c] = \
                    np.sum(K * tmp[y:y+k_size, x:x+k_size, c])

    out = out.clip(0, 255)
    out = out[pad:pad+H, pad:pad+W].astype(np.uint8)

    return out

# read image
img = cv2.imread("Images/house.jpg")

# gaussian filtering
img_g = gaussian_filter(img,k_size=3,sigma=1.7)

# save result
cv2.imwrite("out_9.jpg", img_g)
cv2.imshow("result", img_g)
cv2.waitKey(0)
cv2.destroyAllWindows()
