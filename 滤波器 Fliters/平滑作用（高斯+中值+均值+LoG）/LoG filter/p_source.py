import cv2
import numpy as np

# gray scale
def BGR2GRAY(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    out = 0.2126 * r + 0.7122 * g + 0.0722 * b
    out = out.astype(np.uint8)

    return out

# LoG filter
def LoG_filter(img, K_size=3, sigma=3):
    H, W = img.shape

    ## zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)

    tmp = out.copy()

    ## LoG kernel
    K = np.zeros((K_size, K_size), dtype=np.float)
    for y in range(-pad, -pad + K_size):
        for x in range(-pad, -pad + K_size):
            K[y+pad, x+pad] = (x ** 2 + y ** 2 - sigma ** 2) * \
                              np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    K /= 2 * np.pi * sigma ** 6
    K /= K.sum()

    ## Filtering
    for y in range(H):
        for x in range(W):
            out[pad+y, pad+x] = np.sum(K * tmp[y:y+K_size, x:x+K_size])

    out = np.clip(out, 0, 255)
    out = out[pad:pad+H, pad:pad+W].astype(np.uint8)

    return out


# read image
img = cv2.imread("Images/house.jpg")

# gray scale
gray = BGR2GRAY(img)

# LoG filter
out = LoG_filter(gray, K_size=3, sigma=5)

# show and save
cv2.imwrite("out_19.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()