import cv2
import numpy as np

# gray scale
def RBG2GRAY(img):
    b = img[:,:,0].copy()
    g = img[:,:,1].copy()
    r = img[:,:,2].copy()

    out = 0.2126 * r + 0.7152 * g + 0.0722 * b
    out = out.astype(np.uint8)

    return out

# laplacian filter
def laplacian_filter(img, K_size):
    H, W = img.shape

    ## zero padding
    pad = K_size // 2
    out = np.zeros((H+pad*2, W+pad*2), dtype=np.float)
    out[pad:pad+H, pad:pad+W] = img.copy().astype(np.float)

    tmp = out.copy()

    ## Kernel
    K = [
        [0., 1., 0.],
        [1., -4., 1.],
        [0., 1., 0.]
    ]

    ## filtering
    for y in range(H):
        for x in range(W):
            out[y+pad, x+pad] = np.sum(K * tmp[y:y+K_size, x:x+K_size])

    out = np.clip(out, 0, 255)
    out = out[pad: pad+H, pad: pad+W].astype(np.uint8)

    return out

# read image
img = cv2.imread("Images/house.jpg").astype(np.float)

# gray scale
gray = RBG2GRAY(img)

# laplacian filter
out = laplacian_filter(gray, K_size=3)

# save and show
cv2.imwrite("out_17.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()