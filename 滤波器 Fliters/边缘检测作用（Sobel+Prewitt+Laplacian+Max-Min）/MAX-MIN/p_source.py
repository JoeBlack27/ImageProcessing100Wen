import cv2
import numpy as np

# gray scale
def BGR2GRAY(img):
    b = img[:,:,0].copy()
    g = img[:,:,1].copy()
    r = img[:,:,2].copy()

    # gray scale
    out = 0.2126*r + 0.7152 * g + 0.0722 * b
    out = out.astype(np.uint8)

    return out

# max-min filter
def max_min_filter(img, K_size=3):
    H, W = img.shape

    # zero padding
    pad = K_size // 2
    out = np.zeros((H + pad*2, W + pad*2), dtype=np.float)
    out[pad:pad+H, pad:pad+W] = img.copy().astype(np.float)

    tmp = out.copy()

    # filtering
    for y in range(H):
        for x in range(W):
            out[pad+y, pad+x] = np.max(tmp[y:y+K_size, x:x+K_size]) - \
                np.min(tmp[y:y+K_size, x:x+K_size])

    out = out[pad:pad+H, pad:pad+W].astype(np.uint8)

    return out

# read image
img = cv2.imread("Images/house.jpg")

# gray scale
out = BGR2GRAY(img)

# max min filter
out = max_min_filter(out)

# save result
cv2.imwrite("out_13.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()