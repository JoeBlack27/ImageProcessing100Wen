import cv2
import numpy as np

# average filter
def mean_filter(img, K_size=3):

    #
    H, W, C = img.shape


    # zero padding
    pad = K_size // 2
    out = np.zeros((H + pad*2, W + pad*2, C), dtype=np.float)
    out[pad:pad+H, pad:pad+W] = img.copy().astype(np.float)

    tmp = out.copy()

    # filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad+y, pad+x, c] = np.mean(\
                    tmp[y:y+K_size, x:x+K_size, c])

    out = out[pad:pad+H, pad:pad+W].astype(np.uint8)

    return out

# read image
img = cv2.imread("Images/house.jpg")

# average filter
img_avg = mean_filter(img, K_size=3)

# save result
cv2.imwrite("out_11.jpg", img_avg)
cv2.imshow("result",img_avg)
cv2.waitKey(0)
cv2.destroyAllWindows()