import cv2
import numpy as np

# Affine
def affine(img_ori, a, b, c, d, tx, ty):
    H, W, C = img_ori.shape

    # temporary image
    img = np.zeros((H+2, W+2, C), dtype=np.float32)
    img[1: H+1, 1: W+1] = img_ori

    # get new image shape
    W_new = np.round(a * W).astype(np.int)
    H_new = np.round(d * H).astype(np.int)
    out = np.zeros((H_new+1, W_new+1, C), dtype=np.float32)

    # get position of new image
    y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)
    x_new = np.tile(np.arange(W_new), (H_new, 1))

    # get position of original image by affine
    adbc = a * d - b * c
    x = np.round((d * x_new - b * y_new) / adbc).astype(np.int) - tx + 1
    y = np.round((-c * x_new + a * y_new) / adbc).astype(np.int) - ty + 1

    x = np.minimum(np.maximum(x, 0), W+1).astype(np.int)
    y = np.minimum(np.maximum(y, 0), H+1).astype(np.int)

    # assign pixel to new image
    out[y_new, x_new] = img[y, x]

    out = out[:H_new, :W_new]
    out = out.astype(np.uint8)

    return out

# Read image
img = cv2.imread("Images/house_c.jpg").astype(np.float32)

# Affine
out = affine(img, a=1.3, b=0, c=0, d=0.8, tx=30, ty=-30)

# Save result
cv2.imshow("result", out)
cv2.imwrite("out_29.jpg", out)
cv2.waitKey(0)