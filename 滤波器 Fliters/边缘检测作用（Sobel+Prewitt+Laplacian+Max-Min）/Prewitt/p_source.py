import cv2
import numpy as np

# gray scale
def BGR2GRAY(img):
    b = img[:,:,0].copy()
    g = img[:,:,1].copy()
    r = img[:,:,2].copy()

    out = 0.2126*r + 0.7152*g + 0.0722
    out = out.astype(np.uint8)

    return out
# prewitt filter
def prewitt_filter(img, K_size=3):
    H, W = img.shape

    # zero padding
    pad = K_size // 2
    out = np.zeros((H+pad*2, W+pad*2), dtype=np.float)
    out[pad:pad+H, pad:pad+W] = img.copy().astype(np.float)

    tmp = out.copy()
    out_v = out.copy()
    out_h = out.copy()

    # kernel
    Kv = [
        [-1., -1., -1.],
        [ 0.,  0.,  0.],
        [ 1.,  1.,  1.]
    ]
    Kh = [
        [-1., 0., 1.],
        [-1., 0., 1.],
        [-1., 0., 1.]
    ]

    # filtering
    for y in range(H):
        for x in range(W):
            out_v[y+pad, x+pad] = np.sum(Kv * tmp[y: y+K_size, x: x+K_size])
            out_h[y+pad, x+pad] = np.sum(Kh * tmp[y: y+K_size, x: x+K_size])

    out_v = np.clip(out_v, 0, 255)
    out_h = np.clip(out_h, 0, 255)

    out_v = out_v[pad:pad+H, pad:pad+W].astype(np.float)
    out_h = out_h[pad:pad+H, pad:pad+W].astype(np.float)

    return out_v, out_h

# read image
img = cv2.imread("Images/house.jpg").astype(np.float)

# gray scale
gray = BGR2GRAY(img)

# prewitt filter
out_v, out_h = prewitt_filter(gray, K_size=3)

# save and show
cv2.imwrite("out_16_1.jpg", out_v)
cv2.imshow("result_v", out_v)
while cv2.waitKey(100) != 27:
    if cv2.getWindowProperty("result_v", cv2.WND_PROP_VISIBLE) <= 0:
        break
cv2.destroyWindow("result_v")

cv2.imwrite("out_16_2.jpg", out_h)
cv2.imshow("result_h", out_h)
while cv2.waitKey(100) != 27:
    if cv2.getWindowProperty("result_h", cv2.WND_PROP_VISIBLE) <= 0:
        break
cv2.destroyWindow("result_h")
cv2.destroyAllWindows()