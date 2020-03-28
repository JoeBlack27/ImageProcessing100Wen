import cv2
import numpy as np

# Gray scale
def BGR2GRAY(img):
    b = img[..., 0].copy()
    g = img[..., 1].copy()
    r = img[..., 2].copy()

    out = 0.2126 * r + 0.7152 * g + 0.0722 * b
    out = out.astype(np.uint8)

    return out

# OTSU binarization
def otsu_binarization(img, th=128):
    H, W = img.shape
    out = img.copy()

    max_departDegree = 0
    thres = 0

    ## determine threshold
    for _t in range(1, 255):
        v0 = out[np.where(out < _t)]
        m0 = np.mean(v0) if len(v0) > 0 else 0.
        w0 = len(v0) / (H * W)

        v1 = out[np.where(out >= _t)]
        m1 = np.mean(v1) if len(v1) > 0 else 0.
        w1 = len(v1) / (H * W)

        departDegree = w0 * w1 * ((m0 - m1) ** 2)
        if departDegree > max_departDegree:
            max_departDegree = departDegree
            thres = _t

    # Binarization
    print("threshold >> ", thres)

    th = thres
    out[out < th] = 0
    out[out >= th] = 255

    return out


def morphology_dilate(img, Erode_time=1):
    H, W = img.shape
    out = img.copy()

    # kernel
    K = np.array((
        (0, 1, 0),
        (1, 0, 1),
        (0, 1, 0),
    ), dtype=np.float)

    # each erode
    for i in range(Erode_time):
        tmp = np.pad(out, (1, 1), 'edge')
        # erode
        for y in range(1, H+1):
            for x in range(1, W+1):
                if np.sum(K * tmp[y-1:y+2, x-1:x+2]) < 255*4:
                    out[y-1, x-1] = 0

    return out


# Read image
img = cv2.imread("Images/house.jpg").astype(np.float)

# Gray scale
gray = BGR2GRAY(img)

# Otsu
binary = otsu_binarization(gray)

# Morphology - dilate
out = morphology_dilate(binary, Erode_time=1)

# Save and show
cv2.imshow("result", out)
cv2.imwrite("out_48.jpg", out)
cv2.waitKey(0)
cv2.destroyAllWindows()