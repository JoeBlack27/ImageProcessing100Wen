import cv2
import numpy as np
import matplotlib.pyplot as plt

# Gray scale
def BGR2GRAY(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    # Gray scale
    out = 0.2126 * r + 0.7152 * g + 0.0722 * b
    out = out.astype(np.uint8)

    return out

def otsu_binarization(img, th=128):
    H, W = img.shape
    out = img.copy()

    max_departDegree = 0
    thres = 0

    # determine threshold
    for t in range(1, 255):
        v0 = out[np.where(out < t)]
        m0 = np.mean(v0) if len(v0) > 0 else 0.
        w0 = len(v0) / (H * W)

        v1 = out[np.where(out >= t)]
        m1 = np.mean(v1) if len(v1) > 0 else 0.
        w1 = len(v1) / (H * W)

        departDegree = w0 * w1 * ((m0 - m1) ** 2)
        if departDegree > max_departDegree:
            max_departDegree = departDegree
            thres = t

    # Binarization
    print("threshold >>", thres)
    th = thres
    out[out < th] = 0
    out[out >= th] = 255

    return out

# Morphology - dilate
def Morphology_Dilate(img, Dilate_time=1):
    H, W = img.shape
    out = img.copy()

    # kernel
    K = np.array((
        (0, 1, 0),
        (1, 0, 1),
        (0, 1, 0)
    ), dtype=np.int)

    # each dilate
    for i in range(Dilate_time):
        tmp = np.pad(out, (1, 1), 'edge')
        # dilate
        for y in range(1, H+1):
            for x in range(1, W+1):
                if np.sum(K * tmp[y-1: y+2, x-1:x+2]) < 255*4:
                    out[y-1, x-1] = 0

    return out

# Morphology - Erode
def Morphology_Erode(img, Erode_time=1):
    H, W = img.shape

    # Kernel
    K = np.array((
        (0, 1, 0,),
        (1, 0, 1,),
        (0, 1, 0,)
    ), dtype=np.int)

    # each dilate time
    out = img.copy()
    for i in range(Erode_time):
        tmp = np.pad(out, (1, 1), 'edge')
        for y in range(1, H+1):
            for x in range(1, W+1):
                if np.sum(K * tmp[y-1:y+2, x-1:x+2]) >= 255:
                    out[y-1, x-1] = 255

    return out


# Morphology - opening
def Morphology_opening(img, time=1):
    out = Morphology_Dilate(img, Dilate_time=time)
    out = Morphology_Erode(out, Erode_time=time)

    return out

# Read image
img = cv2.imread("Images/house.jpg").astype(np.float32)


# Grayscale
gray = BGR2GRAY(img)

# Otsu's binarization
otsu = otsu_binarization(gray)

# Morphology - opening
out = Morphology_opening(otsu, time=1)

# Save result
cv2.imwrite("out_49.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
