import cv2
import numpy as np

# Read image
img = cv2.imread("Images/house.jpg").astype(np.float32)

# OTSU binary
## gray scale
H, W, C = img.shape
out = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
out = out.astype(np.uint8)

## determine threshold of otsu's binarization
max_departDegree = 0
max_t = 0
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
        max_t = t

## binarization
print("threshold >>", max_t)
out[out >= max_t] = 255
out[out < max_t] = 0

# Morphology filter
MF = np.array((
    (0, 1, 0),
    (1, 0, 1),
    (0, 1, 0),
), dtype=np.float)

# Morphology - dilate
Dilate_time = 3

mor = out.copy()
for i in range(Dilate_time):
    tmp = np.pad(out, (1, 1), 'edge')
    for y in range(1, H+1):
        for x in range(1, W+1):
            if np.sum(MF * tmp[y-1:y+2, x-1:x+2]) >= 255:
                mor[y-1, x-1] = 255

# Morphology - erode
Erode_time = 3
for i in range(Erode_time):
    tmp = np.pad(mor, (1, 1), 'edge')
    for y in range(1, H+1):
        for x in range(1, W+1):
            if np.sum(MF * tmp[y-1:y+2, x-1:x+2]) < 255*4:
                mor[y-1, x-2] = 0

# Black hatt
out = mor - out

# Save result
cv2.imwrite("out_53.jpg", out)
cv2.imshow('result', out)
cv2.waitKey(0)
cv2.destroyAllWindows()