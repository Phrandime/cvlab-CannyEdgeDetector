import cv2
import numpy as np
from time import time


C = [[0] * 100 for i in range(100)]
for i in range(100):
    C[i][0] = 1
    for j in range(1, i):
        C[i][j] = C[i - 1][j - 1] + C[i - 1][j]
    C[i][i] = 1
del i, j
neighbors = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

extension = ['.png', '.jpg']


def bigkernel(a, unit):
    size = len(unit)
    if (size & 1) == 0:
        unit = np.concatenate(([0], unit))
        size += 1
    halfsize = int(size / 2)
    outmatrix = np.zeros((a, a))
    halfsize = size // 2
    for i in range(halfsize, a - halfsize):
        outmatrix[i, i - halfsize: i + halfsize + 1] = unit
    for i in range(0, halfsize):
        unit_edge = unit[halfsize - i:] + \
                    np.concatenate((np.flip(unit[:halfsize - i]), np.zeros(2 * i + 1)))
        outmatrix[i, :i + halfsize + 1] = unit_edge
    for i in range(0, halfsize):
        unit_edge = unit[:halfsize + i + 1] + \
                    np.concatenate((np.zeros(2 * i + 1), np.flip(unit[halfsize + i + 1:])))
        outmatrix[a - 1 - i, a - 1 - i - halfsize:] = unit_edge
    return outmatrix


def subsampling_kernel(h, s=2):
    a = -(-h // s)
    out = np.zeros((a, h))
    indices = np.array([i * s for i in range(a)]).reshape((a, 1))
    np.put_along_axis(out, indices, 1., axis=1)
    return out


def low_pass(img, kernel=np.array([0.25, 0.5, 0.25])):
    bgr = list(cv2.split(img))
    h, w = img.shape[:2]
    kernelH = bigkernel(h, kernel)
    kernelW = bigkernel(w, kernel).T
    for i in range(len(bgr)):
        bgr[i] = kernelH @ bgr[i] @ kernelW
    return cv2.merge(bgr)


def image_subsampling(img, s=2):
    return img[::s, ::s]


def lowpass_and_subsampling(img, s=2, kernel=np.array([0.25, 0.5, 0.25])):
    bgr = list(cv2.split(img))
    h, w = img.shape[:2]
    kernelH = bigkernel(h, kernel)
    subH = subsampling_kernel(h, s)
    kernelH = subH @ kernelH
    kernelW = bigkernel(w, kernel)
    subW = subsampling_kernel(w, s)
    kernelW = (subW @ kernelW).T
    for i in range(len(bgr)):
        bgr[i] = kernelH @ bgr[i] @ kernelW
    return cv2.merge(bgr)


def gaussian_pyramid(img, depth=3, s=2, kernel=np.array([0.25, 0.5, 0.25])):
    pyramid = [np.array(img)]
    for i in range(depth):
        sub = lowpass_and_subsampling(pyramid[-1], s, kernel)
        pyramid.append(sub)
    return pyramid


def get_edge(img, threshold=10):
    bgr = list(cv2.split(img))
    h, w = img.shape[:2]
    kernelH = bigkernel(h, np.array([1, 0, -1]))
    kernelW = bigkernel(w, np.array([1, 0, -1])).T
    pdH, pdW, pd = [], [], []
    for i in range(len(bgr)):
        pdH.append(kernelH @ bgr[i])
        pdW.append(bgr[i] @ kernelW)
        pd.append(pdH[i] ** 2 + pdW[i] ** 2)
    out = sum(pd) >= len(bgr) * threshold ** 2
    return out


def sobel(img, threshold=10, kernel=np.array([0.25, 0.5, 0.25])):
    bgr = list(cv2.split(img))
    h, w = img.shape[:2]
    filterH = bigkernel(h, kernel)
    filterW = bigkernel(w, kernel).T
    detectorH = bigkernel(h, np.array([1, 0, -1]))
    detectorW = bigkernel(w, np.array([1, 0, -1])).T
    pdH, pdW, pd = [], [], []
    for i in range(len(bgr)):
        pdH.append(detectorH @ bgr[i] @ filterW)
        pdW.append(filterH @ bgr[i] @ detectorW)
        pd.append(pdH[i] ** 2 + pdW[i] ** 2)
    out = sum(pd) >= len(bgr) * threshold ** 2
    return out


def canny_simplified(img, threshold=10, kernel=np.array([0.25, 0.5, 0.25])):
    bgr = list(cv2.split(img))
    h, w = img.shape[:2]
    filterH = bigkernel(h, kernel)
    filterW = bigkernel(w, kernel).T
    detectorH = bigkernel(h, np.array([1, 0, -1]))
    detectorW = bigkernel(w, np.array([1, 0, -1])).T
    pdH, pdW, pd = [], [], []
    for i in range(len(bgr)):
        pdH.append(detectorH @ bgr[i] @ filterW)
        pdW.append(filterH @ bgr[i] @ detectorW)
        pd.append(pdH[i] ** 2 + pdW[i] ** 2)
    argmax = np.argmax(pd, axis=0)
    pd = (sum(pd) / len(bgr)) ** 0.5
    out = np.zeros((h, w), dtype=bool)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            p0 = pd[y, x]
            if p0 < threshold:
                continue
            dy = pdH[argmax[y][x]][y, x]
            dx = pdW[argmax[y][x]][y, x]
            if abs(dy) > abs(dx):
                k = dx / dy
                if k > 0:
                    p1 = (1 - k) * pd[y + 1, x] + k * pd[y + 1, x + 1]
                    p2 = (1 - k) * pd[y - 1, x] + k * pd[y - 1, x - 1]
                else:
                    p1 = (1 + k) * pd[y + 1, x] - k * pd[y + 1, x - 1]
                    p2 = (1 + k) * pd[y - 1, x] - k * pd[y - 1, x + 1]
            else:
                k = dy / dx
                if k > 0:
                    p1 = (1 - k) * pd[y, x + 1] + k * pd[y + 1, x + 1]
                    p2 = (1 - k) * pd[y, x - 1] + k * pd[y - 1, x - 1]
                else:
                    p1 = (1 + k) * pd[y, x + 1] - k * pd[y - 1, x + 1]
                    p2 = (1 + k) * pd[y, x - 1] - k * pd[y + 1, x - 1]
            if p0 >= p1 and p0 >= p2:
                out[y, x] = True
    return out
    

def canny(img, threshold=(15, 10), kernel=np.array([0.25, 0.5, 0.25])):
    start = time()
    bgr = list(cv2.split(img))
    h, w = img.shape[:2]
    print(f'canny for {h} × {w}, {len(bgr)} channels, ', end='')
    filterH = bigkernel(h, kernel)
    filterW = bigkernel(w, kernel).T
    detectorH = bigkernel(h, np.array([1, 0, -1]))
    detectorW = bigkernel(w, np.array([1, 0, -1])).T
    pdH, pdW, pd = [], [], []
    for i in range(len(bgr)):
        pdH.append(detectorH @ bgr[i] @ filterW)
        pdW.append(filterH @ bgr[i] @ detectorW)
        pd.append(pdH[i] ** 2 + pdW[i] ** 2)
    argmax = np.argmax(pd, axis=0)
    pd = (sum(pd) / len(bgr)) ** 0.5
    out = np.zeros((h, w), dtype=int)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            p0 = pd[y, x]
            if p0 <= threshold[-1]:
                continue
            dy = pdH[argmax[y][x]][y, x]
            dx = pdW[argmax[y][x]][y, x]
            if abs(dy) > abs(dx):
                k = dx / dy
                if k > 0:
                    p1 = (1 - k) * pd[y + 1, x] + k * pd[y + 1, x + 1]
                    p2 = (1 - k) * pd[y - 1, x] + k * pd[y - 1, x - 1]
                else:
                    p1 = (1 + k) * pd[y + 1, x] - k * pd[y + 1, x - 1]
                    p2 = (1 + k) * pd[y - 1, x] - k * pd[y - 1, x + 1]
            else:
                k = dy / dx
                if k > 0:
                    p1 = (1 - k) * pd[y, x + 1] + k * pd[y + 1, x + 1]
                    p2 = (1 - k) * pd[y, x - 1] + k * pd[y - 1, x - 1]
                else:
                    p1 = (1 + k) * pd[y, x + 1] - k * pd[y - 1, x + 1]
                    p2 = (1 + k) * pd[y, x - 1] - k * pd[y + 1, x - 1]
            if p0 >= p1 and p0 >= p2:
                for i in range(len(threshold)):
                    if p0 >= threshold[i]:
                        out[y, x] = 255 - i
                        break

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if out[y, x] == 0 or out[y, x] == 255:
                continue
            for _i, _j in neighbors:
                if out[y + _i, x + _j] == 255:
                    out[y, x] = 255
                    break

    for y in range(h - 2, 0, -1):
        for x in range(w - 2, 0, -1):
            if out[y, x] == 0 or out[y, x] == 255:
                continue
            for _i, _j in neighbors:
                if out[y + _i, x + _j] == 255:
                    out[y, x] = 255
                    break
    print(f'finished in {time() - start:.2f} seconds.')
    return out == 255


def imread(img_name):
    img = None
    for s in extension:
        img = cv2.imread(img_name + s)
        if img is not None:
            return img
    raise FileNotFoundError(f'No file named {img_name}')


def imwrite_gauss_pyramid(img_name, depth):
    img = imread(img_name)
    pyramid = gaussian_pyramid(img, depth)
    for i in range(1, depth + 1):
        cv2.imwrite(f'{img_name}_{i}_gauss.png', pyramid[i])


def imwrite_edge(img_name, thresh, combine=False):
    img_old = imread(img_name)
    img = low_pass(img_old)
    out = canny(img, thresh).astype('uint8') * 255
    if combine:
        cv2.imwrite(f'{img_name}_with_canny_{thresh}.png', cv2.hconcat([img_old, cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)]))
    else:
        cv2.imwrite(f'{img_name}_gauss.png', img)
        cv2.imwrite(f'{img_name}_canny_{thresh}.png', out)


def imwrite_edge_pyramid_direct(img_name, depth, thresh):
    img = imread(img_name)
    pyramid = gaussian_pyramid(img, depth)
    for i in range(1, depth + 1):
        out = canny(pyramid[i], thresh).astype('uint8') * 255
        cv2.imwrite(f'{img_name}_{i}_canny_{thresh}_direct.png', out)


def imwrite_edge_pyramid(img_name, depth, thresh):
    img = imread(img_name)
    img = low_pass(img)
    for i in range(1, depth + 1):
        img = image_subsampling(img)
        img = low_pass(img)
        out = canny(img, thresh).astype('uint8') * 255
        cv2.imwrite(f'{img_name}_{i}_canny_{thresh}.png', out)


def imwrite_gauss_pyramid_gray(img_name, depth):
    img = imread(img_name)
    img = cv2.cvtColor(img, 6)
    pyramid = gaussian_pyramid(img, depth)
    for i in range(1, depth + 1):
        cv2.imwrite(f'{img_name}_{i}_gauss_gray.png', pyramid[i])


def imwrite_edge_gray(img_name, thresh):
    img = imread(img_name)
    img = cv2.cvtColor(img, 6)
    img = low_pass(img)
    out = canny(img, thresh).astype('uint8') * 255
    cv2.imwrite(f'{img_name}_canny_{thresh}_gray.png', out)


def imwrite_edge_pyramid_direct_gray(img_name, depth, thresh):
    img = imread(img_name)
    img = cv2.cvtColor(img, 6)
    pyramid = gaussian_pyramid(img, depth)
    for i in range(1, depth + 1):
        out = canny(pyramid[i], thresh).astype('uint8') * 255
        cv2.imwrite(f'{img_name}_{i}_canny_{thresh}_direct_gray.png', out)


def imwrite_edge_pyramid_gray(img_name, depth, thresh):
    img = imread(img_name)
    img = cv2.cvtColor(img, 6)
    img = lowpass_and_subsampling(img)
    for i in range(1, depth + 1):
        img = low_pass(img)
        out = canny(img, thresh).astype('uint8') * 255
        cv2.imwrite(f'{img_name}_{i}_canny_{thresh}_gray.png', out)
        if i < depth:
            img = image_subsampling(img)




if __name__ == '__main__':
    img_name = 'sparky'
    # imwrite_gauss_pyramid(img_name, 2)
    # imwrite_edge_pyramid(img_name, 5, (20, 10))
    imwrite_edge(img_name, thresh=(18, 1), combine=True)