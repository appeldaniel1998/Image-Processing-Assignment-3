import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 207386699


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------

def opticalFlowMatrix(im1: np.ndarray, im2: np.ndarray, step_size: int, win_size: int) -> (np.ndarray, np.ndarray):  # Completed
    I_t = im1 - im2
    xDerivative = cv2.filter2D(im2 / 255, -1, np.array([-1, 0, 1]), borderType=cv2.BORDER_REPLICATE)
    yDerivative = cv2.filter2D(im2 / 255, -1, np.array([[-1], [0], [1]]), borderType=cv2.BORDER_REPLICATE)
    maskOnes = np.ones((win_size, win_size))

    ixSquared = np.square(cv2.filter2D(xDerivative, -1, maskOnes, borderType=cv2.BORDER_REPLICATE))
    iySquared = np.square(cv2.filter2D(yDerivative, -1, maskOnes, borderType=cv2.BORDER_REPLICATE))

    ixiy = xDerivative * yDerivative
    sigmaIxiy = cv2.filter2D(ixiy, -1, maskOnes, borderType=cv2.BORDER_REPLICATE)

    ixit = xDerivative * I_t
    sigmaIxit = cv2.filter2D(ixit, -1, maskOnes, borderType=cv2.BORDER_REPLICATE)

    iyit = yDerivative * I_t
    sigmaIyit = cv2.filter2D(iyit, -1, maskOnes, borderType=cv2.BORDER_REPLICATE)

    v_x = np.zeros(im2.shape)
    v_y = np.zeros(im2.shape)

    for x in range(0, im2.shape[0], step_size):
        for y in range(0, im2.shape[1], step_size):
            try:
                mat1 = np.array([[ixSquared[x][y], sigmaIxiy[x][y]],
                                 [sigmaIxiy[x][y], iySquared[x][y]]])
                lambda1, lambda2 = np.linalg.eig(mat1)[0][0], np.linalg.eig(mat1)[0][1]
                if lambda1 >= lambda2 > 1 and lambda1 / lambda2 < 100:
                    inverseMat1 = np.linalg.inv(mat1)
                    mat2 = np.array([[-sigmaIxit[x][y]], [-sigmaIyit[x][y]]])
                    result = np.matmul(inverseMat1, mat2)
                    v_x[x][y] = result[0][0]
                    v_y[x][y] = result[1][0]
                else:
                    v_x[x][y] = 0
                    v_y[x][y] = 0
            except Exception:
                v_x[x][y] = 0
                v_y[x][y] = 0
    return v_x, v_y


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):  # Completed
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """

    v_x, v_y = opticalFlowMatrix(im1, im2, step_size, win_size)
    retLstOriginal = []
    retLstMoved = []
    for x in range(v_x.shape[0]):
        for y in range(v_x.shape[1]):
            if v_x[x][y] != 0 or v_y[x][y] != 0:
                retLstOriginal.append([y, x])
                retLstMoved.append([np.round(v_x[x][y]), np.round(v_y[x][y])])

    return np.array(retLstOriginal), np.array(retLstMoved)


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int, stepSize: int, winSize: int) -> np.ndarray:  # TODO
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """

    reducedImg1 = img1.copy()
    reducedImg2 = img2.copy()

    for i in range(k):
        reducedImg1 = cv2.pyrDown(reducedImg1)
        reducedImg2 = cv2.pyrDown(reducedImg2)

    u, v = opticalFlowMatrix(reducedImg1, reducedImg2, stepSize, winSize)

    newUV = np.array((u.shape[0], u.shape[1], 2))
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            # newUV[i][j][0] =
            pass  # TODO


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    pass


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    pass


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    pass


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    pass


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    pass


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------

def getGaussKernel(kSize: int, sigma: float = -1) -> np.ndarray:  # Completed
    k = cv2.getGaussianKernel(kSize, sigma)  # Getting 1D gaussian kernel
    k /= k[0, 0]  # Restoring original form before normalization due to the original form containing 1 as the first element
    kernel = k @ k.T  # Getting the dot product of the gaussian kernel with itself transposed
    kernel /= kernel.sum()  # Divide by sum to normalize and sum to 1
    return kernel


def gaussianReduce(img: np.ndarray) -> np.ndarray:  # Completed
    gaussKer = getGaussKernel(5, 1.2)
    blurredImg = cv2.filter2D(img, -1, gaussKer, borderType=cv2.BORDER_REPLICATE)
    imgAsList = []
    for i in range(0, blurredImg.shape[0], 2):
        row = []
        for j in range(0, blurredImg.shape[1], 2):
            row.append(blurredImg[i][j])
        imgAsList.append(row)
    ret = np.array(imgAsList)
    return ret


def gaussianExpand(img: np.ndarray) -> np.ndarray:  # Completed
    paddedImg = np.zeros((img.shape[0] * 2, img.shape[1] * 2))

    # Pad image with 0-es
    for i in range(img.shape[0]):
        row = []
        for j in range(img.shape[1]):
            row.append(img[i][j])
            row.append(0)
        # row = row[:-1]
        paddedImg[i * 2] = row

    # Pseudo-convolve with 1,2,1
    # Over rows
    expandedImg = np.zeros((img.shape[0] * 2, img.shape[1] * 2))
    for i in range(0, paddedImg.shape[0], 2):
        row = [paddedImg[i][0]]
        for j in range(1, paddedImg.shape[1] - 1):
            newVal = 0.5 * (paddedImg[i][j - 1] + (2 * paddedImg[i][j]) + paddedImg[i][j + 1])
            row.append(newVal)
        row.append(paddedImg[i][-1])
        expandedImg[i] = row

    expandedImg = np.array(expandedImg)
    paddedImg = expandedImg.copy()

    # Over columns
    expandedImg = np.zeros((img.shape[1] * 2, img.shape[0] * 2))
    for i in range(paddedImg.shape[1]):
        col = [paddedImg[0][i]]
        for j in range(1, paddedImg.shape[0] - 1):
            newVal = 0.5 * (paddedImg[j - 1][i] + (2 * paddedImg[j][i]) + paddedImg[j + 1][i])
            col.append(newVal)
        col.append(paddedImg[-1][i])
        expandedImg[i] = col
    expandedImg = expandedImg.transpose()

    expandedImg = np.array(expandedImg)
    expandedImg[-1] = expandedImg[-2]
    expandedImg[:, -1] = expandedImg[:, -2]
    return expandedImg


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:  # Completed
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    ret = [img]
    for i in range(1, levels):
        ret.append(gaussianReduce(ret[-1]))
    return ret


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    gaussianPyramid = gaussianPyr(img, levels)
    ret = [gaussianPyramid[-1]]
    for i in range(2, levels + 1):
        laplacianImage = gaussianPyramid[-i] - gaussianExpand(gaussianPyramid[-i + 1])
        ret.append(laplacianImage)
    ret.reverse()
    return ret


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    img = lap_pyr[-1]
    for i in range(2, len(lap_pyr) + 1):
        img = gaussianExpand(img) + lap_pyr[-i]
    return img


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,
             mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    pass
