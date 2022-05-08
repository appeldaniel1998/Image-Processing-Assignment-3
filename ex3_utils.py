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


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10,
                win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    I_t = im1 - im2
    xDerivative = cv2.filter2D(im2 / 255, -1, np.array([-1, 0, 1]), borderType=cv2.BORDER_REPLICATE)
    yDerivative = cv2.filter2D(im2 / 255, -1, np.array([[-1], [0], [1]]), borderType=cv2.BORDER_REPLICATE)
    maskOnes = np.ones((win_size, win_size))

    ixSquared = cv2.filter2D(xDerivative, -1, maskOnes, borderType=cv2.BORDER_REPLICATE)
    iySquared = cv2.filter2D(yDerivative, -1, maskOnes, borderType=cv2.BORDER_REPLICATE)

    ixiy = xDerivative * yDerivative
    sigmaIxiy = cv2.filter2D(ixiy, -1, maskOnes, borderType=cv2.BORDER_REPLICATE)

    ixit = xDerivative * I_t
    sigmaIxit = cv2.filter2D(ixit, -1, maskOnes, borderType=cv2.BORDER_REPLICATE)

    iyit = yDerivative * I_t
    sigmaIyit = cv2.filter2D(iyit, -1, maskOnes, borderType=cv2.BORDER_REPLICATE)

    v_x = np.zeros(im2.shape)
    v_y = np.zeros(im2.shape)

    for x in range(im2.shape[0]):
        for y in range(im2.shape[1]):
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

    retLstOriginal = []
    retLstMoved = []
    for x in range(v_x.shape[0]):
        for y in range(v_x.shape[1]):
            if v_x[x][y] != 0 or v_y[x][y] != 0:
                retLstOriginal.append([y, x])
                retLstMoved.append([np.round(v_x[x][y]), np.round(v_y[x][y])])

    return np.array(retLstOriginal), np.array(retLstMoved)


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int,
                     stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """
    pass


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


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    pass


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    pass


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    pass


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
