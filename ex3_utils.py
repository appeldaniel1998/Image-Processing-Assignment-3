import sys
from typing import List
from sklearn.metrics import mean_squared_error
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

def opticalFlowMatrix(im1: np.ndarray, im2: np.ndarray, step_size: int, win_size: int) -> (np.ndarray, np.ndarray):
    I_t = im2 - im1
    ker = np.array([[-1, 0, 1]])
    xDerivative = cv2.filter2D(im2, -1, ker, borderType=cv2.BORDER_REPLICATE)  # Derivative of X
    yDerivative = cv2.filter2D(im2, -1, ker.T, borderType=cv2.BORDER_REPLICATE)  # Derivative of Y
    maskOnes = np.ones((win_size, win_size))  # Mask of ones

    ixSquared = cv2.filter2D(xDerivative ** 2, -1, maskOnes, borderType=cv2.BORDER_REPLICATE)  # x derivative squared
    iySquared = cv2.filter2D(yDerivative ** 2, -1, maskOnes, borderType=cv2.BORDER_REPLICATE)  # y derivative squared

    ixiy = xDerivative * yDerivative
    sigmaIxiy = cv2.filter2D(ixiy, -1, maskOnes, borderType=cv2.BORDER_REPLICATE)  # Sum of window around ixiy

    ixit = xDerivative * I_t
    sigmaIxit = cv2.filter2D(ixit, -1, maskOnes, borderType=cv2.BORDER_REPLICATE)  # Sum of window around ixit

    iyit = yDerivative * I_t
    sigmaIyit = cv2.filter2D(iyit, -1, maskOnes, borderType=cv2.BORDER_REPLICATE)  # Sum of window around iyit

    v_x = np.zeros(im2.shape)  # Inint of arrays
    v_y = np.zeros(im2.shape)

    for x in range(0, im2.shape[0], step_size):
        for y in range(0, im2.shape[1], step_size):
            try:
                mat1 = np.array([[ixSquared[x][y], sigmaIxiy[x][y]],  # matrix of sums
                                 [sigmaIxiy[x][y], iySquared[x][y]]])

                eig = np.linalg.eig(mat1)  # returns eigenvalues of the matrix
                lambda1 = max(eig[0][0], eig[0][1])  # max eigenvalue
                lambda2 = min(eig[0][0], eig[0][1])  # min eigenvalue

                if lambda1 >= lambda2 > 1 and lambda1 / lambda2 < 100:
                    inverseMat1 = np.linalg.inv(mat1)  # inverse of matrix
                    mat2 = np.array([[-sigmaIxit[x][y]], [-sigmaIyit[x][y]]])
                    result = np.matmul(inverseMat1, mat2)  # matrix multiplication
                    v_x[x][y] = result[0][0]
                    v_y[x][y] = result[1][0]
                else:
                    v_x[x][y] = 0
                    v_y[x][y] = 0
            except Exception:  # If there is no inverse matrix
                v_x[x][y] = 0
                v_y[x][y] = 0
    return v_x, v_y


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """

    if im1.ndim == 2:  # BW Image
        v_x, v_y = opticalFlowMatrix(im1, im2, step_size, win_size)
    else:  # RGB
        v_x, v_y = opticalFlowMatrix(im1[:, :, 0], im2[:, :, 0], step_size, win_size)

    retLstOriginal = []  # construct return lists
    retLstMoved = []
    for x in range(v_x.shape[0]):  # For every element in the returned matrix v_x
        for y in range(v_x.shape[1]):
            if v_x[x][y] != 0 or v_y[x][y] != 0:  # wherever the change is not 0
                retLstOriginal.append([y, x])  # add to lists
                retLstMoved.append([v_x[x][y], v_y[x][y]])

    return np.array(retLstOriginal), np.array(retLstMoved)


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int, stepSize: int = 10, winSize: int = 5) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """

    reducedImg1 = gaussianPyr(img1, k)  # construct pyramids
    reducedImg2 = gaussianPyr(img2, k)

    newUV = np.zeros((img1.shape[0], img1.shape[1], 2))

    for ind in range(k - 1, -1, -1):  # for every level beginning with the smallest
        if img1.ndim == 2:  # BW Image
            dx, dy = opticalFlowMatrix(reducedImg1[ind], reducedImg2[ind], stepSize, winSize)
        else:  # RGB Image
            dx, dy = opticalFlowMatrix(reducedImg1[ind][:, :, 0], reducedImg2[ind][:, :, 0], stepSize, winSize)

        for i in range(dx.shape[0]):  # For every pixel in the returned matrix
            for j in range(dx.shape[1]):
                newUV[i][j][0] = dx[i][j] + (2 * newUV[i][j][0])
                newUV[i][j][1] = dy[i][j] + (2 * newUV[i][j][1])

        # Pad image with 0-es (part of gaussian expand but without the pseudo convolution with 121
        expandedImg = np.zeros((newUV.shape[0] * 2, newUV.shape[1] * 2, newUV.shape[2]))
        for k in range(newUV.shape[2]):
            for i in range(newUV.shape[0]):
                for j in range(newUV.shape[1]):
                    expandedImg[i * 2][j * 2][k] = newUV[i][j][k]  # Fill wherever there was supposed to be a value

        newUV = expandedImg

    newUV = newUV[:img2.shape[0], :img2.shape[1], :]
    return newUV


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """

    minMSE = sys.maxsize
    minUV = []
    points, diff = opticalFlow(im1, im2, 10, 5)  # Run LK
    for pointId in range(len(points)):  # for every return point, check the u and v for all image. take the lowest MSE value move
        t = np.array([[1, 0, diff[pointId][0]],  # temp matrix
                      [0, 1, diff[pointId][1]],
                      [0, 0, 1]], dtype=float)
        tempImg2 = cv2.warpPerspective(im1, t, im1.shape[::-1])  # warp image
        currMSE = mean_squared_error(im2, tempImg2)  # calculate MSE
        if currMSE < minMSE:  # Compare MSE
            minMSE = currMSE
            minUV = diff[pointId]
    return np.array([[1, 0, minUV[0]],
                     [0, 1, minUV[1]],
                     [0, 0, 1]], dtype=float)


def optimalAngle(im1: np.ndarray, im2: np.ndarray):
    minT = 0
    t = []
    minMSE = sys.maxsize
    for theta in range(360):  # for every possible angle
        t = np.array([[np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta)), 0],  # temp matrix
                      [np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta)), 0],
                      [0, 0, 1]], dtype=float)
        tempImg2 = cv2.warpPerspective(im1, t, im1.shape[::-1])  # warp image
        currMSE = mean_squared_error(im2, tempImg2)
        if currMSE < minMSE:  # update MSE and minT
            minMSE = currMSE
            minT = theta
    im2 = cv2.warpPerspective(im1, t, im1.shape[::-1])
    return minT, im2


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    minT, im2 = optimalAngle(im1, im2)  # Find optimal angle
    points, diff = opticalFlow(im1, im2, 10, 5)  # find optical flow
    minMSE = sys.maxsize
    minUV = []
    for pointId in range(len(points)):  # For every point in the return points (same as first sub-question)
        t = np.array([[np.cos(np.deg2rad(minT)), -np.sin(np.deg2rad(minT)), diff[pointId][0]],
                      [np.sin(np.deg2rad(minT)), np.cos(np.deg2rad(minT)), diff[pointId][1]],
                      [0, 0, 1]], dtype=float)
        tempImg2 = cv2.warpPerspective(im1, t, im1.shape[::-1])  # Warp image
        currMSE = mean_squared_error(im2, tempImg2)
        if currMSE < minMSE:  # update minMSE
            minMSE = currMSE
            minUV = diff[pointId]
    return np.array([[np.cos(np.deg2rad(minT)), -np.sin(np.deg2rad(minT)), minUV[0]],
                     [np.sin(np.deg2rad(minT)), np.cos(np.deg2rad(minT)), minUV[1]],
                     [0, 0, 1]], dtype=float)


def findIndents(im2: np.ndarray):
    """
    Function to find the indents of a picture from every side.
    :param im2:
    :return:
    """

    # top Indent
    i = 0
    flag = False
    while not flag:
        if np.all((im2[i] == 0)):
            i += 1
        else:
            flag = True
    topIndent = i

    # Bottom Indent
    i = 1
    flag = False
    while not flag:
        if np.all((im2[-i] == 0)):
            i += 1
        else:
            flag = True
    bottomIndent = -i + 1

    # Left Indent
    i = 0
    flag = False
    while not flag:
        if np.all((im2[:, i] == 0)):
            i += 1
        else:
            flag = True
    leftIndent = -i

    # Right Indent
    i = 1
    flag = False
    while not flag:
        if np.all((im2[:, -i] == 0)):
            i += 1
        else:
            flag = True
    rightIndent = -i + 1

    if topIndent != 0:  # finalize top and bottom indents
        topBotIndent = topIndent
    else:
        topBotIndent = bottomIndent

    if leftIndent != 0:  # finalize left and right indents
        leftRightIndent = leftIndent
    else:
        leftRightIndent = rightIndent

    return topBotIndent, leftRightIndent


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """

    topBotIndent, leftRightIndent = findIndents(im2)
    t = np.array([[1, 0, leftRightIndent],
                  [0, 1, topBotIndent],
                  [0, 0, 1]], dtype=float)
    return t


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """

    minT, im2 = optimalAngle(im1, im2)  # Find optimal angle
    t = findTranslationCorr(im1, im2)  # find optimal translation

    t[0][0] = np.cos(np.deg2rad(minT))  # Update t with the correct angle
    t[0][1] = -np.sin(np.deg2rad(minT))
    t[1][0] = np.sin(np.deg2rad(minT))
    t[1][1] = np.cos(np.deg2rad(minT))

    return t


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    newIm1 = np.zeros(im2.shape)
    inverseT = np.linalg.inv(T)  # find inverse of the transformation matrix
    if inverseT[0][1] != 0:  # If there is an angle in the matrix
        deg = np.rad2deg(np.arcsin(inverseT[1][0]))
        rad = np.deg2rad(-deg)
        inverseT[0][0] = np.cos(rad)  # filling with the inverse angle
        inverseT[1][1] = np.cos(rad)
        inverseT[0][1] = -np.sin(rad)
        inverseT[1][0] = np.sin(rad)

    for x in range(im2.shape[0]):  # For every pixel in the image
        for y in range(im2.shape[1]):
            coordinates = np.matmul(inverseT, np.array([[x], [y], [1]]))  # Find ne coordinates
            coordinates = np.round(coordinates)  # Round coordinates to nearest pixel
            try:  # try to add the pixel in the correct place (if indexes don't exist, do nothing)
                newIm1[int(coordinates[0][0])][int(coordinates[1][0])] = im2[x][y]
            except Exception:
                pass

    f, ax = plt.subplots(1, 3)  # Plot the results
    ax[0].imshow(im1)
    ax[0].set_title("Image 1")
    ax[1].imshow(newIm1)
    ax[1].set_title("Supposed Image 1")
    ax[2].imshow(im2)
    ax[2].set_title("Image 2")
    plt.show()

    return newIm1


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------

def getGaussKernel(kSize: int, sigma: float = -1) -> np.ndarray:
    k = cv2.getGaussianKernel(kSize, sigma)  # Getting 1D gaussian kernel
    k /= k[0, 0]  # Restoring original form before normalization due to the original form containing 1 as the first element
    kernel = k @ k.T  # Getting the dot product of the gaussian kernel with itself transposed
    kernel /= kernel.sum()  # Divide by sum to normalize and sum to 1
    return kernel


def gaussianReduce(img: np.ndarray) -> np.ndarray:
    """
    reduce the image: blur the image and take every second pixel
    :param img:
    :return:
    """
    gaussKer = getGaussKernel(5, 1.2)
    if img.ndim == 3:  # RGB image
        ret = np.zeros((img.shape[0] // 2, img.shape[1] // 2, 3))
        for dim in range(3):  # for every channel
            blurredImg = cv2.filter2D(img[:, :, dim], -1, gaussKer, borderType=cv2.BORDER_REPLICATE)  # Blur channel
            for i in range(0, blurredImg.shape[0], 2):
                for j in range(0, blurredImg.shape[1], 2):
                    try:
                        ret[i // 2][j // 2][dim] = blurredImg[i][j]  # Fill new smaller array
                    except Exception:
                        pass

    else:  # BW Image
        blurredImg = cv2.filter2D(img, -1, gaussKer, borderType=cv2.BORDER_REPLICATE)  # Blur Image
        imgAsList = []
        for i in range(0, blurredImg.shape[0], 2):
            row = []  # init list of row
            for j in range(0, blurredImg.shape[1], 2):
                row.append(blurredImg[i][j])  # Fill new smaller array
            imgAsList.append(row)  # Add row to final list
        ret = np.array(imgAsList)  # convert list to array

    return ret


def pseudoConvolve(img: np.ndarray, paddedImg: np.ndarray) -> np.ndarray:
    """
    Helper function of Expand. When image is padded with 0-es, perform a pass over rows and columns and calculate the "in between" values
    :param img:
    :param paddedImg:
    :return:
    """
    # Pseudo-convolve with 1,2,1
    # Over rows
    expandedImg = np.zeros((img.shape[0] * 2, img.shape[1] * 2))
    for i in range(0, paddedImg.shape[0], 2):
        row = [paddedImg[i][0]]
        for j in range(1, paddedImg.shape[1] - 1):
            newVal = 0.5 * (paddedImg[i][j - 1] + (2 * paddedImg[i][j]) + paddedImg[i][j + 1])  # ccalc new value
            row.append(newVal)  # add value to row
        row.append(paddedImg[i][-1])
        expandedImg[i] = row  # add row to 2d list

    expandedImg = np.array(expandedImg)  # Convert list to  array
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


def gaussianExpand(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        paddedImg = np.zeros((img.shape[0] * 2, img.shape[1] * 2))

        # Pad image with 0-es
        for i in range(img.shape[0]):
            row = []
            for j in range(img.shape[1]):
                row.append(img[i][j])
                row.append(0)
            paddedImg[i * 2] = row
        expandedImg = pseudoConvolve(img, paddedImg)
    else:
        # Pad image with 0-es
        expandedImg = np.zeros((img.shape[0] * 2, img.shape[1] * 2, 3))
        for dim in range(3):
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    expandedImg[i * 2][j * 2][dim] = img[i][j][dim]
            expandedImg[:, :, dim] = pseudoConvolve(img, expandedImg[:, :, dim])

    return expandedImg


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
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
        currExpand = gaussianExpand(gaussianPyramid[-i + 1])
        try:
            laplacianImage = gaussianPyramid[-i] - currExpand
        except Exception:
            temp = gaussianPyramid[-i].copy()
            gaussianPyramid[-i] = gaussianPyramid[-i][:-(temp.shape[0] - currExpand.shape[0]), :-(temp.shape[1] - currExpand.shape[1]), :]
            laplacianImage = gaussianPyramid[-i] - currExpand
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
        gaussianExpanded = gaussianExpand(img)
        try:
            img = gaussianExpanded + lap_pyr[-i]
        except Exception:
            gaussianExpanded = gaussianExpanded[:-1, :-1, :]
            img = gaussianExpanded + lap_pyr[-i]

    return img


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    mask = np.round(mask)
    laplacianImg1 = laplaceianReduce(img_1, levels)
    laplacianImg2 = laplaceianReduce(img_2, levels)
    maskGaussianPyr = gaussianPyr(mask, levels)
    for i in range(len(maskGaussianPyr)):
        temp = maskGaussianPyr[i].copy()
        tempLap1 = laplacianImg1[i]
        if temp.shape[0] - tempLap1.shape[0] != 0:
            temp = temp[:-(temp.shape[0] - tempLap1.shape[0]), :, :]
        if temp.shape[1] - tempLap1.shape[1] != 0:
            temp = temp[:, :-(temp.shape[1] - tempLap1.shape[1]), :]
        maskGaussianPyr[i] = temp

    ls = []
    for la, lb, maskPyrInd in zip(laplacianImg1, laplacianImg2, maskGaussianPyr):
        ls.append((la * maskPyrInd) + (lb * (1 - maskPyrInd)))

    expanded = laplaceianExpand(ls)

    # True blend complete

    # Naive blend:
    naiveBlend = np.zeros(expanded.shape)
    if img_1.ndim == 3:
        for i in range(expanded.shape[0]):  # For every pixel in the image
            for j in range(expanded.shape[1]):
                for k in range(expanded.shape[2]):
                    try:
                        if mask[i][j][k] == 1:  # Wherever the mask is 1, the new image's pixel value will be placed
                            naiveBlend[i][j][k] = img_1[i][j][k]
                        else:
                            naiveBlend[i][j][k] = img_2[i][j][k]
                    except Exception:
                        naiveBlend[i][j][k] = img_2[i][j][k]
    else:
        for i in range(expanded.shape[0]):  # For every pixel in the image
            for j in range(expanded.shape[1]):
                try:
                    if mask[i][j] == 1:  # Wherever the mask is 1, the new image's pixel value will be placed
                        naiveBlend[i][j] = img_1[i][j]
                    else:
                        naiveBlend[i][j] = img_2[i][j]
                except Exception:
                    naiveBlend[i][j] = img_2[i][j]

    return naiveBlend, expanded
