import matplotlib.pyplot as plt
import numpy as np

from ex3_utils import *
import time
import cv2
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def lkDemo(img_path):
    print("LK Demo")

    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=float)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])

    st = time.time()
    pts, uv = opticalFlow(img_1.astype(float), img_2.astype(float), step_size=20, win_size=5)
    et = time.time()

    print("Time: {:.4f}".format(et - st))
    print(np.median(uv, 0))
    print(np.mean(uv, 0))

    displayOpticalFlow(img_2, pts, uv)

    # RGB
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    img_2 = np.zeros(img_1.shape)
    img_2[:, :, 0] = cv2.warpPerspective(img_1[:, :, 0], t, img_1[:, :, 0].shape[::-1])
    img_2[:, :, 1] = cv2.warpPerspective(img_1[:, :, 1], t, img_1[:, :, 1].shape[::-1])
    img_2[:, :, 2] = cv2.warpPerspective(img_1[:, :, 2], t, img_1[:, :, 2].shape[::-1])

    st = time.time()
    pts, uv = opticalFlow(img_1.astype(float), img_2.astype(float), step_size=20, win_size=5)
    et = time.time()

    print("Time: {:.4f}".format(et - st))
    print(np.median(uv, 0))
    print(np.mean(uv, 0))

    displayOpticalFlow(img_2[:, :, 0], pts, uv)  # displaying only one channel


def hierarchicalDemoHelper(img_1, img_2):
    """
    input of 2 images, running hierarchical LK on the images and converting the results into ones the output function can read.
    :param img_1:
    :param img_2:
    :return:
    """
    st = time.time()
    result = opticalFlowPyrLK(img_1.astype(float), img_2.astype(float), k=4)
    et = time.time()

    retLstOriginal = []
    retLstMoved = []
    for x in range(result.shape[0]):
        for y in range(result.shape[1]):
            if result[x][y][0] != 0 or result[x][y][1] != 0:
                retLstOriginal.append([y, x])
                retLstMoved.append([result[x][y][0], result[x][y][1]])

    retLstOriginal = np.array(retLstOriginal)
    retLstMoved = np.array(retLstMoved)

    print("Time: {:.4f}".format(et - st))
    return retLstOriginal, retLstMoved


def hierarchicalkDemo(img_path1, img_path2):
    """
    ADD TEST
    :param img_path1: Image 1 input
    :param img_path2: Image 2 input
    :return:
    """
    print("Hierarchical LK Demo")
    print("BW image")
    img_1 = cv2.cvtColor(cv2.imread(img_path1), cv2.COLOR_BGR2GRAY)
    img_2 = cv2.cvtColor(cv2.imread(img_path2), cv2.COLOR_BGR2GRAY)

    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    img_2 = cv2.resize(img_2, (0, 0), fx=.5, fy=0.5)

    retLstOriginal, retLstMoved = hierarchicalDemoHelper(img_1, img_2)

    displayOpticalFlow(img_2, retLstOriginal, retLstMoved)

    #############################################

    print("RGB image")
    img_1 = cv2.cvtColor(cv2.imread(img_path1), cv2.COLOR_BGR2RGB)
    img_2 = cv2.cvtColor(cv2.imread(img_path2), cv2.COLOR_BGR2RGB)

    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    img_2 = cv2.resize(img_2, (0, 0), fx=.5, fy=0.5)

    retLstOriginal, retLstMoved = hierarchicalDemoHelper(img_1, img_2)

    displayOpticalFlow(img_2[:, :, 0], retLstOriginal, retLstMoved)


def compareLK(img_path):
    """
    ADD TEST
    Compare the two results from both functions.
    :param img_path: Image input
    :return:
    """
    print("Compare LK & Hierarchical LK")
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=float)
    img1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (0, 0), fx=.5, fy=0.5)
    img2 = cv2.warpPerspective(img1, t, img1.shape[::-1])

    pts, uv = opticalFlow(img1.astype(float), img2.astype(float), step_size=20, win_size=5)
    displayOpticalFlow(img2, pts, uv)

    retLstOriginal, retLstMoved = hierarchicalDemoHelper(img1, img2)
    displayOpticalFlow(img2, retLstOriginal, retLstMoved)


def displayOpticalFlow(img: np.ndarray, pts: np.ndarray, uvs: np.ndarray):
    plt.imshow(img, cmap='gray')
    plt.quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')

    plt.show()


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------

def presentTranslation(st, et, img1, img2, ret):
    print("Time: {:.4f}".format(et - st))
    print("MSE of the 2 images is: " + str(mean_squared_error(img2, img1)))
    print("MSE of the second image to the returned image: " + str(mean_squared_error(img2, cv2.warpPerspective(img1, ret, img1.shape[::-1]))))
    print("The final U and V (according to the algorithm) were: " + str(ret[0][2]) + ", " + str(ret[1][2]) + "\n\n")
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(img1)
    ax[0].set_title("Image 1")
    ax[2].imshow(img2)
    ax[2].set_title("Image 2")
    ax[1].imshow(cv2.warpPerspective(img1, ret, img1.shape[::-1]))
    ax[1].set_title("Translated image")
    plt.show()


def imageWarpingDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """

    print("Image Translation Demo")
    img1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -4],
                  [0, 1, -2],
                  [0, 0, 1]], dtype=float)
    img2 = cv2.warpPerspective(img1, t, img1.shape[::-1])

    cv2.imwrite('imTransA1.jpg', cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite('imTransB1.jpg', cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_RGB2BGR))

    st = time.time()
    ret = findTranslationLK(img1.astype(float), img2.astype(float))
    et = time.time()
    print("The U and V used in the original transformation were -4, -2.")
    presentTranslation(st, et, img1, img2, ret)

    img1 = cv2.cvtColor(cv2.imread("input/door1.jpeg"), cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, 4],
                  [0, 1, -2],
                  [0, 0, 1]], dtype=float)
    img2 = cv2.warpPerspective(img1, t, img1.shape[::-1])
    cv2.imwrite('imTransA2.jpg', cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite('imTransB2.jpg', cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_RGB2BGR))

    ret = findTranslationLK(img1.astype(float), img2.astype(float))
    print("The U and V used in the original transformation were 4, -2.")
    presentTranslation(st, et, img1, img2, ret)

    print("Rigid Transformation (Angles)")
    img1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[np.cos(np.deg2rad(5)), -np.sin(np.deg2rad(5)), 0],
                  [np.sin(np.deg2rad(5)), np.cos(np.deg2rad(5)), 0],
                  [0, 0, 1]], dtype=float)
    img2 = cv2.warpPerspective(img1, t, img1.shape[::-1])
    cv2.imwrite('imRigidA1.jpg', cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite('imRigidB1.jpg', cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_RGB2BGR))

    st = time.time()
    ret = findRigidLK(img1.astype(float), img2.astype(float))
    et = time.time()
    print("Time: {:.4f}".format(et - st))
    print("The angle used in the original transformation was 5.")
    print("The final angle (according to the algorithm) was: " + str(np.rad2deg(np.arccos(ret[0][0]))) + "\n\n")

    img1 = cv2.cvtColor(cv2.imread("input/door1.jpeg"), cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[np.cos(np.deg2rad(10)), -np.sin(np.deg2rad(10)), 0],
                  [np.sin(np.deg2rad(10)), np.cos(np.deg2rad(10)), 0],
                  [0, 0, 1]], dtype=float)
    img2 = cv2.warpPerspective(img1, t, img1.shape[::-1])
    cv2.imwrite('imRigidA2.jpg', cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite('imRigidB2.jpg', cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_RGB2BGR))
    ret = findRigidLK(img1.astype(float), img2.astype(float))
    print("Time: {:.4f}".format(et - st))
    print("The angle used in the original transformation was 10.")
    print("The final angle (according to the algorithm) was: " + str(np.rad2deg(np.arccos(ret[0][0]))) + "\n\n")

    print("Translation: Correlation")
    st = time.time()
    ret = findTranslationCorr(img1.astype(float), img2.astype(float))
    et = time.time()
    presentTranslation(st, et, img1, img2, ret)

    print("Rigid: Correlation")
    st = time.time()
    ret = findRigidCorr(img1.astype(float), img2.astype(float))
    et = time.time()
    print("Time: {:.4f}".format(et - st))
    print("The angle used in the original transformation was 10.")
    print("The final angle (according to the algorithm) was: " + str(np.rad2deg(np.arccos(ret[0][0]))) + "\n\n")

    print("Image Warping Demo")
    img2 = cv2.warpPerspective(img1, t, img1.shape[::-1])
    st = time.time()
    warpImages(img1.astype(float), img2.astype(float), t)
    et = time.time()
    print("Time: {:.4f}".format(et - st))

    t = np.array([[np.cos(np.deg2rad(5)), -np.sin(np.deg2rad(5)), 10],
                  [np.sin(np.deg2rad(5)), np.cos(np.deg2rad(5)), 10],
                  [0, 0, 1]], dtype=float)
    img2 = cv2.warpPerspective(img1, t, img1.shape[::-1])
    warpImages(img1.astype(float), img2.astype(float), t)

    t = np.array([[np.cos(np.deg2rad(10)), -np.sin(np.deg2rad(10)), 0],
                  [np.sin(np.deg2rad(10)), np.cos(np.deg2rad(10)), -10],
                  [0, 0, 1]], dtype=float)
    img2 = cv2.warpPerspective(img1, t, img1.shape[::-1])
    warpImages(img1.astype(float), img2.astype(float), t)


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def pyrGaussianDemo(img_path):
    print("Gaussian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY) / 255
    lvls = 4
    gau_pyr = gaussianPyr(img, lvls)

    h, w = gau_pyr[0].shape[:2]
    canv_h = h
    widths = np.cumsum([w // (2 ** i) for i in range(lvls)])
    widths = np.hstack([0, widths])
    canv_w = widths[-1]
    canvas = np.zeros((canv_h, canv_w, 3))

    for lv_idx in range(lvls):
        h = gau_pyr[lv_idx].shape[0]
        canvas[:h, widths[lv_idx]:widths[lv_idx + 1], :] = gau_pyr[lv_idx]

    plt.imshow(canvas)
    plt.show()


def pyrLaplacianDemo(img_path):
    print("Laplacian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY) / 255
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 7

    lap_pyr = laplaceianReduce(img, lvls)
    re_lap = laplaceianExpand(lap_pyr)

    f, ax = plt.subplots(2, lvls + 1)
    plt.gray()
    for i in range(lvls):
        ax[0, i].imshow(lap_pyr[i])
        ax[1, i].hist(lap_pyr[i].ravel(), 256, [lap_pyr[i].min(), lap_pyr[i].max()])

    ax[0, -1].set_title('Original Image')
    ax[0, -1].imshow(re_lap)
    ax[1, -1].hist(re_lap.ravel(), 256, [0, 1])
    plt.show()


def blendDemo():
    im1 = cv2.cvtColor(cv2.imread('input/sunset.jpg'), cv2.COLOR_BGR2RGB) / 255
    im2 = cv2.cvtColor(cv2.imread('input/cat.jpg'), cv2.COLOR_BGR2RGB) / 255
    mask = cv2.cvtColor(cv2.imread('input/mask_cat.jpg'), cv2.COLOR_BGR2RGB) / 255

    n_blend, im_blend = pyrBlend(im1, im2, mask, 4)

    f, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(im1)
    ax[0, 1].imshow(im2)
    ax[0, 2].imshow(mask)
    ax[1, 0].imshow(n_blend)
    ax[1, 1].imshow(np.abs(n_blend - im_blend))
    ax[1, 2].imshow(im_blend)

    plt.show()

    cv2.imwrite('sunset_cat.png', cv2.cvtColor((im_blend * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def main():
    print("ID:", myID())

    img_path = 'input/boxMan.jpg'
    lkDemo(img_path)
    hierarchicalkDemo("input/door1.jpeg", "input/door2.jpeg")
    compareLK(img_path)

    imageWarpingDemo(img_path)

    pyrGaussianDemo('input/pyr_bit.jpg')
    pyrLaplacianDemo('input/pyr_bit.jpg')
    blendDemo()


if __name__ == '__main__':
    main()
