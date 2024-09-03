import cv2
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def grayscale_image(img):
    # 0.299 ∙ Red +
    # 0.587 ∙ Green +
    # 0.114 ∙ Blue

    # my_img = mpimg.imread('test_img.jpg')
    # print('Image shape:', my_img.shape)
    # dt = datetime.now()

    gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)  # RGB
    # gray = np.dot(img[..., :3], [0.1140, 0.5870, 0.2989]).astype(np.uint8)  # BGR

    # print("time = ", datetime.now() - dt)
    # print('Gray dimensions:', gray.shape)

    # cv2.imwrite("gray_scale_image.jpg", gray)

    # plt.imshow(gray, cmap=plt.get_cmap('gray'))
    # plt.show()

    return gray

def undistorted_image(img):
    original_img = img.copy()

    h_original, w_original = original_img.shape[0], original_img.shape[1]
    h_original_2, w_original_2 = int(h_original / 2), int(w_original / 2)
    new_img = np.zeros(original_img.shape)

    '''
    # leave img as original 0, 0, 0, 1
    a = -0.01
    b = 0
    c = 0
    d = 1

    # Using negative values shifts distant points away from the center. This counteracts barrel distortion
    # Using positive values shifts distant points towards the center

    # Correcting using 'a' affects only the outermost pixels of the image,  while 'b' correction is more uniform

    # distance of a pixel from the center of the source image (r_src)
    # distance in the corrected image (r_corr):

    # r_src = (a * r_dest ** 3 + b * r_dest ** 2 + c * r_dest + d) * r_dest
    '''
    for h in range(h_original):
        for w in range(int(w_original / 2)):
            if (h != h_original_2) & (w != w_original_2):
                # angle from center to current pixel
                if w >= w_original_2:
                    theta = 180 + np.arctan((h - h_original_2) / (w - w_original_2))
                else:
                    theta = np.arctan((h - h_original_2) / (w - w_original_2))

                # r from current pixel from center of image
                r = np.sqrt((w_original_2 - w) ** 2 + (h_original_2 - h) ** 2)

                r_change = 1.0105 ** r

                if r_change > 50:
                    r_change = 50

                # r_change = (r / 400) * 20

                # r_change = 0

                new_r = r - r_change

                old_h = h_original_2 - int(new_r * np.sin(theta))
                old_w = w_original_2 - int(new_r * np.cos(theta))

                new_img[h][w] = original_img[old_h][old_w]

                new_img[h_original - 1 - h][w_original - 1 - w] = original_img[h_original - 1 - old_h][w_original - 1 - old_w]
            else:
                new_img[h][w] = new_img[h-1][w]
                new_img[h_original - 1 - h][w_original - 1 - w] = new_img[h_original - 1 - h + 1][w_original - 1 - w]

    # plt.imshow(new_img, cmap=plt.get_cmap('gray'))
    # plt.show()

    # cv2.imwrite("gray_scale_image_undistort.jpg", new_img)

    return new_img


def undistorted_image_optimize(img):
    original_img = img

    h_original, w_original = original_img.shape[0], original_img.shape[1]
    h_original_2, w_original_2 = int(h_original / 2), int(w_original / 2)
    new_img = np.zeros(original_img.shape, dtype=np.uint8)

    for h in range(int(h_original / 2)):
        for w in range(int(w_original / 2)):
            if (h != h_original_2) & (w != w_original_2):
                # angle from center to current pixel
                theta = np.arctan((h - h_original_2) / (w - w_original_2))

                # r from current pixel from center of image
                r = np.sqrt((w_original_2 - w) ** 2 + (h_original_2 - h) ** 2)

                # r_change = 1.0107 ** r
                r_change = 1.0105 ** r
                # r_change = 1.0105 ** (r + 10)

                # 50
                if r_change > 50:
                    r_change = 50

                # r_change = (r / 400) * 20
                # r_change = 0

                new_r = r - r_change

                old_h = h_original_2 - int(new_r * np.sin(theta))
                old_w = w_original_2 - int(new_r * np.cos(theta))

                # top left
                new_img[h][w] = original_img[old_h][old_w]

                # top right
                new_img[h][w_original - 1 - w] = original_img[old_h][w_original - 1 - old_w]

                # bottom left
                new_img[h_original - 1 - h][w] = original_img[h_original - 1 - old_h][old_w]

                # bottom right
                new_img[h_original - 1 - h][w_original - 1 - w] = original_img[h_original - 1 - old_h][w_original - 1 - old_w]


    # plt.imshow(new_img, cmap=plt.get_cmap('gray'))
    # plt.show()

    # cv2.imwrite("gray_scale_image_undistort.jpg", new_img)

    return new_img


def undistorted_image_v2(img):
    original_img = img.copy()

    h_original, w_original = original_img.shape[0], original_img.shape[1]
    new_img = np.zeros(original_img.shape, dtype=np.uint8)

    # height stretch
    max_h_distort = 20
    for w in range(w_original):
        h_change = int((max_h_distort * np.cos((2 * np.pi * w) / (w_original - 1)) + max_h_distort) / 2)

        h_temp = original_img[:, w]

        for i in range(int(h_original / 2)):
            pos = int((i / (h_original - 1)) * ((h_original - 1) - h_change)) + h_change
            h_temp[i] = original_img[pos, w]

            h_temp[h_original - 1 - i] = original_img[h_original - 1 - pos, w]

        new_img[:, w] = h_temp

    # width stretch
    max_w_distort = 20
    for h in range(h_original):
        w_change = int((max_w_distort * np.cos((2 * np.pi * h) / (h_original - 1)) + max_w_distort) / 2)

        w_temp = original_img[h, :]

        for i in range(int(w_original / 2)):
            pos = int((i / (w_original - 1)) * ((w_original - 1) - w_change)) + w_change
            w_temp[i] = original_img[h, pos]

            w_temp[w_original - 1 - i] = original_img[h, w_original - 1 - pos]

        new_img[h, :] = w_temp

    # plt.imshow(new_img, cmap=plt.get_cmap('gray'))
    # plt.show()

    # cv2.imwrite("gray_scale_image_undistort_v2.jpg", new_img)

    return new_img


def undistorted_image_v21(img):
    original_img = img.copy()

    h_original, w_original = original_img.shape[0], original_img.shape[1]
    h_original_2, w_original_2 = int(h_original / 2), int(w_original / 2)
    new_img = np.zeros(original_img.shape, dtype=np.uint8)

    # height stretch
    for w in range(w_original):
        h_change = -1 * int(-1 * ((w - w_original_2) * 0.018) ** 2 + 0)
        if h_change < 0:
            h_change = 0

        h_temp = original_img[:, w]

        for i in range(h_original_2):
            pos = int((i / (h_original - 1)) * ((h_original - 1) - h_change)) + h_change
            h_temp[i] = original_img[pos, w]
            h_temp[h_original - 1 - i] = original_img[h_original - 1 - pos, w]

        new_img[:, w] = h_temp

    # width stretch
    for h in range(h_original):
        w_change = -1 * int(-1 * ((h - h_original_2) * 0.02) ** 2)

        w_temp = original_img[h, :]

        for i in range(w_original_2):
            pos = int((i / (w_original - 1)) * ((w_original - 1) - w_change)) + w_change
            w_temp[i] = original_img[h, pos]
            w_temp[w_original - 1 - i] = original_img[h, w_original - 1 - pos]

        new_img[h, :] = w_temp

    # plt.imshow(new_img, cmap=plt.get_cmap('gray'))
    # plt.show()

    # cv2.imwrite("gray_scale_image_undistort_v2.jpg", new_img)

    return new_img

'''
def undistorted_image_v22(img):
    original_img = img.copy()

    h_original, w_original = original_img.shape[0], original_img.shape[1]
    new_img = np.zeros(original_img.shape, dtype=np.uint8)

    # height stretch
    for w in range(w_original):
        h_change = -1 * int(-1 * ((w - 400) * 0.021) ** 2 + 0)
        if h_change < 0:
            h_change = 0

        h_temp = original_img[:, w]

        for i in range(int(h_original / 2)):
            pos = int((i / ((h_original / 2) - 1)) * (((h_original / 2) - 1) - h_change)) + h_change

            h_temp[i] = original_img[pos, w]
            h_temp[h_original - 1 - i] = original_img[h_original - 1 - pos, w]

        new_img[:, w] = h_temp

    # width stretch
    for h in range(h_original):
        w_change = -1 * int(-1 * ((h - 300) * 0.024) ** 2)

        w_temp = original_img[h, :]

        for i in range(int(w_original / 2)):
            pos = int((i / ((w_original / 2) - 1)) * (((w_original / 2) - 1) - w_change)) + w_change

            w_temp[i] = original_img[h, pos]
            w_temp[w_original - 1 - i] = original_img[h, w_original - 1 - pos]

        new_img[h, :] = w_temp

    # plt.imshow(new_img, cmap=plt.get_cmap('gray'))
    # plt.show()

    # cv2.imwrite("gray_scale_image_undistort_v2.jpg", new_img)

    return new_img
'''

def undistorted_image_v22(img):
    original_img = img.copy()

    h_original, w_original = original_img.shape[0], original_img.shape[1]
    h_original_div_2 = int(h_original / 2)
    w_original_div_2 = int(w_original / 2)
    new_img = np.zeros(original_img.shape, dtype=np.uint8)

    # height stretch
    for w in range(w_original):
        # 800x600 = 0.021
        # 640x480 = 0.024
        h_change = -1 * int(-1 * ((w - w_original_div_2) * 0.024) ** 2 + 0)
        if h_change < 0:
            h_change = 0

        h_temp = original_img[:, w]

        for i in range(h_original_div_2):
            pos = int((i / (h_original_div_2 - 1)) * ((h_original_div_2 - 1) - h_change)) + h_change

            h_temp[i] = original_img[pos, w]
            h_temp[h_original - 1 - i] = original_img[h_original - 1 - pos, w]

        new_img[:, w] = h_temp

    # width stretch
    for h in range(h_original):
        # 800x600 = 0.024
        # 640x480 = 0.026
        w_change = -1 * int(-1 * ((h - h_original_div_2) * 0.026) ** 2)

        w_temp = original_img[h, :]

        for i in range(w_original_div_2):
            pos = int((i / (w_original_div_2 - 1)) * ((w_original_div_2 - 1) - w_change)) + w_change

            w_temp[i] = original_img[h, pos]
            w_temp[w_original - 1 - i] = original_img[h, w_original - 1 - pos]

        new_img[h, :] = w_temp

    # plt.imshow(new_img, cmap=plt.get_cmap('gray'))
    # plt.show()

    # cv2.imwrite("gray_scale_image_undistort_v2.jpg", new_img)

    return new_img


def undistorted_image_v3(img):
    original_img = img.copy()

    h_original, w_original = original_img.shape[0], original_img.shape[1]
    new_img = np.zeros(original_img.shape, dtype=np.uint8)

    # height stretch
    max_h_distort = 30
    for w in range(w_original):
        h_change = int((max_h_distort * np.cos((2 * np.pi * w) / (w_original - 1)) + max_h_distort) / 2)

        h_temp = original_img[:, w]

        for i in range(int(h_original / 2)):
            pos = h_change + int(np.sin((2 * np.pi * i) / (2 * h_original - 1)) * (h_original * 0.5 - h_change))
            h_temp[i] = original_img[pos, w]
            h_temp[h_original - 1 - i] = original_img[h_original - 1 - pos, w]

        new_img[:, w] = h_temp

    # width stretch
    max_w_distort = 20
    for h in range(h_original):
        w_change = int((max_w_distort * np.cos((2 * np.pi * h) / (h_original - 1)) + max_w_distort) / 2)

        w_temp = original_img[h, :]

        for i in range(int(w_original / 2)):
            pos = w_change + int(np.sin((2 * np.pi * i) / (2 * w_original - 1)) * (w_original * 0.5 - w_change))
            w_temp[i] = original_img[h, pos]
            w_temp[w_original - 1 - i] = original_img[h, w_original - 1 - pos]

        new_img[h, :] = w_temp

    # plt.imshow(new_img, cmap=plt.get_cmap('gray'))
    # plt.show()

    # cv2.imwrite("gray_scale_image_undistort_v2.jpg", new_img)

    return new_img


def edges():
    # my_img = mpimg.imread('test_img.jpg')
    my_img = mpimg.imread('gray_scale_image.jpg')
    # my_img = mpimg.imread('gray_scale_image_undistort.jpg')
    img_edges = cv2.Canny(my_img, 10, 250)
    cv2.imshow("Edges", img_edges)
    cv2.waitKey(0)

    my_img = mpimg.imread('gray_scale_image_undistort.jpg')
    img_edges = cv2.Canny(my_img, 10, 250)
    cv2.imshow("Edges", img_edges)
    cv2.waitKey(0)


def blur_image(img):

    '''
    kernel_size = 3

    # Gaussian blur 3 × 3
    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) * (1 / 16)

    # Box blur
    # kernel = np.array([
    #     [1, 1, 1],
    #     [1, 1, 1],
    #     [1, 1, 1]
    # ]) * (1 / 9)

    # Ridge or edge detection
    # kernel = np.array([
    #     [0, -1, 0],
    #     [-1, 4, -1],
    #     [0, -1, 0]
    # ])
    # kernel = np.array([
    #     [-1, -1, -1],
    #     [-1, 8, -1],
    #     [-1, -1, -1]
    # ])

    original_img = img
    # padded_img = np.pad(original_img, (kernel_size - 1, kernel_size - 1))
    padded_img = np.pad(original_img, (kernel_size // 2, kernel_size // 2))

    h_original, w_original = original_img.shape[0], original_img.shape[1]

    output = []
    for i in range(h_original):
        for j in range(w_original):
            output.append(np.sum(padded_img[i:kernel_size + i, j:kernel_size + j] * kernel))

    new_img = np.array(output).reshape((h_original, w_original)).astype(np.uint8)

    # plt.imshow(new_img, cmap=plt.get_cmap('gray'))
    # plt.show()

    return new_img
    '''

    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) * (1 / 16)

    # kernel = np.array([
    #     [1, 4, 7, 4, 1],
    #     [4, 16, 26, 16, 4],
    #     [7, 26, 41, 26, 7],
    #     [4, 16, 26, 16, 4],
    #     [1, 4, 7, 4, 1]
    # ]) * (1 / 273)

    return custom_convolution_operation(img, kernel, -1)


def get_adaptive_threshold_mean_image(img, threshold=0.0):
    # adaptive_thresholdMean

    block_size = 3

    # https://medium.com/geekculture/image-thresholding-from-scratch-a66ae0fb6f09

    ''' 1 '''
    # binary = np.zeros_like(img, dtype=np.uint8)
    ''' 2 '''
    binary = np.full_like(img, 255, dtype=np.uint8)

    h_original, w_original = img.shape[0], img.shape[1]

    for h in range(h_original):
        for w in range(w_original):
            x_min = max(0, h - block_size // 2)
            y_min = max(0, w - block_size // 2)
            x_max = min(h_original - 1, h + block_size // 2)
            y_max = min(w_original - 1, w + block_size // 2)
            block = img[x_min:x_max + 1, y_min:y_max + 1]

            ''' 1 '''
            # thresh = np.mean(block) - threshold
            # if img[h, w] >= thresh:
            #     binary[h, w] = 255

            thresh = np.mean(block) - threshold
            if img[h, w] >= thresh:
                binary[h, w] = 0

            ''' 2 '''
            # thresh = np.mean(block) - 0.4
            # if img[h, w] >= thresh:
            #     binary[h, w] = 0

    return binary


def get_adaptive_threshold_Gaussian_image(img, threshold=0.0):
    # Calculate the local threshold for each pixel using a Gaussian filter
    threshold_img = blur_image(img)
    threshold_img = threshold_img - threshold

    # Apply the threshold to the input image
    binary = np.zeros_like(img, dtype=np.uint8)
    binary[img >= threshold_img] = 255

    return binary


def custom_threshold_pooling(img, n_h, n_w, threshold):
    h_original, w_original = img.shape[0], img.shape[1]

    n_h_space = int(h_original / n_h)
    n_w_space = int(w_original / n_w)

    grid = np.zeros((n_h, n_w))

    for h in range(n_h):
        for w in range(n_w):
            grid_sum = np.sum(img[h * n_h_space: (h + 1) * n_h_space, w * n_w_space: (w + 1) * n_w_space])

            if grid_sum >= threshold:
                grid[h][w] = 255
            else:
                grid[h][w] = 0

    return grid


def custom_normal_pooling(img, n_h, n_w):
    h_original, w_original = img.shape[0], img.shape[1]

    n_h_space = int(h_original / n_h)
    n_w_space = int(w_original / n_w)

    grid = np.zeros((n_h, n_w))

    for h in range(n_h):
        for w in range(n_w):
            grid_sum = np.sum(img[h * n_h_space: (h + 1) * n_h_space, w * n_w_space: (w + 1) * n_w_space])
            grid[h][w] = grid_sum

    return grid


def custom_average_pooling(img, n_h, n_w):
    h_original, w_original = img.shape[0], img.shape[1]

    n_h_space = int(h_original / n_h)
    n_w_space = int(w_original / n_w)

    grid = np.zeros((n_h, n_w))

    for h in range(n_h):
        for w in range(n_w):
            grid_sum = np.sum(img[h * n_h_space: (h + 1) * n_h_space, w * n_w_space: (w + 1) * n_w_space])
            grid[h][w] = grid_sum / (n_h_space * n_w_space)

    return grid


def get_board_grid(img):
    conv_100 = custom_threshold_pooling(img, 100, 100, 3000)
    # plt.imshow(conv_100, cmap=plt.get_cmap('gray'))
    # plt.show()

    conv_10 = custom_threshold_pooling(conv_100, 10, 10, 15000)
    # plt.imshow(conv_10, cmap=plt.get_cmap('gray'))
    # plt.show()

    return conv_10


def get_board_grid_v2(img):
    conv_100 = custom_threshold_pooling(img, 100, 100, 4000)
    # plt.imshow(conv_100, cmap=plt.get_cmap('gray'))
    # plt.show()

    conv_40 = custom_threshold_pooling(conv_100, 40, 40, 1000)
    # plt.imshow(conv_10, cmap=plt.get_cmap('gray'))
    # plt.show()

    return conv_40


def custom_convolution_operation(input_img, kernel, threshold):
    h_img, w_img = input_img.shape[0], input_img.shape[1]
    h_kernel, w_kernel = kernel.shape[0], kernel.shape[1]

    pad = h_kernel // 2

    new_array = np.zeros((h_img + 2 * pad, w_img + 2 * pad))
    new_array[pad:-pad, pad:-pad] = input_img
    input_img = new_array

    result = np.zeros((h_img, w_img))

    for h in range(0, h_img, 1):
        for w in range(0, w_img, 1):
            sum = 0
            for h_k in range(h_kernel):
                for w_k in range(w_kernel):
                    sum += kernel[h_k][w_k] * input_img[h + h_k][w + w_k]

            if threshold > 0:
                if sum >= threshold:
                    result[h][w] = 255
                else:
                    result[h][w] = 0
            else:
                result[h][w] = sum / (h_kernel * w_kernel)

    return result


def custom_convolution_operation_no_padding(input_img, kernel, threshold, stride):
    h_img, w_img = input_img.shape[0], input_img.shape[1]
    h_kernel, w_kernel = kernel.shape[0], kernel.shape[1]

    result = np.zeros((h_img, w_img))

    for h in range(0, h_img - h_kernel + 1, stride):
        for w in range(0, w_img - w_kernel + 1, stride):
            sum = 0
            for h_k in range(h_kernel):
                for w_k in range(w_kernel):
                    sum += kernel[h_k][w_k] * input_img[h + h_k][w + w_k]

            if threshold > 0:
                if sum >= threshold:
                    result[h][w] = 255
                else:
                    result[h][w] = 0
            else:
                result[h][w] = sum / (h_kernel * w_kernel)

    return result


def get_board_grid_calibrate(img):
    # input 480 640 px
    dt_ = datetime.now()
    img_gray = grayscale_image(img)
    print("time gray = ", datetime.now() - dt_)

    dt_ = datetime.now()
    img_un = undistorted_image_v22(img_gray)
    print("time undis = ", datetime.now() - dt_)

    dt_ = datetime.now()
    img_blur = blur_image(img_un)
    print("time blur = ", datetime.now() - dt_)

    dt_ = datetime.now()
    img_threshold = get_adaptive_threshold_mean_image(img_blur, 0.0)
    print("time ad thresh = ", datetime.now() - dt_)

    # reduce 10x in size
    dt_ = datetime.now()
    conv_calibrate_0 = custom_threshold_pooling(img_threshold, 48, 64, 18000)
    print("time 10x small = ", datetime.now() - dt_)

    # plt.imshow(conv_100, cmap=plt.get_cmap('gray'))
    # plt.show()

    # do convolution

    # kernel = np.array([
    #     [0, 0, 0, 0, 0],
    #     [0, 1, 1, 1, 0],
    #     [0, 1, 1, 1, 0],
    #     [0, 1, 1, 1, 0],
    #     [0, 0, 0, 0, 0]
    # ])

    # kernel = np.array([
    #     [1, 1, 1],
    #     [1, 1, 1],
    #     [1, 1, 1]
    # ])
    #
    # h_img, w_img = conv_calibrate_0.shape[0], conv_calibrate_0.shape[1]
    # h_kernel, w_kernel = kernel.shape[0], kernel.shape[1]
    #
    # # kernel_stride = 1
    #
    # res_h = h_img - h_kernel + 1
    # res_w = w_img - w_kernel + 1
    # result = np.zeros((res_h, res_w))
    #
    # threshold = 255 * (3 * 3) * 0.5
    #
    # for h in range(0, res_h, 1):
    #     for w in range(0, res_w, 1):
    #         sum = 0
    #         for h_k in range(h_kernel):
    #             for w_k in range(w_kernel):
    #                 sum += kernel[h_k][w_k] * conv_calibrate_0[h + h_k][w + w_k]
    #
    #         # result[h][w] = sum
    #
    #         # 1500 for 3x3 5x5 small block detector
    #         if sum >= threshold:
    #             result[h][w] = 255
    #         else:
    #             result[h][w] = 0

    # kernel_0 = np.array([
    #     [1, 1, 1],
    #     [1, 1, 1],
    #     [1, 1, 1]
    # ])
    #
    # conv_res_0 = custom_convolution_operation(conv_calibrate_0, kernel_0, 255 * (3 * 3) * 0.5)

    dt_ = datetime.now()

    kernel_0 = np.array([
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
    ])

    conv_res_0 = custom_convolution_operation(conv_calibrate_0, kernel_0, 255 * (4 * 4) * 16 * 1.1)

    h_list = []
    w_list = []
    for h in range(conv_res_0.shape[0]):
        for w in range(conv_res_0.shape[1]):
            if conv_res_0[h, w] == 255:
                h_list.append(h)
                w_list.append(w)

    h_board = (np.sum(h_list) / len(h_list)) / 48
    w_board = (np.sum(w_list) / len(w_list)) / 64

    print("time board h w = ", datetime.now() - dt_)

    return h_board, w_board, img_un


def get_board_black_square_finder(img):
    dt_ = datetime.now()
    img_gray = grayscale_image(img)
    print("time gray = ", datetime.now() - dt_)

    dt_ = datetime.now()
    img_un = undistorted_image_v22(img_gray)
    print("time undis = ", datetime.now() - dt_)

    dt_ = datetime.now()
    img_blur = blur_image(img_un)
    print("time blur = ", datetime.now() - dt_)

    dt_ = datetime.now()
    img_threshold = get_adaptive_threshold_mean_image(img_blur, 0.0)
    print("time ad thresh = ", datetime.now() - dt_)

    '''
    factors of 60 = 1, 2, 3, 4, 5, 6,    10, 12, 15,     20, 30,     60
    factors of 80 = 1, 2,    4, 5,    8, 10,         16, 20,     40,    80
    
    480: 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30, 32, 40, 48, 60,     80, 96, 120,      160, 240, 480
    640: 1, 2,    4, 5,    8, 10,         16, 20,         32, 40,         64, 80,          128, 160, 320,     640.
    '''

    # reduce 8x in size
    reduce_size = 8
    h_new = int(480 / reduce_size)
    w_new = int(640 / reduce_size)
    dt_ = datetime.now()
    # conv_calibrate_0 = custom_threshold_pooling(img_threshold, 48, 64, 18000)
    conv_calibrate_0 = custom_threshold_pooling(img_threshold, h_new, w_new, reduce_size * reduce_size * 255 * 0.75)  # = 60x80 and 11000 thesh
    print("time 8x small = ", datetime.now() - dt_)

    # do convolution

    # kernel = np.array([
    #     [0, 0, 0, 0, 0],
    #     [0, 1, 1, 1, 0],
    #     [0, 1, 1, 1, 0],
    #     [0, 1, 1, 1, 0],
    #     [0, 0, 0, 0, 0]
    # ])

    kernel_0 = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])

    conv_res_0 = custom_convolution_operation(conv_calibrate_0, kernel_0, 255 * (3 * 3) * 0.5)

    # search for 0 255 0 255 0 255 0 255 0 lines

    h_pos = []
    w_pos = []

    for h in range(conv_res_0.shape[0]):
        change_counter = 0
        prev_val = 0
        for w in range(conv_res_0.shape[1]):
            if conv_res_0[h, w] != prev_val:
                prev_val = conv_res_0[h, w]
                change_counter += 1

        if change_counter >= 8:
            h_pos.append(h)

    for w in range(conv_res_0.shape[1]):
        change_counter = 0
        prev_val = 0
        for h in range(conv_res_0.shape[0]):
            if conv_res_0[h, w] != prev_val:
                prev_val = conv_res_0[h, w]
                change_counter += 1

        if change_counter >= 8:
            w_pos.append(w)

    # h_pos = ((np.array(h_pos) / h_new) * img_un.shape[0])
    # w_pos = ((np.array(w_pos) / w_new) * img_un.shape[1])

    print(h_pos)
    print(w_pos)

    h_pos = np.array([int((i / h_new) * img_un.shape[0]) for i in h_pos])
    w_pos = np.array([int((i / w_new) * img_un.shape[1]) for i in w_pos])

    # return h_pos[0], w_pos[0], h_pos[-1], w_pos[-1], img_un
    return h_pos, w_pos, img_threshold


########################################################################
########################################################################
# my_img = mpimg.imread('name.jpg')
# cv2.imwrite("name.jpg", img_in)

# img_original = mpimg.imread('img_0.jpg')
#
# dt = datetime.now()
# img_gray = grayscale_image(img_original)
# print("time gray = ", datetime.now() - dt)
#
# plt.imshow(img_gray, cmap=plt.get_cmap('gray'))
# plt.show()

# dt = datetime.now()
# un_img = undistorted_image_optimize(my_img)
# un_img = undistorted_image_v2(my_img)
# un_img = undistorted_image_v21(my_img)
# un_img = undistorted_image_v3(my_img)

# dt = datetime.now()
# img_un = undistorted_image_v22(img_gray)
# print("time un-distort = ", datetime.now() - dt)
#
# plt.imshow(img_un, cmap=plt.get_cmap('gray'))
# plt.show()

# print("distort time = ", datetime.now() - dt)
# dt = datetime.now()
# img_blur = blur_image(img_un)
# print("time blur = ", datetime.now() - dt)

# dt = datetime.now()
# edg_img = obtain_bin_image(img_blur)
# print("time bin image = ", datetime.now() - dt)

# my_img = mpimg.imread('edge_detected.jpg')
# plt.imshow(my_img, cmap=plt.get_cmap('gray'))
# plt.show()

# h_image = detect_h_lines(my_img)



'''
img_init = mpimg.imread('img_0.jpg')
plt.imshow(img_init, cmap=plt.get_cmap('gray'))
plt.show()

img_gray = grayscale_image(img_init)
# img_gray = mpimg.imread('gray_scale_image.jpg')
plt.imshow(img_gray, cmap=plt.get_cmap('gray'))
plt.show()

img_un = undistorted_image_v22(img_gray)
plt.imshow(img_un, cmap=plt.get_cmap('gray'))
plt.show()

# img_crop = img_un[20:540, 120:640]  # img_un[20:420, 120:520]
# plt.imshow(img_crop, cmap=plt.get_cmap('gray'))
# plt.show()

# img_threshold = get_adaptive_threshold_Gaussian_image(img_crop, 2)
# plt.imshow(img_threshold, cmap=plt.get_cmap('gray'))
# plt.show()

img_blur = blur_image(img_un)
plt.imshow(img_blur, cmap=plt.get_cmap('gray'))
plt.show()

img_threshold = get_adaptive_threshold_mean_image(img_blur, 0.0)
plt.imshow(img_threshold, cmap=plt.get_cmap('gray'))
plt.show()

# cv2.imwrite("Board_robot_view_temp/img_threshold_test.jpg", img_threshold)

# img_grid = get_board_grid_v2(img_threshold)

'''

# img_0
# frame928

img_init = mpimg.imread('img_0.jpg').copy()

'''
dt = datetime.now()
b_h, b_w, img_return_grid = get_board_grid_calibrate(img_init)
print("time center = ", datetime.now() - dt)

center_h = int(img_return_grid.shape[0] * b_h)
center_w = int(img_return_grid.shape[1] * b_w)

print("center", center_h, center_w)

cv2.circle(img_return_grid, (center_w, center_h), 10, (125, 125, 125), 3)
cv2.imshow("Detected Board", img_return_grid)
cv2.waitKey(0)
# '''

'''
dt = datetime.now()
h_0, w_0, h_1, w_1, img_return = get_board_black_square_finder(img_init)
cv2.rectangle(img_return, (int(img_return.shape[1] * w_0), int(img_return.shape[0] * h_0)), (int(img_return.shape[1] * w_1), int(img_return.shape[0] * h_1)), (125, 125, 125), 3)
cv2.imshow("Detected Board", img_return)
cv2.waitKey(0)
print("time total = ", datetime.now() - dt)
# '''



'''
dt = datetime.now()
h_array, w_array, img_return_black_square = get_board_black_square_finder(img_init)
print("time squares = ", datetime.now() - dt)

# print(h_array)
# print(w_array)

# for h in h_array:
#     for w in w_array:
#         cv2.circle(img_return, (int(img_return.shape[1] * w), int(img_return.shape[0] * h)), 2, (125, 125, 125), 3)

for h in h_array:
    cv2.line(img_return_black_square, (0, h), (img_return_black_square.shape[1], h), (125, 125, 125), 2)

for w in w_array:
    cv2.line(img_return_black_square, (w, 0), (w, img_return_black_square.shape[0]), (125, 125, 125), 2)

cv2.imshow("Detected Squares", img_return_black_square)
cv2.waitKey(0)
# '''



'''
canny edge detector

Apply Gaussian filter to smooth the image in order to remove the noise
Find the intensity gradients of the image
Apply gradient magnitude thresholding or lower bound cut-off suppression to get rid of spurious response to edge detection
Apply double threshold to determine potential edges
Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.
'''


'''
corner detection

1. Obtain binary image. We load the image, convert to grayscale, Gaussian blur, 
then adaptive (local) threshold to obtain a black/white binary image. 
We then remove small noise using contour area filtering. 
At this stage we also create two blank masks.

2. Detect horizontal and vertical lines. 
Now we isolate horizontal lines by creating a horizontal shaped 
kernel and perform morphological operations. To detect vertical lines, 
we do the same but with a vertical shaped kernel. 
We draw the detected lines onto separate masks.

3. Find intersection points. The idea is that if we combine the horizontal 
and vertical masks, the intersection points will be the corners. 
We can perform a bitwise-and operation on the two masks. 
Finally we find the centroid of each intersection point and 
highlight corners by drawing a circle.

'''



''' Lab book stuff '''

'''
img_in = mpimg.imread('img_1.jpg').copy()

# out = undistorted_image(img_in)
# out = undistorted_image_optimize(img_in)
# out = undistorted_image_v2(img_in)
# out = undistorted_image_v21(img_in)
# out = undistorted_image_v22(img_in)
out = undistorted_image_v3(img_in)


cv2.imwrite("Board_robot_view_temp/undistort_v3.jpg", out)
'''

