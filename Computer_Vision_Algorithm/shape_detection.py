import cv2
from datetime import datetime
import numpy as np

import matplotlib.image as mpimg

from image_transforms import *

vid = cv2.VideoCapture(0)

# vid.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
# vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

my_exposure = -4.5  # -4.5 and -3.5
vid.set(cv2.CAP_PROP_EXPOSURE, my_exposure)

# dt = datetime.now()

file_name = 0
count = 0

custom_function = True

# dt = datetime.now()
# print("time = ", datetime.now() - dt)

if custom_function:


    '''
    ring light #5
    exposure = -4.5
    threshold = 0.08 (Black+White board), 0.12 (wood board)
    '''

    threshold = 0.08

    while True:
        dt = datetime.now()
        ret, img = vid.read()

        # img_un = undistorted_image_v22(img)
        # img_crop = img_un[50:450, 110:510]

        # my_gray = grayscale_image(img)
        # img_un = undistorted_image_v22(my_gray)
        # img_crop = img_un[50:450, 110:510]
        # img_blur = blur_image(img_crop)
        # img_edges = get_adaptive_threshold_mean_image(img_blur, threshold)

        '''
        print("Got image")
        # cv2.imshow("My Image", img)
        # cv2.waitKey(0)

        my_gray = grayscale_image(img)
        # cv2.imshow("My Image", my_gray)
        # cv2.waitKey(0)

        img_un = undistorted_image_v22(my_gray)
        # cv2.imshow("My Image", img_un)
        # cv2.waitKey(0)

        img_crop = img_un[50:450, 110:510]
        # cv2.imshow("My Image", img_crop)
        # cv2.waitKey(0)

        img_blur = blur_image(img_crop)
        # cv2.imshow("My Image", img_blur)
        # cv2.waitKey(0)

        img_edges = get_adaptive_threshold_mean_image(img_blur, threshold)
        # cv2.imshow("My Image", img_edges)
        # cv2.waitKey(0)

        # img_grid = get_board_grid(img_edges)
        # img_grid_v2 = get_board_grid_v2(img_edges)

        # plt.imshow(img_grid, cmap=plt.get_cmap('gray'))
        # plt.show()

        # img_grid_big = np.zeros((400, 400), dtype=np.uint8)
        # for h in range(img_grid.shape[0]):
        #     for w in range(img_grid.shape[1]):
        #         img_grid_big[40 * h: 40 * (h+1), 40 * w: 40 * (w+1)] = img_grid[h][w]

        # img_grid_big = np.zeros((400, 400), dtype=np.uint8)
        # for h in range(img_grid_v2.shape[0]):
        #     for w in range(img_grid_v2.shape[1]):
        #         img_grid_big[10 * h: 10 * (h + 1), 10 * w: 10 * (w + 1)] = img_grid_v2[h][w]

        # img_edged_pool = custom_threshold_pooling(img_edges, 80, 80, 2000)

        # kernel = np.zeros((30, 30))
        #
        # for h in range(kernel.shape[0]):
        #     for w in range(kernel.shape[1]):
        #         r = np.sqrt((h - 15) ** 2 + (w - 15) ** 2)
        #         if (r > 12) & (r < 16):
        #             kernel[h, w] = 1
        
        # '''


        '''
        kernel = np.zeros((50, 50))
        # kernel = np.full((50, 50), -0.5)
        thickness = 6
        offset = 3

        for h in range(kernel.shape[0]):
            if (h >= offset) & (h <= (offset + thickness - 1)):
                kernel[h, offset: (kernel.shape[0] - offset)] = 1
            elif (h >= (kernel.shape[0] - offset - thickness)) & (h <= (kernel.shape[0] - offset - 1)):
                kernel[h, offset: (kernel.shape[0] - offset)] = 1

        for w in range(kernel.shape[1]):
            if (w >= offset) & (w <= (offset + thickness - 1)):
                kernel[offset: (kernel.shape[1] - offset), w] = 1
            elif (w >= (kernel.shape[1] - offset - thickness)) & (w <= (kernel.shape[1] - offset - 1)):
                kernel[offset: (kernel.shape[1] - offset), w] = 1

        # count_1 = 0
        # for h in range(kernel.shape[0]):
        #     for w in range(kernel.shape[1]):
        #         if kernel[h, w] == 1:
        #             count_1 += 1
        # print(count_1)

        # '''

        img_save = img_edges

        '''
        dt_ = datetime.now()
        print("begin circle conv", datetime.now())
        circle_piece_conv_out = custom_convolution_operation_no_padding(img_edges, kernel, 200 * 255, thickness)
        for h in range(circle_piece_conv_out.shape[0]):
            for w in range(circle_piece_conv_out.shape[1]):
                if circle_piece_conv_out[h, w] == 255:
                    cv2.circle(img_save, (w, h), 1, (125, 125, 125), 3)
        print("time circle conv =", datetime.now() - dt_)
        
        '''

        key_input = cv2.waitKey(1)
        if key_input & 0xFF == ord('q'):
            break
        elif key_input & 0xFF == ord('e'):
            my_exposure += 1.0
            vid.set(cv2.CAP_PROP_EXPOSURE, my_exposure)
        elif key_input & 0xFF == ord('d'):
            my_exposure -= 1.0
            vid.set(cv2.CAP_PROP_EXPOSURE, my_exposure)
        elif key_input & 0xFF == ord('+'):
            threshold += 0.02
        elif key_input & 0xFF == ord('-'):
            threshold -= 0.02
        elif key_input & 0xFF == ord('s'):
            # cv2.imwrite("Board_robot_view_480p/%d_%d.jpg" {file_name, count}, img_save)
            count += 1
            print("saving image")
        elif key_input & 0xFF == ord('f'):
            file_name += 1
            print("changed file name")

        # cv2.circle(img_save, (int(img_save.shape[1] / 2), int(img_save.shape[0] / 2)), 5, (125, 125, 125), 3)

        cv2.imshow("My Image", img_save)
        print("Image done, time =", datetime.now() - dt, ",exposure =", my_exposure, ",threshold =", threshold)
        # cv2.waitKey(0)

else:
    while True:
        ret, img = vid.read()
        # cv2.imshow('Camera Feed', frame)

        key_input = cv2.waitKey(1)
        if key_input & 0xFF == ord('q'):
            break
        elif key_input & 0xFF == ord('e'):
            my_exposure += 1.0
            vid.set(cv2.CAP_PROP_EXPOSURE, my_exposure)
            print(my_exposure)
        elif key_input & 0xFF == ord('d'):
            my_exposure -= 1.0
            vid.set(cv2.CAP_PROP_EXPOSURE, my_exposure)
            print(my_exposure)
        elif key_input & 0xFF == ord('r'):
            cv2.imwrite("frame_measure.jpg", img)
            count += 1
            print("saving image")
        elif key_input & 0xFF == ord('f'):
            file_name += 1
            print("changed file name")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Blur using 3 * 3 kernel.
        gray_blurred = cv2.blur(gray, (3, 3))

        # Apply Hough transform on the blurred image.
        detected_circles = cv2.HoughCircles(gray_blurred,
                                            cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                                            param2=30, minRadius=1, maxRadius=30)

        # Draw circles that are detected.
        if detected_circles is not None:

            # Convert the circle parameters a, b and r to integers.
            detected_circles = np.uint16(np.around(detected_circles))

            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]

                # Draw the circumference of the circle.
                cv2.circle(img, (a, b), r, (0, 255, 0), 2)

                # Draw a small circle (of radius 1) to show the center.
                cv2.circle(img, (a, b), 1, (0, 0, 255), 3)

        cv2.imshow("Detected Circle", img)
        cv2.waitKey(0)

vid.release()
cv2.destroyAllWindows()
