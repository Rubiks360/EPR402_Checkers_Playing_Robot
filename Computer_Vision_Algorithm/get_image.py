import cv2
import numpy as np
from datetime import datetime
import time


vid = cv2.VideoCapture(0)

# vid = cv2.VideoCapture(cv2.CAP_DSHOW)

# vid = cv2.VideoCapture(cv2.CAP_MSMF)

# vid = cv2.VideoCapture(cv2.CAP_VFW)

# vid.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
# vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

# vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# vid.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
# vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

dt = datetime.now()

enable_save = False
count = 0

while True:
    ret, frame = vid.read()
    cv2.imshow('Camera Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('r'):
        enable_save = not enable_save

    print(datetime.now() - dt)
    print(frame.shape)

    if enable_save is True:
        print("Saving...")
        frame_resize = cv2.resize(frame, (300, 300))
        cv2.imwrite("board_images/frame%d.jpg" % count, frame_resize)
        count = count + 1

        time.sleep(3)

    dt = datetime.now()

vid.release()
cv2.destroyAllWindows()
