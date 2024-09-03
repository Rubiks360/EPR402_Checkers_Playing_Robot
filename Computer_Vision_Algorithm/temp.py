import cv2
from datetime import datetime

from sympy import primenu

vid = cv2.VideoCapture(0)

vid.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

# vid.set(cv2.CAP_PROP_EXPOSURE, -3.0)

exposure = -3.0
# vid.set(cv2.CAP_PROP_EXPOSURE,exposure)

dt = datetime.now()

while True:
    ret, frame = vid.read()
    cv2.imshow('Camera Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('e'):
        exposure += 0.5
        vid.set(cv2.CAP_PROP_EXPOSURE, exposure)
        print(exposure)

    if cv2.waitKey(1) & 0xFF == ord('d'):
        exposure -= 0.5
        vid.set(cv2.CAP_PROP_EXPOSURE, exposure)
        print(exposure)

    # print(datetime.now() - dt)
    # print(frame.shape)
    # dt = datetime.now()

vid.release()
cv2.destroyAllWindows()

