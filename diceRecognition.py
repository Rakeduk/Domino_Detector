import cv2
import numpy as np
import time
from sklearn import cluster
import os

params = cv2.SimpleBlobDetector_Params()

params.filterByInertia
params.minInertiaRatio = 0.4
params.filterByConvexity
params.minConvexity = 0.5
params.filterByArea
params.minArea = 800


detector = cv2.SimpleBlobDetector_create(params)


def get_yellow(frame):
    frame_blurred = cv2.medianBlur(frame, 11)
    # cv2.imshow("Blured image", resizeDown(frame_blurred))

    frame_hsv = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2HSV)
    # cv2.imshow("Blured hsv", resizeDown(frame_hsv))


    lower_yellow = np.array([10, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    mask = cv2.inRange(frame_hsv, lower_yellow, upper_yellow)
    rev_image = cv2.bitwise_not(mask)
    # cv2.imshow("Reversed", resizeDown(rev_image))
    # cv2.imwrite('no_convercity.jpg', resizeDown(rev_image))

    return rev_image


def get_blobs(frame):
    blobs = detector.detect(frame)

    return blobs


def get_dice_from_blobs(blobs):
    # Get centroids of all blobs
    X = []
    for b in blobs:
        pos = b.pt

        if pos != None:
            X.append(pos)

    X = np.asarray(X)

    if len(X) > 0:
        # Important to set min_sample to 0, as a dice may only have one dot
        clustering = cluster.DBSCAN(eps=240, min_samples=0).fit(X)

        # Find the largest label assigned + 1, that's the number of dice found
        num_dice = max(clustering.labels_) + 1

        dice = []

        # Calculate centroid of each dice, the average between all a dice's dots
        for i in range(num_dice):
            X_dice = X[clustering.labels_ == i]

            centroid_dice = np.mean(X_dice, axis=0)

            dice.append([len(X_dice), *centroid_dice])

        return dice

    else:
        return []


def overlay_info(frame, dice, blobs):
    # Overlay blobs
    for b in blobs:
        pos = b.pt
        r = b.size / 2

        cv2.circle(frame, (int(pos[0]), int(pos[1])),
                   int(r), (255, 0, 0), 2)

    # Overlay dice number
    for d in dice:
        # Get textsize for text centering
        textsize = cv2.getTextSize(
            str(d[0]), cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]

        cv2.putText(frame, str(d[0]),
                    (int(d[1] - textsize[0] / 2),
                     int(d[2] + textsize[1] / 2)),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)


def resizeDown(img):
    scale_percent = 30  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def photos_check():
    # listOfFiles = ["w10.jpg", "w2.jpg", "w3.jpg", "w4.jpg", "w5.jpg", "w6.jpg", "w7.jpg", "w8.jpg", "w9.jpg", "w1.jpg"]
    # listOfFiles = ["b1.jpg", "b2.jpg", "b3.jpg", "b4.jpg", "b5.jpg", "b6.jpg", "b7.jpg", "b8.jpg", "b9.jpg", "b10.jpg"]
    # listOfFiles = ["y10.jpg", "y2.jpg", "y3.jpg", "y4.jpg", "y5.jpg", "y6.jpg", "y7.jpg", "y8.jpg", "y9.jpg", "y1.jpg"]
    listOfFiles = ["q1.jpg", "q2.jpg", "q3.jpg", "q4.jpg", "q5.jpg", "q6.jpg", "q7.jpg", "q8.jpg", "q9.jpg", "q10.jpg"]

    files = []

    for nameOfFile in listOfFiles:
        files.append(cv2.imread(nameOfFile))

    # while(True):
    for img in files:
        yellow_frame = get_yellow(img)
        blobs = get_blobs(yellow_frame)
        dice = get_dice_from_blobs(blobs)
        out_frame = overlay_info(img, dice, blobs)
        cv2.imshow("Resized image", resizeDown(img))
        time.sleep(4)
        # cv2.imwrite('no_area.jpg', img)
        res = cv2.waitKey(1)
        # # Stop if the user presses "q"
        if res & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def video_check():
    cap = cv2.VideoCapture(0)
    while (True):
        # Grab the latest image from the video feed
        ret, frame = cap.read()

        # We'll define these later
        yellow_frame = get_yellow(frame)
        blobs = get_blobs(yellow_frame)
        dice = get_dice_from_blobs(blobs)
        out_frame = overlay_info(frame, dice, blobs)

        cv2.imshow("frame", frame)
        res = cv2.waitKey(1)

        # Stop if the user presses "q"
        if res & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

photos_check()
# video_check()
