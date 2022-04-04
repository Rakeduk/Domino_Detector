import cv2
import numpy as np
import time
from sklearn import cluster

params = cv2.SimpleBlobDetector_Params()

# params.filterByInertia
params.minInertiaRatio = 0.4

# params.filterByConvexity
params.minConvexity = 0.5

# params.filterByArea
params.minArea = 800


detector = cv2.SimpleBlobDetector_create(params)


def get_yellow(frame):
    frame_blurred = cv2.medianBlur(frame, 11)
    frame_hsv = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2HSV)

    # Setting boundaries of yellow color
    lower_yellow = np.array([10, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    # Extracting yellow from frame
    mask = cv2.inRange(frame_hsv, lower_yellow, upper_yellow)

    # Reversing colors as detector work with black dots
    rev_image = cv2.bitwise_not(mask)

    return rev_image


def get_blobs(frame):
    blobs = detector.detect(frame)

    return blobs


def get_domino_from_blobs(blobs):
    # Get centroids
    tab_with_blobs = []
    for b in blobs:
        pos = b.pt

        if pos != None:
            tab_with_blobs.append(pos)

    tab_with_blobs = np.asarray(tab_with_blobs)

    if len(tab_with_blobs) > 0:
        clustering = cluster.DBSCAN(eps=240, min_samples=0).fit(tab_with_blobs)

        num_domino = max(clustering.labels_) + 1

        domino = []

        for i in range(num_domino):
            x_domino = tab_with_blobs[clustering.labels_ == i]

            centroid_domino = np.mean(x_domino, axis=0)

            domino.append([len(x_domino), *centroid_domino])

        return domino

    else:
        return []


def overlay_info(frame, domino, blobs):
    # Overlay
    for b in blobs:
        pos = b.pt
        r = b.size / 2

        cv2.circle(frame, (int(pos[0]), int(pos[1])),
                   int(r), (255, 255, 0), 2)

    for d in domino:
        textsize = cv2.getTextSize(
            str(d[0]), cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]

        cv2.putText(frame, str(d[0]),
                    (int(d[1] - textsize[0] / 2),
                     int(d[2] + textsize[1] / 2)),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 125), 2)


def resize_down(img):
    scale_percent = 30  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def photos_check():
    # listOfFiles = ["w10.jpg", "w2.jpg", "w3.jpg", "w4.jpg", "w5.jpg",
    #                   "w6.jpg", "w7.jpg", "w8.jpg", "w9.jpg", "w1.jpg"]
    # listOfFiles = ["b1.jpg", "b2.jpg", "b3.jpg", "b4.jpg", "b5.jpg",
    #                   "b6.jpg", "b7.jpg", "b8.jpg", "b9.jpg", "b10.jpg"]
    # listOfFiles = ["y10.jpg", "y2.jpg", "y3.jpg", "y4.jpg", "y5.jpg",
    #                     y6.jpg", "y7.jpg", "y8.jpg", "y9.jpg", "y1.jpg"]
    list_of_files = ["q7.jpg", "w4.jpg", "b5.jpg", "y8.jpg", "b4.jpg"]

    files = []

    for nameOfFile in list_of_files:
        files.append(cv2.imread(nameOfFile))

    # while(True):
    for img in files:
        yellow_frame = get_yellow(img)
        blobs = get_blobs(yellow_frame)
        domino = get_domino_from_blobs(blobs)
        overlay_info(img, domino, blobs)
        cv2.imshow("Resized image", resize_down(img))

        time.sleep(4)
        res = cv2.waitKey(1)

        # Stop if "q"
        if res & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


photos_check()
