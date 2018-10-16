import cv2
import argparse
import numpy as np
import os
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time

# Different models will have different outputs from the prediction/ net.forward step. this needs to be known to be able to handle response correctly
# base_data_path = os.path.join("../../../", "_test_data")
# base_drive = os.path.join("..", "..", "..", "..", "Google Drive", "Drive",
#                           "Work", "SAP", "test_data", "models")
base_path = "/Users/i849438/Google Drive/Drive/Work/SAP/test_data/models/"
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=False,
                default=os.path.join(base_path, "MobileNetSSD_deploy.caffemodel"))
ap.add_argument("-p", "--prototxt", required=False,
                default=os.path.join(base_path, "MobileNetSSD_deploy.prototxt"))
ap.add_argument("-c", "--confidence", type=float, default=0.2)
ap.add_argument("-l", "--labels")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(0)
temp, preimage = cap.read()

if args["labels"] is None:
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
else:
    rows = open(args["labels"]).read().strip().split("\n")
    CLASSES = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
print(CLASSES)

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()
i = 0
while(True):
    # temp, image = cap.read()
    # image = cv2.imread(image)

    frame = vs.read()
    originalFrame = frame
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(
        frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    # print(detections)
    # detections.shape[2] corresponds to how many objects were detected in this model
    for i in np.arange(0, detections.shape[2]):
        # print(detections.shape[2])
        confidence = detections[0, 0, i, 2]
        if confidence > args["confidence"]:
            idx = int(detections[0, 0, i, 1])
            # print(idx)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(frame, (startX, startY),
                          (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF
    if(key == ord("q")):
        break
    fps.update()
    time.sleep(0.05)
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
vs.stop()
