import cv2
import time
import imutils
from imutils.video import VideoStream
import os
import zbar
import random

IDS = {'id123': 'Jeff Lord'}

vs = VideoStream(src=0).start()
cam = cv2.VideoCapture(0)
time.sleep(2.0)

#! HAAR
haar_face_cascade = cv2.CascadeClassifier(
    os.path.join("haar_frontalface_default.xml"))
haar_eye_cascade = cv2.CascadeClassifier(
    os.path.join("haar_eye.xml"))

#! HOG
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#! DNN


_face, _bar = False, False
while(True):
    if(_bar and _face):
        time.sleep(5.0)
    _face, _bar = False, False
    s, frame = cam.read()
    # frame = cv2.resize(frame, (300, 300))
    frame_color = frame
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#! FACES HAAR
    faces = haar_face_cascade.detectMultiScale(
        frame_gray, scaleFactor=1.3, minNeighbors=5)

    if(len(faces) > 0):
        _face = True
    for (x, y, w, h) in faces:
        color = (random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255))
        frame_color = cv2.rectangle(
            frame_color, (x, y), (x+w, y+h), color, 5)
        frame_color = cv2.putText(
            frame_color, "Face", (x, y), cv2.FONT_HERSHEY_PLAIN, 5, color, 5)

#! FACES HOG
    # (faces, weights) = hog.detectMultiScale(
    #     frame_gray, winStride=(8, 8), padding=(16, 16), scale=1.05, useMeanshiftGrouping=1)
    # # faces = hog.compute(frame_gray)
    # if(len(faces) > 0):
    #     print("Faces = " + str(len(faces)))
    #     _face = True
    # for (x, y, w, h) in faces:
    #     color = (random.randint(0, 255), random.randint(
    #         0, 255), random.randint(0, 255))
    #     frame_color = cv2.rectangle(
    #         frame_color, (x, y), (x+w, y+h), color, 5)
    #     frame_color = cv2.putText(
    #         frame_color, "Face", (x, y), cv2.FONT_HERSHEY_PLAIN, 5, color, 5)

#! BARCODES
    barScanner = zbar.Scanner()
    barRes = barScanner.scan(frame_gray)
    if(len(barRes) > 0):
        _bar = True
    for res in barRes:
        # print(res)
        print("Type:\t" + str(res.type))
        print("Data:\t" + str(res.data))
        print("Decoded:\t" + str(res.data.decode('ascii')))
        print("Quality:\t" + str(res.quality))
        print("Position:\t" + str(res.position))

        if(res.type == "QR-Code"):
            color = (random.randint(0, 255), random.randint(
                0, 255), random.randint(0, 255))
            # frame_color = cv2.rectangle(
            #     frame_color, res.position[0], res.position[2], (255, 0, 0), 2)
            frame_color = cv2.line(
                frame_color, res.position[0], res.position[1], color, 5)
            frame_color = cv2.line(
                frame_color, res.position[1], res.position[2], color, 5)
            frame_color = cv2.line(
                frame_color, res.position[2], res.position[3], color, 5)
            frame_color = cv2.line(
                frame_color, res.position[3], res.position[0], color, 5)
            text_decode = str(res.data.decode('ascii'))
            frame_color = cv2.putText(
                frame_color, text_decode + " = " + IDS[text_decode], (res.position[0][0], res.position[0][1]-20), cv2.FONT_HERSHEY_PLAIN, 5, color, 5)

#! EYES
        # eyes = haar_eye_cascade.detectMultiScale(frame_gray)
        # for(ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(
        #         frame_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

        #? Do things when there are faces,
        #? Like send it to another analysis script, classifier, etc.
        # if(len(faces) > 0):
    cv2.imshow("output", frame_color)
    # frame = imutils.resize(frame, width=300)

    key = cv2.waitKey(1) & 0xFF
    if(key == ord("q")):
        break


cv2.destroyAllWindows()
vs.stop()


# def RandomColor():
#     return (random.randint(0, 255), 0, 0)
