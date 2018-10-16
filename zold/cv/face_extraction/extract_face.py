import cv2
import argparse
import os

parser = argparse.ArgumentParser(description='Argument parser.')
parser.add_argument('--dir', '-d', help='Path to root directory')
parser.add_argument('--output', '-o')
args = parser.parse_args()
file_path, file_name = os.path.split(__file__)


haar_face_cascade = cv2.CascadeClassifier(
    os.path.join(file_path, "haar_frontalface_default.xml"))
#? this gets every file in all subdirectories

for path, subdirs, files in os.walk(args.dir):
    for name in files:
        file = os.path.join(path, name)
        if(os.path.isfile(file) and file.endswith(".jpg")):
            print("[INFO] Current file: " + str(file))
            label = os.path.basename(os.path.normpath(path))
            image = cv2.imread(file)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = haar_face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.4,
                minNeighbors=5
            )
            print("[INFO] Faces found: " + str(len(faces)))
            i = 0
            for(x, y, w, h) in faces:
                crop = image[y:y+h, x:x+w]
                cv2.imshow("cropped", crop)
                # cv2.waitKey(0)
                # cv2.imwrite("test.jpg", crop)
                save_base = os.path.join(args.output, label)
                if(not os.path.isdir(save_base)):
                    os.makedirs(save_base)
                save_path = os.path.join(
                    save_base, "crop_" + str(i)+"_" + name)
                cv2.imwrite(save_path, crop)
                i += 1
                print("[INFO] Saved face.")
