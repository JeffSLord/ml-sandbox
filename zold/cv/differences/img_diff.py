from skimage.measure import compare_ssim
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()q
ap.add_argument("-f", "--first", required=True)
ap.add_argument("-s", "--second", required=True)
args = ap.parse_args()

im1 = cv2.imread(args.first)
im2 = cv2.imread(args.second)
# im1 = cv2.resize(im1, (300, 300))
# im2 = cv2.resize(im2, (300, 300))

gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

(score, diff) = compare_ssim(gray1, gray2, full=True)
diff = (diff * 255).astype("uint8")
print("[INFO] SSIM: {}".format(score))


thresh = cv2.threshold(diff, 0, 255,
                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]


for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(im1, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow("Original", im1)
cv2.imshow("Modified", im2)
cv2.imshow("Diff", diff)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)
