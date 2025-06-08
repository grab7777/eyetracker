import cv2
import os

debug = True
image_names = [image for image in os.listdir("./images")]

def search_pupils(roi: cv2.Mat) -> tuple[int, int]:
    debug = False
    # Search circles in region of interest by doing Hough Transform
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (13, 13), 0)
    highConstrast = cv2.convertScaleAbs(blurred, alpha=2.0, beta=0)
    if debug:
        debug_image = roi.copy()
        cv2.imshow("High", highConstrast)
    _, threshold = cv2.threshold(highConstrast, 44, 255, cv2.THRESH_BINARY_INV)
    erosion_size = 10
    dilate_size = 8
    threshold = cv2.erode(threshold, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size)))
    threshold = cv2.dilate(threshold, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size)))
    contours = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if debug:
        cv2.drawContours(debug_image, contours[0], -1, (255, 255, 255), 3)
        cv2.imshow("Contours", debug_image)
    # get largest contour
    if len(contours) == 0 or len(contours[0]) == 0:
        print("No contours found")
        return None
    largest_contour = max(contours[0], key=cv2.contourArea)
    center_of_mass = cv2.moments(largest_contour)
    if center_of_mass["m00"] == 0:
        print("No pupil found")
        return None
    cx = int(center_of_mass["m10"] / center_of_mass["m00"])
    cy = int(center_of_mass["m01"] / center_of_mass["m00"])
    if debug:
        cv2.circle(debug_image, (cx, cy), 5, (0, 0, 255), -1)
        cv2.imshow("center of mass", debug_image)
        cv2.waitKey(0)
    return (cx, cy)

def detect_eyes(image_name: str) -> list[tuple[int, int, int, int]]:
    img = cv2.imread(f"./images/{image_name}")
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
    # eye_cascade = cv2.CascadeClassifier("haarcascade_lefteye_2splits.xml")
    for ex, ey, ew, eh in eye_cascade.detectMultiScale(
        img,
        scaleFactor=1.6,
        minNeighbors=2,
        minSize=(200, 200),
        flags=cv2.CASCADE_SCALE_IMAGE,
    ):
        center = (ex + ew // 2, ey + eh // 2)
        regionOfInterest = img.copy()[ey : ey + eh, ex : ex + ew]
        if debug:
            cv2.circle(img, center, 5, (0, 0, 255), 6)
            cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 6)
            print(f"Eye detected at: ({ex}, {ey}), ({ex +ew}, {ey+ eh})")
            print(f"Center of eye: {center}")
        pupil = search_pupils(regionOfInterest)
        if pupil is None:
            continue
        cx, cy = pupil
        if debug:
            cv2.circle(img, (cx + ex, cy + ey), 5, (0, 0, 255), 10)
            print(f"Pupil found at: ({cx}, {cy}) in region of interest")
        else:
            print("No pupil found in region of interest")
    cv2.namedWindow("image", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("image", img)
    cv2.waitKey(0)

for image in image_names:
    detect_eyes(image)
    cv2.destroyAllWindows()

