import cv2
import os
import dlib
import numpy as np

debug = False
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
    threshold = cv2.erode(
        threshold,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size)),
    )
    threshold = cv2.dilate(
        threshold,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size)),
    )
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


# for image in image_names:
# detect_eyes(image)
# cv2.destroyAllWindows()


def midpoint(p1, p2) -> tuple[int, int]:
    return (int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2))


def midpoint_tuple(p1: tuple[int, int], p2: tuple[int, int]) -> tuple[int, int]:
    return (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))


base_left = (1377, 2080)
base_right = (2032, 2076)
base_mouth = (1683, 3046)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def detect_eyes_with_dlib(image_name: str) -> None:
    img = cv2.imread(f"./images/{image_name}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        left_hor_start = (landmarks.part(36).x, landmarks.part(36).y)
        left_hor_end = (landmarks.part(39).x, landmarks.part(39).y)
        right_hor_start = landmarks.part(42).x, landmarks.part(42).y
        right_hor_end = landmarks.part(45).x, landmarks.part(45).y
        if debug:
            left_vert_start = midpoint(landmarks.part(37), landmarks.part(38))
            left_vert_end = midpoint(landmarks.part(41), landmarks.part(40))
            right_vert_start = midpoint(landmarks.part(43), landmarks.part(44))
            right_vert_end = midpoint(landmarks.part(47), landmarks.part(46))
            left_hor = cv2.line(img, left_hor_start, left_hor_end, (0, 255, 0), 2)
            right_hor = cv2.line(img, right_hor_start, right_hor_end, (0, 255, 0), 2)
        mouth = (landmarks.part(57).x, landmarks.part(57).y)
        center_left = midpoint_tuple(left_hor_start, left_hor_end)
        center_right = midpoint_tuple(right_hor_start, right_hor_end)
        print(f"Left eye center:  {center_left}")
        print(f"Right eye center: {center_right}")
        print(f"mouth: {mouth}")
        if debug:
            cv2.circle(img, mouth, 5, (0, 255, 0), 5)
            cv2.circle(img, center_left, 5, (0, 0, 255), 5)
            cv2.circle(img, center_right, 5, (0, 0, 255), 5)

        img_x = img.shape[1]
        img_y = img.shape[0]
        print(f"Image shape: height {img_y} px, width {img_x} px")

        src_pts = np.array([center_left, center_right, mouth], dtype=np.float32)
        dst_pts = np.array([base_left, base_right, base_mouth], dtype=np.float32)

        M = cv2.getAffineTransform(
            src_pts,
            dst_pts,
        )

        print(M)
        img = cv2.warpAffine(img, M, (img_x, img_y))
    if debug:
        cv2.imshow("dlib eyes", img)
        cv2.waitKey(0)
    else:
        if not os.path.exists("./output"):
            os.makedirs("./output")
        cv2.imwrite(f"./output/{image_name}", img)
        print(f"Processed {image_name} and saved to output folder.")


for image in image_names:
    detect_eyes_with_dlib(image)
