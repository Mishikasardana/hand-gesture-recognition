import numpy as np
import cv2
import math
from sklearn.metrics import classification_report, average_precision_score

# Initialize lists to store results
y_true = []
y_pred = []

# Define gesture labels
gesture_labels = ["ONE", "TWO", "THREE", "FOUR", "FIVE"]

#camera
capture = cv2.VideoCapture(0)

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) 

    height, width = frame.shape[:2]
    x1, y1 = width // 2 - 100, height // 2 - 100
    x2, y2 = width // 2 + 100, height // 2 + 100
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    crop_image = frame[y1:y2, x1:x2]

    # Gaussian blur
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

    #BGR to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    #skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create skin mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Morphological transformations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    # Threshold the image
    ret, thresh = cv2.threshold(mask, 127, 255, 0)

    #contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros(crop_image.shape, np.uint8)

    if contours:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            hull = cv2.convexHull(contour)
            cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 2)

            hull_indices = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull_indices)

            count_defects = 0
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    far = tuple(contour[f][0])

                    a = math.dist(end, start)
                    b = math.dist(far, start)
                    c = math.dist(end, far)

                    angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * (180 / math.pi)
                    if angle <= 90:
                        count_defects += 1
                        cv2.circle(crop_image, far, 5, [0, 0, 255], -1)
                    cv2.line(crop_image, start, end, [0, 255, 0], 2)

            # Label based on number of fingers
            text = ["ONE", "TWO", "THREE", "FOUR", "FIVE"]
            if count_defects < 5:
                cv2.putText(frame, text[count_defects], (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            else:
                cv2.putText(frame, "HAND", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            current_truth = "TWO"

            # Append true and predicted
            y_true.append(current_truth)
            y_pred.append(text[count_defects] if count_defects < 5 else "HAND")

    cv2.imshow("Gesture", frame)
    cv2.imshow("Threshold", thresh)
    cv2.imshow("Contours", np.hstack((drawing, crop_image)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Classification Evaluation Report:")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, average='macro', zero_division=0))
print("Recall:", recall_score(y_true, y_pred, average='macro', zero_division=0))
print("F1 Score:", f1_score(y_true, y_pred, average='macro', zero_division=0))

# For mAP — one-vs-rest binarization
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
y_true_bin = lb.fit_transform(y_true)
y_pred_bin = lb.transform(y_pred)

try:
    mAP = average_precision_score(y_true_bin, y_pred_bin, average='macro')
    print("Mean Average Precision (mAP):", mAP)
except:
    print("Insufficient data for mAP (need at least 2 classes).")
    