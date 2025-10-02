import cv2
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0)

# K-Means parameters
K = 1  # we only want the most dominant color
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define central rectangle
    h, w, _ = hsv.shape
    rect_size = 100  # square size
    cx, cy = w // 2, h // 2
    x1, y1 = cx - rect_size//2, cy - rect_size//2
    x2, y2 = cx + rect_size//2, cy + rect_size//2

    # Draw rectangle for visualization
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Crop the central rectangle
    roi = hsv[y1:y2, x1:x2]

    # Reshape for K-Means (pixels as rows)
    pixels = roi.reshape((-1, 3)).astype(np.float32)

    # Apply K-Means
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # The dominant color
    dominant_hsv = centers[0].astype(int)
    dominant_bgr = cv2.cvtColor(np.uint8([[dominant_hsv]]), cv2.COLOR_HSV2BGR)[0][0]

    # Show the dominant color as a rectangle on the frame
    cv2.rectangle(frame, (10, 10), (110, 110), tuple(int(c) for c in dominant_bgr), -1)

    # Print HSV values
    print("Dominant HSV:", dominant_hsv)

    cv2.imshow("Dominant Color Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

