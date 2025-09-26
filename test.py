import cv2

# Force DirectShow backend with CAP_DSHOW (700)
cap = cv2.VideoCapture(2, cv2.CAP_MSMF)
  # change 1 → 0, 2, etc if needed

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("DroidCam USB", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
