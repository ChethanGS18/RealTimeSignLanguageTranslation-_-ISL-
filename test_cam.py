import cv2

print("Attempting to open camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
else:
    print("SUCCESS: Camera opened! Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame.")
            break
        cv2.imshow('Test Camera', frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()