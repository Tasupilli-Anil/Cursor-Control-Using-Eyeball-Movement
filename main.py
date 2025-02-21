import cv2

# Load the Haar Cascade for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Region of interest (ROI) for eye detection within the face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        # Detect eyes within the face
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # Draw rectangle around each eye
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Eye Detection', frame)

    # Exit on pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close the window
cap.release()
cv2.destroyAllWindows()
