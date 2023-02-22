import cv2

# Load the image
image = cv2.imread('face.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create an LBP object
lbp = cv2.face.LBPHFaceRecognizer_create()

# Detect faces using LBP
faces = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_rects = faces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around the detected faces
for (x, y, w, h) in face_rects:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Show the result
cv2.imshow("Result", image)
cv2.waitKey(0)
