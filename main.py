import sys
import cv2

# Get user supplied values
imagePath = sys.argv[1]
cascPath = sys.argv[2]

# Create the cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.01,
    minNeighbors=6,
    minSize=(34, 34),
    flags=cv2.CASCADE_SCALE_IMAGE
)

print(f"Found {len(faces)} faces!")

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
