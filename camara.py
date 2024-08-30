import cv2

# Carga el clasificador de Haar para la detección de rostros
face_cascade = cv2.CascadeClassifier('detectar/haarcascade_frontalface_default.xml')

#esto es ejemplo de cambio
cap = cv2.VideoCapture(1)

while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('img', img)
    k = cv2.waitKey(30)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()