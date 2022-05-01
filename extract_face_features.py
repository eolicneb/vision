from imutils import paths
import face_recognition
import pickle
import cv2
from pathlib import Path


def cascade_from_file(xml_file: str):
    casc_path_face = Path(cv2.__file__).parent / "data" / xml_file
    face_cascade = cv2.CascadeClassifier(casc_path_face.as_posix())
    return face_cascade


def find_encodings(base_path: Path, encodings_file: str, cascade_file="haarcascade_frontalface_alt2.xml"):
    image_paths = base_path / "targets"
    known_encodings = []
    known_names = []

    for image_path in paths.list_images(image_paths.as_posix()):
        name = Path(image_path).stem
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_cascade = cascade_from_file(cascade_file)
        # boxes = face_cascade.detectMultiScale(gray_image,
        #                                       scaleFactor=1.1,
        #                                       minNeighbors=5,
        #                                       minSize=(40, 40),
        #                                       flags=cv2.CASCADE_SCALE_IMAGE)

        # locate faces
        boxes = face_recognition.face_locations(gray_image, model="hog")

        # facial embedding
        encodings = face_recognition.face_encodings(rgb_image, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(name)

    for name in set(known_names):
        print(name)

    data = {"encodings": known_encodings, "names": known_names}
    (base_path / encodings_file).write_bytes(pickle.dumps(data))
