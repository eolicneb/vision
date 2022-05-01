from imutils import paths
import face_recognition
import pickle
import cv2
from pathlib import Path


def cascade_from_file(xml_file: str):
    casc_path_face = Path(cv2.__file__).parent / "data" / xml_file
    face_cascade = cv2.CascadeClassifier(casc_path_face.as_posix())
    return face_cascade


def show(image, boxes, labels):
    for (x, y, w, h), name in zip(boxes, labels):
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, name, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.imshow("Face", image)
    cv2.waitKey(0)


def find_encodings(base_path: Path, encodings_file: str,
                   cascade_file=None):
    image_paths = base_path / "targets"
    known_encodings = []
    known_names = []

    for image_path in paths.list_images(image_paths.as_posix()):
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # locate faces
        if cascade_file is None:
            boxes = face_recognition.face_locations(rgb_image, model="hog")
        else:
            face_cascade = cascade_from_file(cascade_file)
            boxes = face_cascade.detectMultiScale(gray_image,
                                                  scaleFactor=1.1,
                                                  minNeighbors=5,
                                                  minSize=(60, 60),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)

        # facial embedding
        encodings = face_recognition.face_encodings(rgb_image, boxes)

        name = Path(image_path).stem.split("_")[0]
        print(f"{len(encodings)} encodings for {name}")
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(name)

        # show(image, boxes, [name for _ in boxes])

    data = {"encodings": known_encodings, "names": known_names}
    (base_path / encodings_file).write_bytes(pickle.dumps(data))
