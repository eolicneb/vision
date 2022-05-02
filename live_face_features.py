import face_recognition
import imutils
import pickle
import time
import cv2
import os
from pathlib import Path

from extract_face_features import find_encodings
from apply_face_features import analyse_image, retrieve_data

xml_file = "haarcascade_frontalface_alt2.xml"
# xml_file = "haarcascade_frontalcatface.xml"
# xml_file = "haarcascade_frontalcatface_extended.xml"

casc_path_face = Path(cv2.__file__).parent / "data" / xml_file
face_cascade = cv2.CascadeClassifier(casc_path_face.as_posix())


def inspect_feed(data):
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        analyse_image(frame, data)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    base_path = Path("./famibia")
    force_new_encodings = True

    data = retrieve_data(['famibia', 'avengers'], force_new_encodings)

    print(data['names'])

    inspect_feed(data)
