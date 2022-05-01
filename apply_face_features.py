import face_recognition
import pickle
import cv2
from sklearn.svm import SVC
from pathlib import Path

from extract_face_features import find_encodings

TOLERANCE = 0.6

xml_file = "haarcascade_frontalface_alt2.xml"
# xml_file = "haarcascade_frontalcatface.xml"
# xml_file = "haarcascade_frontalcatface_extended.xml"

casc_path_face = Path(cv2.__file__).parent / "data" / xml_file
face_cascade = cv2.CascadeClassifier(casc_path_face.as_posix())


clf_ = None


def classifier(data):
    global clf_
    if clf_ is None:
        clf_ = SVC(kernel='linear')
        clf_.fit(data['encodings'], data['names'])
    return clf_


def sub_image(image, box):
    x, y, w, h = box
    return image[y:y+h, x:x+w]


def identify(image, data) -> tuple[str, dict]:
    name = "Unknown"
    counts = {}

    encodings = face_recognition.face_encodings(image)
    if encodings:
        for encoding in encodings:
            distances = face_recognition.face_distance(data['encodings'], encoding)
            min_arg = distances.argmin()
            if distances[min_arg] > TOLERANCE:
                continue
            name = data['names'][min_arg]
            counts[name] = counts.get(name, []) + [distances[min_arg]]

        if counts:
            for i_name in counts.keys():
                counts[i_name] = 1 / sum(1/d for d in counts[i_name])
            name = min(counts, key=counts.get)

    return name, counts


def analyse_image(image_file: str, data):
    image = cv2.imread(image_file)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray_image,
    #                                       scaleFactor=1.1,
    #                                       minNeighbors=5,
    #                                       minSize=(30, 30),
    #                                       flags=cv2.CASCADE_SCALE_IMAGE)

    # locate faces
    faces = face_recognition.face_locations(gray_image, model="hog")

    names, counters = [], []

    # encodings = face_recognition.face_encodings(rgb_image, faces)
    # print(f"Found {len(encodings)} encodings", end=" ")
    #
    # if encodings:
    #     for encoding in encodings:
    #         distances = face_recognition.face_distance(data['encodings'], encoding)
    #         best_match = distances.argmin() if distances.min() <= TOLERANCE else None
    #
    #         name = "Unknown"
    #         counts = {}
    #         if best_match is not None:
    #             for i in distances.argsort()[:3]:
    #                 name = data['names'][i]
    #                 counts[name] = distances[i]
    #
    #             name = data['names'][best_match]
    #
    #         print(name, end=" ")
    #         print(classifier(data).predict(encoding.reshape(1, -1)))
    #         names.append(name)
    #         counters.append(counts)

    for box in faces:
        name, counts = identify(sub_image(rgb_image, box), data)
        names.append(name)
        counters.append(counts)

    for (x, y, w, h), name, counts in zip(faces, names, counters):
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, name, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        for i, (c_name, count) in enumerate(counts.items()):
            cv2.putText(image, f"{c_name}: {count}", (x, y+h+11+i*11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    cv2.imshow(image_file, image)


if __name__ == "__main__":

    base_path = Path("./avengers")
    force_new_encodings = True

    if not (base_path / "face_enc").exists() or force_new_encodings:
        find_encodings(base_path, "face_enc", xml_file)
    data = pickle.loads((base_path / "face_enc").read_bytes())

    print(data['names'])

    for i, image_path in enumerate((base_path / "subjects").iterdir()):
        if i != 1:
            continue
        analyse_image(image_path.as_posix(), data)
    cv2.waitKey(0)
