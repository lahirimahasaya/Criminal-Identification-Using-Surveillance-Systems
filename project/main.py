while True:
    def criminal_detection():
        import argparse
        import cv2
        import dlib
        import imutils
        from imutils import face_utils
        from imutils.face_utils import FaceAligner
        from imutils.face_utils import rect_to_bb
        import math
        from sklearn import neighbors
        import os
        import os.path
        import pickle
        from PIL import Image, ImageDraw
        import face_recognition
        from face_recognition.face_recognition_cli import image_files_in_folder
        global k, names

        mydir = "C:/Users/HP/PycharmProjects/Projects/project/knn_test"
        mydir1 = "C:/Users/HP/PycharmProjects/Projects/project/outputs"
        filelist = [f for f in os.listdir(mydir) if f.endswith(".jpg")]
        for f in filelist:
            os.remove(os.path.join(mydir, f))
        filelist1 = [f for f in os.listdir(mydir1) if f.endswith(".jpg")]
        for f in filelist1:
            os.remove(os.path.join(mydir1, f))

        ap = argparse.ArgumentParser()
        ap.add_argument('--output-dir', type=str, default='outputs/', help='path to the output directory')
        args = ap.parse_args()

        cap = cv2.VideoCapture(0)

        for i in range(0, 100):
            i = i + 1
            ret, frame = cap.read()
            cv2.imwrite("C:/Users/HP/PycharmProjects/Projects/project/inputs/input" + ".jpg", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        frame = cv2.imread("C:/Users/HP/PycharmProjects/Projects/project/inputs/input.jpg")
        image = imutils.resize(frame, width=960)

        image = cv2.imread('C:/Users/HP/PycharmProjects/Projects/project/inputs/input.jpg')
        # image = cv2.imread('C:/Users/HP/PycharmProjects/Projects/project/example_13.jpg')

        detector = dlib.cnn_face_detection_model_v1('C:/Users/HP/PycharmProjects/Projects/project/CNN_face_detection.dat')

        # apply face detection (cnn)
        faces_cnn = detector(image, 1)

        # loop over detected faces
        count1 = 0
        for face in faces_cnn:
            x = face.rect.left()
            y = face.rect.top()
            w = face.rect.right() - x + 30
            h = face.rect.bottom() - y + 30
            count1 = count1 + 1
            test = imutils.resize(image[y - 20:y + h, x - 20:x + w], width=256)
            cv2.imwrite("C:/Users/HP/PycharmProjects/Projects/project/knn_test/test" + str(count1) + ".jpg", test)

        img_height, img_width = image.shape[:2]

        cv2.waitKey()
        cv2.destroyAllWindows()

        ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

        def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
            if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
                raise Exception("Invalid image path: {}".format(X_img_path))

            if knn_clf is None and model_path is None:
                raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

            if knn_clf is None:
                with open(model_path, 'rb') as f:
                    knn_clf = pickle.load(f)

            X_img = face_recognition.load_image_file(X_img_path)
            X_face_locations = face_recognition.face_locations(X_img)

            if len(X_face_locations) == 0:
                return []

            faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

            closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
            are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

            return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
                    zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

        def show_prediction_labels_on_image(img_path, predictions):
            pil_image = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(pil_image)

            for name, (top, right, bottom, left) in predictions:
                draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

                name = name.encode("UTF-8")

                text_width, text_height = draw.textsize(name)
                draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255),
                               outline=(0, 0, 255))
                draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
                pil_image.save('C:/Users/HP/PycharmProjects/Projects/project/outputs/' + str(name) + '.jpg')

            del draw

        names = []

        if __name__ == "__main__":

            count = 0
            for image_file in os.listdir("knn_test"):
                full_file_path = os.path.join("knn_test", image_file)

                print("Looking for faces in {}".format(image_file))
                predictions = predict(full_file_path, model_path="trained_knn_model.clf")

                for name, (top, right, bottom, left) in predictions:
                    print("- Found {} ".format(name))
                    names.append(str(name))
                    if str(name) == 'unknown':
                        count = count + 1
                    else:
                        count = count - 10000

                show_prediction_labels_on_image(os.path.join("knn_test", image_file), predictions)
            k = count
        return k, names


    criminal_detection()

    if k >= 0:
        back: criminal_detection()
    else:

        import yagmail
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.support.ui import WebDriverWait
        import time

        start = time.time()


        def getLocation():
            options = webdriver.ChromeOptions()
            options.binary_location = "C:/Program Files/Google/Chrome/Application/chrome.exe"
            driver = webdriver.Chrome(executable_path='./chromedriver.exe', chrome_options=options)
            options = Options()
            options.add_argument("--use-fake-ui-for-media-stream")
            timeout = 20
            driver.get("https://mycurrentlocation.net/")
            wait = WebDriverWait(driver, timeout)
            time.sleep(3)
            longitude = driver.find_elements_by_xpath('//*[@id="longitude"]')
            longitude = [x.text for x in longitude]
            longitude = str(longitude[0])
            latitude = driver.find_elements_by_xpath('//*[@id="latitude"]')
            latitude = [x.text for x in latitude]
            latitude = str(latitude[0])
            driver.quit()
            return latitude, longitude

        loca = str
        cood = int
        cood = getLocation()
        loca = str(cood)


        from geopy.geocoders import Nominatim

        geolocator = Nominatim(user_agent="geoapiExercises")


        def city_state_country(coord):
            location = geolocator.reverse(coord, exactly_one=True)
            address = location.raw['address']
            city = address.get('city', '')
            state = address.get('state', '')
            country = address.get('country', '')
            place = [str(city), str(state), str(country)]
            return place


        place = city_state_country(cood)

        yag = yagmail.SMTP(user='lahirimahasaya628@gmail.com', password='Mahasaya@12')
        subject = "Criminal Nearby Alert"
        contents = ['Detected location  latitude, longitude, city, state and country:', str(loca), str(place),  'Criminal names:', str(names)]
        attachments = ['C:/Users/HP/PycharmProjects/Projects/project/inputs/input' + '.jpg']
        yag.send('bestha.beec.16@acharya.ac.in', subject, contents, attachments)
        yag.close()
        end = time.time()
        print("Run time : ", format(end - start, '.2f'))
