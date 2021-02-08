import cv2
import os
import numpy as np
import imutils
import glob

class FaceFinder():
    def __init__(self, target_name):
        self.face_cascade = cv2.CascadeClassifier('./classifiers/haar_frontal_face.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.target_name = target_name

        if target_name[0] != "." and target_name[0] != "/":
            self.results_folder_path = './' + target_name
        else:
            self.results_folder_path = target_name
        
        recognizer_data_files = glob.glob(os.path.join(self.results_folder_path, "*.yml"))
        if len(recognizer_data_files) > 0:
            self.recognizer.read('{}/target.yml'.format(self.results_folder_path))

    def save_face_from_data(self, out_filename, data):
        cc = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        img_np = imutils.resize(cc, width=200)

        faces = self.face_cascade.detectMultiScale(img_np, scaleFactor=1.1,  minNeighbors=4, minSize=(30,30))
        for (x,y,w,h) in faces:
            cv2.imwrite((self.results_folder_path + "/" + out_filename), img_np[y:y+h, x:x+w])

    def learn_target(self):
        image_paths = glob.glob(os.path.join(self.results_folder_path, "*.jpg"))
        ids = []
        samples = []
        for image_path in image_paths:
            gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img_numpy = np.array(imutils.resize(gray_image, width=200), 'uint8')
            faces = self.face_cascade.detectMultiScale(img_numpy, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))
            for (x,y,w,h) in faces:
                samples.append(img_numpy[y:y+h,x:x+w])
                ids.append(0)
        self.recognizer.train(samples, np.array(ids))
        self.recognizer.write('{}/target.yml'.format(self.results_folder_path))

    def scan_pic_for(self, data):
        orig = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        cc = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        img_np = imutils.resize(cc, width=200)

        faces = self.face_cascade.detectMultiScale(cc)
        max_conf = 0
        for(x, y, w, h) in faces:
            cv2.rectangle(orig, (x, y), (x+w, y+h), (0, 225, 0), 2)
            id, confidence = self.recognizer.predict(cc[y:y+h, x:x+w])
            print(id, confidence)
            if confidence < 100 and confidence > max_conf:
                id = self.target_name
                max_conf = round(100-confidence)
                confidence = " {0}%".format(round(100-confidence))
            else:
                id = "unk"
                confidence = " {0}%".format(round(100-confidence))

            cv2.putText(orig, str(id), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(orig, str(confidence), (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
            
        cv2.imshow('frame', orig)
        cv2.waitKey(0)
        cv2.destroyAllWindows()