# By Eric Feng
# For AT CS 2023-2024
# ASL Letter Identifier

# Importing necessary libraries
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Define RGB color constants
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Library Constants
# Importing necessary modules and classes from mediapipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkPoints = mp.solutions.hands.HandLandmark
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
DrawingUtil = mp.solutions.drawing_utils

class HandIdentifier:
    def __init__(self):
        # Initialize the HandIdentifier class
        self.dfcolnames = []
        # Set up Hand Landmarker options
        base_options = BaseOptions(model_asset_path='data/hand_landmarker.task')
        options = HandLandmarkerOptions(base_options=base_options,
                                                num_hands=2)
        self.detector = HandLandmarker.create_from_options(options)
        # Initialize KNeighborsClassifier model
        self.model = KNeighborsClassifier(n_neighbors=7)
        # Train the model
        self.train_model()
        # Initialize video capture device
        self.video = cv2.VideoCapture(0)

    # Method to draw landmarks on the detected hand
    def draw_landmarks_on_hand(self, image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks        
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            DrawingUtil.draw_landmarks(image,
                                       hand_landmarks_proto,
                                       solutions.hands.HAND_CONNECTIONS,
                                       solutions.drawing_styles.get_default_hand_landmarks_style(),
                                       solutions.drawing_styles.get_default_hand_connections_style())

    # Method to convert detection result into data points
    def convert_detection_result(self, detection_result, image):
            hand_landmarks_list = detection_result.hand_landmarks
            for idx in range(len(hand_landmarks_list)):
                listofpointstoreturn = []
                hand_landmarks = hand_landmarks_list[idx]
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
                ])
                # results = mp_hands.Hands.process(image = image)
                if hand_landmarks:
                    for landmrk in hand_landmarks:
                        listofpointstoreturn.append(landmrk.x)
                        listofpointstoreturn.append(landmrk.y)
                        listofpointstoreturn.append(landmrk.z)
                    return listofpointstoreturn
            return "Nothing Here"

    # Method to return data points
    def returndatapoints(self, image):
        to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        results = self.detector.detect(to_detect)            
        points = self.convert_detection_result(results, image)
        if points == "Nothing Here":
            border_size = 150
            frame = cv2.copyMakeBorder(image, top=border_size, 
                                    bottom=border_size, left=border_size, 
                                    right=border_size,
                                    borderType=cv2.BORDER_CONSTANT, 
                                    value=[0, 0, 0]
            )
            to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            results = self.detector.detect(to_detect) 
            points = self.convert_detection_result(results, frame)
        return points

    # Method to train the KNeighborsClassifier model
    def train_model(self):
        df = pd.read_csv("data/acquired_hands.csv")
        X = df.loc[:, df.columns != "label"]
        y = df["label"]
        self.model = self.model.fit(X, y)
        for i in range(1,22):
            for j in ["x","y","z"]:
                self.dfcolnames.append(j+str(i))

    # Method to run the ASL letter identification process
    def run(self):
        while self.video.isOpened():
            frame = self.video.read()[1]
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.detector.detect(to_detect)            
            self.draw_landmarks_on_hand(image, results)
            data_points = self.returndatapoints(image)
            if data_points != "Nothing Here":
                new_points = []
                new_points.append(data_points)
                to_predict = pd.DataFrame(new_points, columns = self.dfcolnames)
                #print(self.model.predict(to_predict)[0])
                cv2.putText(image, 
                            self.model.predict(to_predict)[0], 
                            (100, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, 
                            GREEN, 4,  
                            cv2.LINE_4) 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow("Hand Tracking", image)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
        self.video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":        
    h = HandIdentifier()
    h.run()
