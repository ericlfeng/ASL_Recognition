"""
A game that uses hand tracking to 
hit and destroy green circle enemies.

@author: Nandhini Namasivayam
@version: March 2024

edited from: https://i-know-python.com/computer-vision-game-using-mediapipe-and-python/
"""

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import random
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Library Constants
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkPoints = mp.solutions.hands.HandLandmark
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
DrawingUtil = mp.solutions.drawing_utils


class HandIdentifier:
    def __init__(self):
        # Create the hand detector
        self.dfcolnames = []
        base_options = BaseOptions(model_asset_path='data/hand_landmarker.task')
        options = HandLandmarkerOptions(base_options=base_options,
                                                num_hands=2)
        self.detector = HandLandmarker.create_from_options(options)

        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        self.model = KNeighborsClassifier(n_neighbors=9)
        self.train_model()




        # TODO: Load video
        self.video = cv2.VideoCapture(0)

    def draw_landmarks_on_hand(self, image, detection_result):
        """
        Draws all the landmarks on the hand
        Args:
            image (Image): Image to draw on
            detection_result (HandLandmarkerResult): HandLandmarker detection results
        """
        # Get a list of the landmarks
        hand_landmarks_list = detection_result.hand_landmarks
        
        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]

            # Save the landmarks into a NormalizedLandmarkList
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            # Draw the landmarks on the hand
            DrawingUtil.draw_landmarks(image,
                                       hand_landmarks_proto,
                                       solutions.hands.HAND_CONNECTIONS,
                                       solutions.drawing_styles.get_default_hand_landmarks_style(),
                                       solutions.drawing_styles.get_default_hand_connections_style())

    def convert_detection_result(self, detection_result, image):
            """
            Draws all the landmarks on the hand
            Args:
                image (Image): Image to draw on
                detection_result (HandLandmarkerResult): HandLandmarker detection results
            """
            # Get a list of the landmarks
            hand_landmarks_list = detection_result.hand_landmarks
            
            for idx in range(len(hand_landmarks_list)):
                #NOTE this only works for one hand
                listofpointstoreturn = []
                hand_landmarks = hand_landmarks_list[idx]

                # Save the landmarks into a NormalizedLandmarkList
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
                ])
                # results = mp_hands.Hands.process(image = image)

                if hand_landmarks:
                    for landmrk in hand_landmarks:
                        # for ids, landmrk in enumerate(hand_landmarks.landmark):
                            # print(ids, landmrk)
                        listofpointstoreturn.append(landmrk.x)
                        listofpointstoreturn.append(landmrk.y)
                        listofpointstoreturn.append(landmrk.z)
                    return listofpointstoreturn
            
            return "Nothing Here"
    
    def returndatapoints(self, image):
        to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        results = self.detector.detect(to_detect)            
        # Draw the hand landmarks
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

    def train_model(self):
        df = pd.read_csv("data/acquired_hands.csv")
        X = df.loc[:, df.columns != "label"]
        y = df["label"]
        self.model = self.model.fit(X, y)
        for i in range(1,22):
            for j in ["x","y","z"]:
                self.dfcolnames.append(j+str(i))
    


    def run(self):
        """
        Main game loop. Runs until the 
        user presses "q".
        """    
        
        START_TIME = time.time()
        # TODO: Modify loop condition  
        while self.video.isOpened():
            # Get the current frame
            frame = self.video.read()[1]

            # Convert it to an RGB image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            # Convert the image to a readable format and find the hands
            to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.detector.detect(to_detect)            
            

            # Draw the hand landmarks
            self.draw_landmarks_on_hand(image, results)
            data_points = self.returndatapoints(image)
            #print(data_points)
            if data_points != "Nothing Here":
                new_points = []
                new_points.append(data_points)
                to_predict = pd.DataFrame(new_points, columns = self.dfcolnames)
                print(self.model.predict(to_predict))

            # Change the color of the frame back
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow("Hand Tracking", image)


            # Break the loop if the user presses 'q'
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()
        #cv2.imshow("Hand Tracking", image)
        


if __name__ == "__main__":        
    h = HandIdentifier()
    h.run()