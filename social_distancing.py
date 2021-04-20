import cv2
import imutils
from datetime import datetime
import time
import math
import numpy as np
import itertools
from human_detection import *
from cloud_upload import *
import multiprocessing

class Process(multiprocessing.Process):
    def __init__(self, path, msg):
        super(Process, self).__init__()
        self.path = path
        self.msg = msg

    def run(self):
        cloud_upload(self.path)
        if(self.msg != ""):
            print(self.msg)

COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
MIN_DIST = 50
model_path = "./.tmp/datasets/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb"

print("[ Loading the TENSORFLOW MODEL ... ]")
model = Model(model_path)
print("Done : [ Model loaded and initialized ] ...")

vs = cv2.VideoCapture("./input/pedestrians.mp4")
#vs = cv2.VideoCapture(0)
start_time = time.time()
output_video  = None
print("[ Starting the Stream ... ]")

while True:
    (frame_exists, frame) = vs.read()
    if not frame_exists:
        break
    else:
        frame = imutils.resize(frame, width=700)
        (boxes, scores, classes) =  model.predict(frame)
        array_boxes_detected = get_human_box_detection(boxes,scores[0].tolist(),classes[0].tolist(),frame.shape[0],frame.shape[1])
        array_centroids,array_groundpoints = get_centroids_and_groundpoints(array_boxes_detected)

        violate = set()

        if len(array_boxes_detected) >= 2:
            list_indexes = list(itertools.combinations(range(len(array_boxes_detected)), 2))
            for i,pair in enumerate(itertools.combinations(array_boxes_detected, r=2)):
                if math.sqrt( (pair[0][0] - pair[1][0])**2 + (pair[0][1] - pair[1][1])**2 ) < int(MIN_DIST):
                    index_pt1 = list_indexes[i][0]
                    index_pt2 = list_indexes[i][1]
                    violate.add(index_pt1)
                    violate.add(index_pt2)

        for i in range(len(array_boxes_detected)):
            (endx, startx, endy, starty) = array_boxes_detected[i]
            (cx,cy) = array_centroids[i]

            if i in violate:
                color = COLOR_RED
            else:
                color = COLOR_GREEN

            cv2.rectangle(frame, (startx, endx), (starty, endy), color, 2)
            cv2.circle(frame, (cx, cy), 5, color, 1)

            now = datetime.now()
            text = now.strftime("%a %d-%m-%Y %H:%M:%S")
            cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_BLUE, 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        if  output_video is None:
            video_file_name = "./" + str(datetime.now().strftime("%c")) + ".avi"
            output_video = cv2.VideoWriter(video_file_name, fourcc, 25, (frame.shape[1], frame.shape[0]), True)

        if time.time() - start_time > 3600:
            pvideo_file_name = "./" + str(datetime.now().strftime("%c")) + ".avi"
            p = Process(pvideo_file_name, "")
            p.start()
            start_time = time.time()
            video_file_name = "./" + str(datetime.now().strftime("%c")) + ".avi"
            output_video = cv2.VideoWriter(video_file_name, fourcc, 25,(frame.shape[1], frame.shape[0]), True)

        if output_video is not None:
            output_video.write(frame)

        if key == ord("q"):
            break

pe = Process(video_file_name, "Done : [ Uploaded all files to cloud ] ...")
pe.start()
            
vs.release()
cv2.destroyAllWindows()


                    


