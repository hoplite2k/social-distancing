import tensorflow as tf
import cv2
import imutils
import numpy as np

class Model:
    def __init__(self, model_path):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(model_path, 'rb') as file:
                serialized_graph = file.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)
        
    def predict(self, img):
        img_exp = np.expand_dims(img, axis=0)
        (boxes, scores, classes) = self.sess.run([self.detection_graph.get_tensor_by_name('detection_boxes:0'), self.detection_graph.get_tensor_by_name('detection_scores:0'), self.detection_graph.get_tensor_by_name('detection_classes:0')],feed_dict={self.detection_graph.get_tensor_by_name('image_tensor:0'): img_exp})
        return (boxes, scores, classes)

def get_human_box_detection(boxes,scores,classes,height,width):
    array_boxes = []
    for i in range(boxes.shape[1]):
        if int(classes[i]) == 1 and scores[i] > 0.55:
            box = [boxes[0,i,0],boxes[0,i,1],boxes[0,i,2],boxes[0,i,3]] * np.array([height, width, height, width])
            array_boxes.append((int(box[0]),int(box[1]),int(box[2]),int(box[3])))
    return array_boxes

def get_points_from_box(box):
    center_x = int(((box[1]+box[3])/2))
    center_y = int(((box[0]+box[2])/2))
    center_y_ground = center_y + ((box[2] - box[0])/2)
    return (center_x,center_y),(center_x,int(center_y_ground))

def get_centroids_and_groundpoints(array_boxes_detected):
    array_centroids,array_groundpoints = [],[]
    for index,box in enumerate(array_boxes_detected):
        centroid,ground_point = get_points_from_box(box)
        array_centroids.append(centroid)
        array_groundpoints.append(centroid)
    return array_centroids,array_groundpoints

'''
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)

model_path = "./.tmp/datasets/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb"
model = Model(model_path)

vs = cv2.VideoCapture("./pedestrians.mp4")

while True:
    (frame_exists, frame) = vs.read()
    if not frame_exists:
        break
    else:
        frame = imutils.resize(frame, width=700)
        (boxes, scores, classes) =  model.predict(frame)
        array_boxes_detected = get_human_box_detection(boxes,scores[0].tolist(),classes[0].tolist(),frame.shape[0],frame.shape[1])
        array_centroids,array_groundpoints = get_centroids_and_groundpoints(array_boxes_detected)
        for i in range(len(array_boxes_detected)):
            (endx, startx, endy, starty) = array_boxes_detected[i]
            (cx,cy) = array_centroids[i]
            cv2.rectangle(frame, (startx, endx), (starty, endy), COLOR_RED, 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
'''



