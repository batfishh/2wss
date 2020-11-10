import numpy as np
import cv2 as cv
import subprocess
import time
import os


confidence=0.5
threshold=0.3
config = './yolov3-coco/yolov3-numberplate.cfg'
weights = './yolov3-coco/yolov3-numberplate.weights'
labels = './yolov3-coco/np-labels.txt'
labels = open(labels).read().strip().split('\n')
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
net = cv.dnn.readNetFromDarknet(config, weights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)



layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def show_image(img):
    cv.imshow("Image", img)
    cv.waitKey(0)



def draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels):
    # If there are any detections
    if len(idxs) > 0:
        np_boxes = []
        for i in idxs.flatten():
            j=len(idxs.flatten())-1
            # Get the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            # Get the unique color for this class
            color = [int(c) for c in colors[classids[i]]]

            # Draw the bounding box rectangle and label on the image
            if classids[i] == 0:
                np_boxes.append((x,y,w,h))
                cv.rectangle(img, (x, y), (x + w, y + h), [255,0,0], 1)
                text = "{}: {:4f}".format(labels[classids[i]], confidences[i])
                #cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        np_boxes=[]
    return img,np_boxes





def generate_boxes_confidences_classids(outs, height, width, tconf):
    boxes = []
    confidences = []
    classids = []

    for out in outs:
        for detection in out:
            # print (detection)
            # a = input('GO!')

            # Get the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]

            # Consider only the predictions that are above a certain confidence level
            if confidence > tconf and (classid==0):
                # TODO Check detection
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, bwidth, bheight = box.astype('int')

                # Using the center x, y coordinates to derive the top
                # and the left corner of the bounding box
                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))

                # Append to list
                boxes.append([x, y, int(bwidth), int(bheight)])
                confidences.append(float(confidence))
                classids.append(classid)

    return boxes, confidences, classids


def infer_image(net, layer_names, height, width, img, colors, labels, confidence,threshold,
                boxes=None, confidences=None, classids=None, idxs=None, infer=True):
    if infer:
        # Contructing a blob from the input image
        blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
                                    swapRB=True, crop=False)

        # Perform a forward pass of the YOLO object detector
        net.setInput(blob)

        # Getting the outputs from the output layers
        start = time.time()
        outs = net.forward(layer_names)
        end = time.time()


        # Generate the boxes, confidences, and classIDs
        boxes, confidences, classids = generate_boxes_confidences_classids(outs, height, width, confidence)

        # Apply Non-Maxima Suppression to suppress overlapping bounding boxes
        idxs = cv.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

    if boxes is None or confidences is None or idxs is None or classids is None:
        raise '[ERROR] Required variables are set to None before drawing boxes on images.'

    # Draw labels and boxes on the image
    img,np_boxes = draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels)

    return img,np_boxes,boxes, confidences, classids, idxs

def numberplate_detect(img):
    height, width = img.shape[:2]
    bike_plate,np_boxes, _, _, _, _ = infer_image(net, layer_names, height, width, img.copy(), colors, labels, confidence,threshold)
    return bike_plate,np_boxes


