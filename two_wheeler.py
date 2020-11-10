import numpy as np
import cv2 as cv
import time
import os
from utils import infer_image, show_image,crop_img
from plate_recogniser_api import plate_reader
from calibration import  calibrate

confidence=0.5
threshold=0.3
config = './yolov3-coco/yolov3-personbikehead.cfg'
weights = './yolov3-coco/yolov3-personbikehead.weights'
labels = './yolov3-coco/pbh-labels.txt'
labels = open(labels).read().strip().split('\n')
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
net = cv.dnn.readNetFromDarknet(config, weights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)


video_path = "D:\\Programs\\Final Project\\test_data\\test_1.mp4"
cap = cv.VideoCapture(video_path)
layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

dest="D:\\Programs\\Final Project\\code_output\\"


if video_path:
    # Read the video
    read_plates=[]
    try:
        vid = cv.VideoCapture(video_path)
        height, width = None, None
        writer = None
    except:
        raise 'Video cannot be loaded!\n\
                           Please check the path provided!'

    finally:
        count=0
        while True:
            grabbed, frame = vid.read()
            # Checking if the complete video is read
            if not grabbed:
                break
            count+=1
            if count==1:
                poly = calibrate(frame)
                #print(poly)
            if width is None or height is None:
                height, width = frame.shape[:2]
            try:
                frame1,bike_np_roi, _, _, _, _ = infer_image(net, layer_names, height, width, frame.copy(), colors, labels, confidence,threshold,poly)
            except:
                print("Skipping.................................................")
                continue
            if bike_np_roi != []:
                wait_flag=0
                for k in bike_np_roi:

                    #print(f'Bike co-ordinates {k[0]} , number plate co-ordinates {k[1]}')
                    bike=k[0]
                    bike_img = crop_img(frame.copy(), bike)
                    np=k[1]
                    if np!=():
                        wait_flag += 1
                        np_img = crop_img(frame.copy(), np)
                        np_img_text=np_img.copy()
                        if wait_flag>1:
                            time.sleep(1)
                        read_score= plate_reader(np_img)
                        if(read_score == "Can't read"):
                            np_chars="Can't read"
                            score=0
                        else:
                            np_chars=read_score[0]
                            score=float(read_score[1])
                        cv.putText(np_img_text,str(np_chars), (10, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, [255, 0, 0], 1)

                        if np_chars not in read_plates and np_chars!="Can't read":
                            if score <=.85:
                                print(f'Low score plate , registration number  is : {np_chars} score is: {score}')
                            else:
                                print(f'New number plate detected , registration number  is : {np_chars} score is:{score}')
                                newdir=dest+np_chars
                                if np_chars not in os.listdir(dest):
                                    os.mkdir(newdir)
                                    cv.imwrite(newdir+"\\bike"+".jpg",bike_img)
                                    cv.imwrite(newdir+"\\num_plate"+".jpg",np_img)
                                read_plates.append(np_chars)
                        elif np_chars in read_plates:
                            print(f'Repeating plate detected , registration number  is : {np_chars} ')
                        cv.imshow("numberplate", np_img_text)
                        cv.waitKey(1)
                    #cv.imshow("bike", bike_img)
                    #cv.waitKey(5)



            cv.imshow('vid', frame1)
            if cv.waitKey(1) & 0xFF == ord('c'):
                poly = calibrate(frame)
                #print(poly)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        vid.release()
        cv.destroyAllWindows()

