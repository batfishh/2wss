import cv2
import os

def video_to_frames(video,video_name, path_output_dir):

    vidcap = cv2.VideoCapture(video)
    count = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            cv2.imshow('frame',image)
            if cv2.waitKey(0) & 0xFF == ord('s'):
                print("saving")
                cv2.imwrite(os.path.join(path_output_dir, video_name+'_'+'%d.jpeg') % count, image)
                count += 1
            elif cv2.waitKey(0) & 0xFF == ord('q'):
                print("quitting")
                break
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()

video_directory="D:\\Programs\\test_data\\shrey_2.mp4"
video_name="shrey_2"
save_path="C:\\Users\\ghost\\Desktop\\shrey_2"
video_to_frames(video_directory,video_name,save_path)