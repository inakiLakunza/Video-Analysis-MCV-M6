import utils
import cv2
import os

if __name__ == "__main__":




    video_path = '../../AICity_data/train/S03/c010/vdo.avi'
    vid = cv2.VideoCapture(video_path)
    os.makedirs("./frames", exist_ok=True)

    i = 0
    while True:
        ret, frame = vid.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            out_path_color = "./frames/"+str(i)+".png"
            
            #cv2.imwrite(out_path_gray, frame_gray)
            # COLORED IMAGES FROM BGR TO RGB
            cv2.imwrite(out_path_color, frame_rgb[:, :, ::-1])

            i+=1 

        else:
            break
    
    vid.release()
