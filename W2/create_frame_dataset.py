import utils
import cv2

if __name__ == "__main__":

    video_path = '/ghome/group07/test/AICity_data/train/S03/c010/vdo.avi'

    gray_frames, color_frames = utils.read_video(video_path)

    for i in range(len(gray_frames)):
        out_path_gray = "./frame_dataset/gray/"+str(i)+".png"
        out_path_color = "./frame_dataset/color/"+str(i)+".png"
        
        cv2.imwrite(out_path_gray, gray_frames[i])
        # COLORED IMAGES FROM BGR TO RGB
        cv2.imwrite(out_path_color, color_frames[i][:, :, ::-1])

