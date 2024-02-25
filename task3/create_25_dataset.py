import utils
import cv2

if __name__ == "__main__":

    video_path = '/ghome/group07/test/AICity_data/train/S03/c010/vdo.avi'

    gray_frames, color_frames = utils.read_video(video_path)
    print(f"gray_frames.shape: {gray_frames.shape} \t color_frames.shape: {color_frames.shape}")
    gray_frames_25, gray_frames_75 = utils.split_frames(gray_frames)

    for i in range(len(gray_frames_25)):
        path = "./frames25/"+str(i+1)+".png"
        cv2.imwrite(path, gray_frames_25[i])

