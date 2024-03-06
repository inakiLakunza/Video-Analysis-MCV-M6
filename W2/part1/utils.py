import numpy as np
import cv2
import datetime
from lxml import etree

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pickle
import os
import argparse
import glob
import random

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog





def save_img(img, output, save_path, cfg):
    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(output["instances"].to("cpu"))
    # [:, :, ::-1] converts from RBG to BGR and vice versa
    cv2.imwrite(save_path, out.get_image()[:, :, ::-1])


def make_video(estimation):
    """
    Make a .mp4 from the estimation
    https://stackoverflow.com/questions/62880911/generate-video-from-numpy-arrays-with-opencv

    Parameters
        estimation : np.ndarray([1606, 1080, 1920, 3], dtype=uint8)
    """
    size = estimation.shape[1], estimation.shape[2]
    duration = estimation.shape[0]
    fps = 10
    out = cv2.VideoWriter(f'./Example_sequence.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
    for i in range(duration):
        data = (estimation[i]).astype(np.uint8)
        # I am converting the data to gray but we should look into this...
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        out.write(data)
    out.release()

if __name__ == "__main__":
    import os
    import pickle
    import matplotlib.pyplot as plt
    import glob
    

    image_list = []
    filenames = []

    for filename in glob.glob('plot_results/first_no_shuffle/*.jpg'): 
        path = (os.path.join(filename))
        filenames.append(path)
    
    filenames = sorted(filenames)

    
    #(sorted(filenames, key=lambda x: int(x.split("/")[-1].split(".")[-2].split("_")[-1]))[:60])
    for path in filenames:
        image_list.append(cv2.imread(path))

    # Output GIF path
    # Create GIF
    make_video(np.array(image_list[:250]))
    