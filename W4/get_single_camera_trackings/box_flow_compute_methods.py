
import numpy as np
from collections import Counter
import cv2
import datetime
from lxml import etree
import matplotlib.pyplot as plt
import random
import os


# HABRÁ QUE PONER TMB MEDIAN Y MEAN
def flow_votation(box_flow):
    #print(box_flow)
    # Reshape the flow array to 2D (height*width, 2) for counting
    flow_reshaped = box_flow.reshape(-1, 2)
    #print(flow_reshaped)

    # Count the occurrences of each unique flow vector
    flow_counts = Counter(map(tuple, flow_reshaped))

    #print(flow_counts)

    # Get the most frequent flow vector
    most_common_flow = flow_counts.most_common(1)[0][0]

    return most_common_flow


def flow_median(box_flow):

    flow_reshaped = box_flow.reshape(-1, 2)
    median_x = np.median(flow_reshaped[:,1])
    median_y = np.median(flow_reshaped[:,0])

    return (median_y, median_x)


def flow_mean(box_flow):

    flow_reshaped = box_flow.reshape(-1, 2)
    mean_x = np.mean(flow_reshaped, axis=1)
    mean_y = np.mean(flow_reshaped, axis=0)

    return (mean_y, mean_x)