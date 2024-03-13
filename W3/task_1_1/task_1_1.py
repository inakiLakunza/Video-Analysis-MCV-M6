import numpy as np



from PIL import Image


import tqdm
import cv2


import optical_flow.visualizations as visu
import optical_flow.utils as utils
from optical_flow.Estimators import Flow_Field_Block_Matching

import random 
import time

import json


with open("results/metric_results.json", "r") as file:
    results_random_search = json.load(file) 
    
    
results_random_search["estimators"] = []

range_block_size = range(30,75, 5)
range_search_max_dist =  range(20,40, 2)
estimators = ["ccorr", "ccoeff"]

max_iterations = 1
current_iterations = 0


path_ref_image = 'data_stereo_flow/training/image_0/000045_10.png'
path_target_image = 'data_stereo_flow/training/image_0/000045_11.png'
ground_truth_paht = 'data_stereo_flow/training/flow_noc/000045_10.png'

image_ref = cv2.imread(path_ref_image)[:,:,::-1]
cv2.imwrite("results" + "/image1.png", image_ref)

image_target = cv2.imread(path_target_image)[:,:,::-1]
cv2.imwrite("results" + "/image2.png", image_target)

image_gt = (utils.load_optical_flow(ground_truth_paht))
cv2.imwrite("results" + "/image3.png", image_gt)


    

while  (current_iterations <  max_iterations):
    block_size = 35#random.sample(range_block_size, 1)[0]
    search_max_dist = 38#random.sample(range_search_max_dist, 1)[0]
    matcher = "ccorr"#random.sample(estimators, 1)[0]
    output_dir = f"results/bs_{block_size}_sm_{search_max_dist}"
    print(f"Starting the evaluation with BS:{block_size}, MS:{search_max_dist} and matcher:{matcher}")

    of = Flow_Field_Block_Matching(block_size=block_size, search_max_dist=search_max_dist,
                                  ref_img=image_ref, comp_img=image_target,
                                  output_dir=f"results/bs_{block_size}_sm_{search_max_dist}_{matcher}", matcher=matcher)
    
    #try:
    start_time = time.time()
    of.estimate_of(visualization=False)
    postprocessed_of = of.postprocess(optical_flow_image=of.flow_field, color_diff_thr=3)
    finish_time = time.time()    
    
    results_random_search["block_size"].append(block_size)
    results_random_search["search_area"].append(search_max_dist)
    results_random_search["time"].append(finish_time - start_time)
    
    error_msne, error_sen = utils.OF_MSEN(GT=image_gt, pred=postprocessed_of, output_dir=f"results/bs_{block_size}_sm_{search_max_dist}", visualize=True) 
    error_pepn = utils.calculate_pepn(gt_flow=image_gt, pred_flow=postprocessed_of)
    print(error_msne)
    print(error_pepn)

    results_random_search["msne"].append(error_msne)
    results_random_search["pepn"].append(error_pepn)
    results_random_search["estimators"].append(matcher)
    
    
    visu.visualize_flowvis(flow=postprocessed_of, filepath=output_dir+"/flowvis.png")
    visu.plot_optical_flow_hsv(flow=postprocessed_of[:,:,:2], labelled=postprocessed_of[:,:,2], output_dir=output_dir)
    visu.plot_optical_flow_quiver(postprocessed_of, image_ref, output_dir=output_dir)
    visu.plot_optical_flow_quiver(postprocessed_of, image_ref, flow_with_camera=True, output_dir=output_dir)
    visu.plot_optical_flow_surface(postprocessed_of, image_ref, output_dir=output_dir)
    exit()
    current_iterations +=1
        
    #except Exception as e:
    #    print(e)
    #    current_iterations +=1
    #    continue     
    
    
import os
#os.remove("results/metric_results.json")
#with open("results/metric_results.json", "w") as file:
#    json.dump(results_random_search, file)   
        
    