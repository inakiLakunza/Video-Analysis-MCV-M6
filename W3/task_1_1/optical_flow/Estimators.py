import numpy as np

from optical_flow.matching_costs import *

import tqdm
import copy
import cv2
import matplotlib.pyplot as plt




class Block():
    
    """
    Block class, which will be used for comparing with other blocks
    """

    def __init__(self, block_matcher, block_size, matrix):
        self.block_macther = block_matcher
        self.block_size = self.block_size

        # The matrix contains the pixel values of the block
        self.matrix = matrix

    
    def block_size(self):
        return self.block_size
    

    def match(self, other, matcher="euclidean"):
        if matcher == "mse":
            return MSE(self.matrix, other.matrix)
        elif matcher == "mae":
            return MAE(self.matrix, other.matrix)
        
        elif matcher == "ccorr":
            return NCCORR()
        
        elif matcher == "ccoeff":
            return NCCOEFF()
        
        else:
            raise("Error function not found")
            
        


class Flow_Field_Block_Matching():

    """
    Optical flow class, it will compute the optical flow, and save it if wanted
    """

    def __init__(self, block_size, ref_img, comp_img, matcher, search_max_dist, output_dir):
        # Block size, it MUST be odd
        self.block_size = block_size
        self.output_dir = output_dir
        #assert (self.block_size%2==1), "Inserted block size is not odd"

        self._prev_image = ref_img
        self._next_image = comp_img

        # Metric to use for block comparison
        # Insert more metrics in the list when implemented
        self.matcher = matcher.lower()
        implemented_matchers = ["mse", "mae", "ccorr", "ccoeff"]
        
        self.error_functions = {
            "mse": MSE,
            "mae": MAE, 
            "ccorr": NCCORR(), 
            "ccoeff": NCCOEFF()
        }
        
        self.temp_matchers = {
            "mse": self.match_blocks,
            "mae": self.match_blocks, 
            "ccorr": self.cv2_matching_blocks, 
            "ccoeff": self.cv2_matching_blocks
        }
        
        self.matching_func = self.temp_matchers[self.matcher]
        self.error_func = self.error_functions[self.matcher]
        
        error_msg_matcher = f"The chosen matcher is not implemented, the available options are: {implemented_matchers}"
        assert (matcher in implemented_matchers), error_msg_matcher
        

        # Absolute value of pixels we can displace from the centroid
        # to find the match
        self.search_max_dist = search_max_dist // 2

        # Initialization of the flow field we will get as result
        # We will have a horizontal distance and a vertical distance
        # so we will return an image with shape (img.height, img.width, 2)
        self.flow_field = np.zeros((self._prev_image.shape[0], self._prev_image.shape[1], 3), dtype=np.float32)

        self.visualization = False
        
    def block_size(self):
        return self.block_size
    
    def get_img_size(self):
        return self.img_size
    
    def get_search_max_dist(self):
        return self.search_max_dist

    def cv2_matching_blocks(self, block_prev:Block, image_next, y_min, y_max, x_min, x_max):
        block_next = image_next[y_min:y_max, x_min:x_max]
        tmp = cv2.matchTemplate(block_next, block_prev.matrix, self.error_func)
        min_v, max_v, min_l, max_l = cv2.minMaxLoc(tmp)
        return max_l[0] + x_min, max_l[1] + y_min
    
    def match_blocks(self, block_prev:Block, image_next, y_min, y_max, x_min, x_max):
        min_error = np.inf
        min_x_next = min_y_next = 0

        for y_next in range(y_min, y_max - self.block_size):
            for x_next in range(x_min, x_max - self.block_size):
                block_next = image_next[y_next:y_next +
                                        self.block_size, x_next:x_next + self.block_size]
                

                    
                actual_block = Block(block_matcher=self, block_size=self.block_size, matrix=block_next)
                error = block_prev.match(actual_block, matcher=self.matcher) 
                
                if error < min_error:
                    min_error = error
                    min_x_next = x_next
                    min_y_next = y_next
                    
        

        return min_x_next, min_y_next
    
    
    ## maybe chabge the search max to have other windowd
    def estimate_of(self, visualization=True):
        
        

        if visualization == True:
            self.visualization = visualization
            
        

        for y_prev in tqdm.tqdm(range(0, (self._prev_image.shape[0]-self.block_size), self.block_size), desc="Estimating Optical Flow (searching by row)"):
            
            y_min = max(0, y_prev - self.search_max_dist)
            y_max = min(
                self._prev_image.shape[0], y_prev + self.search_max_dist + self.block_size)
           
            
            for x_prev in range(0, (self._prev_image.shape[1]-self.block_size), self.block_size):
                

                    
                
                x_min = max(0, x_prev - self.search_max_dist)
                x_max = min(self._prev_image.shape[1], x_prev + self.search_max_dist + self.block_size)
                
                block_prev = self._prev_image[y_prev:y_prev + self.block_size, x_prev:x_prev + self.block_size]
                if self.visualization== True:
                    img_copy = copy.copy(self._prev_image)  
                    path_previous_image = os.path.join(self.output_dir, f"row_reference_img_{y_prev}")
                    os.makedirs(path_previous_image, exist_ok=True)
                    
                    start_point = (x_prev, y_prev)
                    end_point = (x_prev + self.block_size, y_prev+self.block_size)

                    cv2.rectangle(img_copy, start_point, end_point, (0,255,0), 4)
                    cv2.imwrite(os.path.join(path_previous_image, f'evolution_{x_prev}.png'), img=img_copy)
                
                actual_block = Block(block_matcher=self, block_size=self.block_size, matrix=block_prev)

                min_x_next, min_y_next = self.matching_func(block_prev=actual_block,
                                            image_next=self._next_image,
                                            y_min=y_min, y_max= y_max,
                                            x_min=x_min, x_max=x_max)

               
                if self.visualization:
                    img_copy = copy.copy(self._next_image)  
                    path_previous_image = os.path.join(self.output_dir, f"comparation_img_row_{y_prev}")
                    os.makedirs(path_previous_image, exist_ok=True)
                    
                    start_point = (min_x_next, min_y_next)
                    end_point = (min_x_next + self.block_size, min_y_next+self.block_size)

                    cv2.rectangle(img_copy, start_point, end_point, (255,0,0), 4)
                    cv2.imwrite(os.path.join(path_previous_image, f'best_match_evolution_{x_prev}.png'), img=img_copy)
                
            
                self.flow_field[y_prev:y_prev+self.block_size, x_prev:x_prev+self.block_size, 0] = min_x_next - x_prev
                self.flow_field[y_prev:self.block_size+y_prev, x_prev:self.block_size+x_prev, 1] = min_y_next - y_prev


        
        self.flow_field[:,:, 2] = 1
        
        


        

    ## Extracted from team 2 2022/2023 
    ## thx Adrià Molina dear friend 
    ## The block size must be the same as used to generate the optical fow image
    def postprocess(self, optical_flow_image, variance_thr: float = 100, color_diff_thr: float = 8, window_size: int = 3):
        new_flow = optical_flow_image.copy()
        # Interpolate the optical flow in border pixels from neighborhood
        # For instance, for each pixel of the left border, we calculate 
        # the optical flow using the pixels of (i, self.block_size), (i-1, self.block_size), (i+1, self.block_size)
        # by interpolation
        for i in range(0, optical_flow_image.shape[0]- self.block_size, self.block_size):
            new_flow[i:i+self.block_size, :self.block_size, :] = (optical_flow_image[i-1, self.block_size, :] + optical_flow_image[i, self.block_size, :] + optical_flow_image[i+1, self.block_size, :]) / 3
            new_flow[i:i+self.block_size, -self.block_size*2:, :] = (optical_flow_image[i-1, -self.block_size*2-1, :] + optical_flow_image[i, -self.block_size*2-1, :] + optical_flow_image[i+1, -self.block_size*2-1, :]) / 3

        for j in range(0, optical_flow_image.shape[1]- self.block_size, self.block_size):
            new_flow[:self.block_size, j:j+self.block_size, :] = (optical_flow_image[self.block_size, j-1, :] + optical_flow_image[self.block_size, j, :] + optical_flow_image[self.block_size, j+1, :]) / 3
            new_flow[-self.block_size:, j:j+self.block_size, :] = (optical_flow_image[-self.block_size-1, j-1, :] + optical_flow_image[-self.block_size-1, j, :] + optical_flow_image[-self.block_size-1, j+1, :]) / 3

        # Special corner cases 
        new_flow[:self.block_size, :self.block_size, :] = (new_flow[self.block_size, self.block_size, :] + new_flow[self.block_size, 0, :] + new_flow[0, self.block_size, :]) / 3
        new_flow[:self.block_size, -self.block_size:, :] = (new_flow[self.block_size, -self.block_size-1, :] + new_flow[self.block_size, -self.block_size*2-1, :] + new_flow[0, -self.block_size-1, :]) / 3
        new_flow[-self.block_size:, :self.block_size, :] = (new_flow[-self.block_size-1, self.block_size, :] + new_flow[-self.block_size-1, 0, :] + new_flow[-self.block_size*2-1, self.block_size, :]) / 3
        new_flow[-self.block_size:, -self.block_size:, :] = (new_flow[-self.block_size-1, -self.block_size-1, :] + new_flow[-self.block_size-1, -self.block_size*2-1, :] + new_flow[-self.block_size*2-1, -self.block_size-1, :]) / 3
        optical_flow_image = new_flow
        new_flow = optical_flow_image.copy()
        hsv = cv2.cvtColor(optical_flow_image, cv2.COLOR_BGR2HSV)
        hue = hsv[:,:,2]
        # save the hue image
        cv2.imwrite("hue.png", hue)

        # Define the window size and threshold for computing the mask
        window_size = self.block_size*window_size
        mask = np.zeros_like(new_flow[:,:,0], dtype=np.uint8)

        # Compute the var of the flow in each window
        variance = np.zeros_like(new_flow[:,:,0])
        for i in range(0, new_flow.shape[0], self.block_size):
            for j in range(0, new_flow.shape[1], self.block_size):
                x_min = max(0, j - window_size)
                x_max = min(new_flow.shape[1], j + window_size)
                y_min = max(0, i - window_size)
                y_max = min(new_flow.shape[0], i + window_size)
                
                # # Compute the variance of the flow in the window only in the masked area
                window = hue[y_min:y_max, x_min:x_max]
                variance[i:i+self.block_size,j:j+self.block_size] = np.var(window)

                current_color = np.mean(optical_flow_image[i:i+self.block_size, j:j+self.block_size, :], axis=(0, 1))
                
                # Sum all the colors in the neighborhood and divide by 8 to get the mean color
                neighbors_color = []
                for k in range(i-self.block_size, i+self.block_size+1, self.block_size):
                    for l in range(j-self.block_size, j+self.block_size+1, self.block_size):
                        if k == i and l == j:
                            continue
                        k = max(0, k)
                        k = min(optical_flow_image.shape[0]-1, k)
                        l = max(0, l)
                        l = min(optical_flow_image.shape[1]-1, l)
                        neighbors_color.append(np.mean(optical_flow_image[k:k+self.block_size, l:l+self.block_size, :], axis=(0, 1)))

                neighbors_color_mean = np.mean(neighbors_color, axis=0)

                # If the average color is different from the mean color, we interpolate the optical flow
                if np.linalg.norm(current_color - neighbors_color_mean) > color_diff_thr:
                    mask[i:i+self.block_size, j:j+self.block_size] = 1

        # Create a binary mask based on the var threshold
        var_mask = (variance > variance_thr).astype(np.uint8)
        var_mask = cv2.dilate(var_mask, np.ones((self.block_size*3,self.block_size*3), np.uint8), iterations=1)
        mask += var_mask

        cv2.imwrite("mask.png", mask*255)
        # Inpaint the masked areas using Navier-Stokes based method
        new_flow[:,:,0] = cv2.inpaint(new_flow[:,:,0], mask, 3, cv2.INPAINT_NS)
        new_flow[:,:,1] = cv2.inpaint(new_flow[:,:,1], mask, 3, cv2.INPAINT_NS)

        optical_flow_image = new_flow

        # Apply a median filter to the optical flow
        optical_flow_image = cv2.medianBlur(optical_flow_image, 5)

        # Apply a Gaussian filter to the optical flow
        optical_flow_image = cv2.GaussianBlur(optical_flow_image, (55, 55), 0)
        optical_flow_image = cv2.GaussianBlur(optical_flow_image, (27, 27), 0)
        optical_flow_image = cv2.GaussianBlur(optical_flow_image, (13, 13), 0)
        optical_flow_image[:,:,2] = 1 # Just in case
        return optical_flow_image






if __name__ == "__main__":
    
    import os
    import cv2 
    
    path_to_save = "./results"
    os.makedirs("./results", exist_ok=True)
    
    
    path_1 = 'data_stereo_flow/training/image_0/000045_10.png'
    path_2 = 'data_stereo_flow/training/image_0/000045_11.png'
    
    image_1 = cv2.imread(path_1)   
    image_2 = cv2.imread(path_2)
    
    
    print("Saving the images whose the optical flow will be computed in: ", os.path.dirname(path_1))
    cv2.imwrite("results/image1.png", image_1)
    
    
    #self.block_size, ref_img, comp_img, matcher, search_max_dist
    of = Flow_Field_Block_Matching(block_size=52, ref_img=image_1, comp_img=image_2, matcher="ccoeff", search_max_dist=26, output_dir="./results")
    
    of.estimate_of(visualization=False)
    h = of.postprocess(optical_flow_image=of.flow_field)
    
    cv2.imwrite(path_to_save + "/OF_2.png", h)
    


