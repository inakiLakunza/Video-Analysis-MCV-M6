import numpy as np

from matching_costs import Euclidean_match

import tqdm


import matplotlib.pyplot as plt



class Block():
    
    """
    Block class, which will be used for comparing with other blocks
    """

    def __init__(self, block_matcher, block_size, matrix):
        self.block_macther = block_matcher
        self.block_size = block_size

        # The matrix contains the pixel values of the block
        self.matrix = matrix

    
    def get_block_size(self):
        return self.block_size
    

    def match(self, other, matcher="euclidean"):
        if matcher == "euclidean":
            return Euclidean_match(self.matrix, other.matrix)
        


class Flow_Field_Block_Matching():

    """
    Optical flow class, it will compute the optical flow, and save it if wanted
    """

    def __init__(self, block_size, ref_img, comp_img, matcher, search_max_dist):
        # Block size, it MUST be odd
        self.block_size = block_size
        #assert (self.block_size%2==1), "Inserted block size is not odd"

        self._prev_image = ref_img
        self._next_image = comp_img

        # Metric to use for block comparison
        # Insert more metrics in the list when implemented
        self.matcher = matcher.lower()
        implemented_matchers = ["euclidean"]
        error_msg_matcher = f"The chosen matcher is not implemented, the available options are: {implemented_matchers}"
        assert (matcher in implemented_matchers), error_msg_matcher
        
        #change this
        self.error_function = Euclidean_match

        # Absolute value of pixels we can displace from the centroid
        # to find the match
        self.search_max_dist = search_max_dist // 2

        # Initialization of the flow field we will get as result
        # We will have a horizontal distance and a vertical distance
        # so we will return an image with shape (img.height, img.width, 2)
        self.flow_field = np.zeros((self._prev_image.shape[0], self._prev_image.shape[1], 3), dtype=np.float32)


    def get_block_size(self):
        return self.block_size
    
    def get_img_size(self):
        return self.img_size
    
    def get_search_max_dist(self):
        return self.search_max_dist


    def match_blocks(self, block_prev:Block, image_next, y_min, y_max, x_min, x_max):
        block_prev_matrix = block_prev.matrix
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
    def estimate_of(self):
        

        for y_prev in tqdm.tqdm(range(0, (self._prev_image.shape[0]-self.block_size), self.block_size), desc="Estimating Optical Flow (searching by row)"):
            
            y_min = max(0, y_prev - self.search_max_dist)
            y_max = min(
                self._prev_image.shape[0], y_prev + self.search_max_dist + self.block_size)
           
            
            for x_prev in range(0, (self._prev_image.shape[1]-self.block_size), self.block_size):

                x_min = max(0, x_prev - self.search_max_dist)
                x_max = min(self._prev_image.shape[1], x_prev + self.search_max_dist + self.block_size)
                
                block_prev = self._prev_image[y_prev:y_prev + self.block_size, x_prev:x_prev + self.block_size]
                actual_block = Block(block_matcher=self, block_size=self.block_size, matrix=block_prev)

                min_x_next, min_y_next = self.match_blocks(block_prev=actual_block,
                                            image_next=self._next_image,
                                            y_min=y_min, y_max= y_max,
                                            x_min=x_min, x_max=x_max)



                self.flow_field[y_prev:y_prev+self.block_size, x_prev:x_prev+self.block_size, 0] = min_x_next - x_prev
                self.flow_field[y_prev:self.block_size+y_prev, x_prev:self.block_size+x_prev, 1] = min_y_next - y_prev
        

        
        self.flow_field[:,:, 2] = 1






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
    
    
    #block_size, ref_img, comp_img, matcher, search_max_dist
    of = Flow_Field_Block_Matching(block_size=4, ref_img=image_1, comp_img=image_2, matcher="euclidean", search_max_dist=26)
    
    of.estimate_of()
    
    cv2.imwrite(path_to_save + "/OF_2.png", of.flow_field)
    

    