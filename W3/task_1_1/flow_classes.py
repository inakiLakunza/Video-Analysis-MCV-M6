import numpy as np

from matching_costs import Euclidean_match






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
        assert (self.block_size%2==1, "Inserted block size is not odd")

        # Displacement where center index of the block is found
        # For instance, in 3x3 matrix would be 1, because the centroid
        # value is found at matrix_3x3[1][1]
        self.block_matrix_center_disp = self.block_size//2 + 1

        self.ref_img = ref_img
        self.comp_img = comp_img

        self.img_width = self.ref_img.shape[1]
        self.img_height = self.ref_img.shape[0]

    
        # Metric to use for block comparison
        # Insert more metrics in the list when implemented
        self.matcher = matcher.lower()
        implemented_matchers = ["euclidean"]
        error_msg_matcher = f"The chosen matcher is not implemented, the available options are: {implemented_matchers}"
        assert (matcher in implemented_matchers, error_msg_matcher)

        # Absolute value of pixels we can displace from the centroid
        # to find the match
        self.search_max_dist = search_max_dist

        # We need to go pixel by pixel, and the block has size larger than 1
        # so for now we will ignore the border pixels and fit each value
        # of the block with a pixel value. (In the future, we could also
        # do something like padding
        self.n_ignore_border = self.block_size//2

        # Initialization of the flow field we will get as result
        # We will have a horizontal distance and a vertical distance
        # so we will return an image with shape (img.height, img.width, 2)
        self.flow_field = np.zeros((self.height, self.width, 2), dtype=int)

        self.flow_field_obtained = False

    def get_block_size(self):
        return self.block_size
    
    def get_img_size(self):
        return self.img_size
    
    def get_search_max_dist(self):
        return self.search_max_dist
    

    def finder(self):

        for i_x in range(0, self.img_width, self.block_size):
            for i_y in range(0, self.img_height, self.block_size):
                
                reference_matrix=None
                if len(self.ref_img.shape) == 3:
                    # Color img
                    reference_matrix = self.ref_img[i_y:i_y+self.block_size,i_x:i_x+self.block_size, :]
                elif len(self.ref_img.shape) == 2:
                    # Grayscale img
                    reference_matrix = self.ref_img[i_y:i_y+self.block_size,i_x:i_x+self.block_size]

                reference_block = Block(self, self.block_size, reference_matrix)
                
                # WE MUST BE SURE THAT 
                block_centroid_location_x = i_x+self.block_matrix_center_disp
                possible_displacements_x = [val for val in range(-self.search_max_dist, self.search_max_dist+1) 
                                            if (self.n_ignore_border <= val+block_centroid_location_x and val+block_centroid_location_x < self.img_width-self.n_ignore_border)]

                block_centroid_location_y = i_y+self.block_matrix_center_disp
                possible_displacements_y = [val for val in range(-self.search_max_dist, self.search_max_dist+1) 
                                            if (self.n_ignore_border <= val+block_centroid_location_y and val+block_centroid_location_y < self.img_height-self.n_ignore_border)]
                


                # NOW WE HAVE TO FIND THE BEST MATCH IN THE COMPARISON
                # IMAGE USING THE POSSIBLE DISPLACEMENTS WE CAN DO

                # INITIALIZE BEST MATCH SCORE (THE LOWER THE BETTER)
                # AND MATCH DISPLACEMENTS (BETTER None than 0) BECAUSE
                # WE WILL GET OUTPUT None IF IT FAILS, OTHERWISE
                # IT WILL TAKE THE EXACT SAME PIXEL
                min_match_val = np.inf
                match_disp_x, match_disp_y = None, None


                for disp_x in possible_displacements_x:
                    for disp_y in possible_displacements_y:
                        
                        x_loc = block_centroid_location_x-self.block_matrix_center_disp+disp_x
                        y_loc = block_centroid_location_y-self.block_matrix_center_disp+disp_y
                        comparison_img_matrix=None
                        if len(self.ref_img.shape) == 3:
                            # Color img
                            comparison_img_matrix = self.comp_img[y_loc:y_loc+self.block_size, x_loc:x_loc+self.block_size, :]
                        elif len(self.ref_img.shape) == 2:
                            # Grayscale img
                            comparison_img_matrix = self.comp_img[y_loc:y_loc+self.block_size, x_loc:x_loc+self.block_size]

                        compare_block = Block(self, self.block_size, comparison_img_matrix)

                        comp_score = reference_block.match(compare_block, matcher=self.matcher)
                        if comp_score < min_match_val:
                            min_match_val = comp_score
                            match_disp_x = disp_x
                            match_disp_y = disp_y


                self.flow_field[i_y:self.block_size+i_y, i_x:self.block_size+i_x, 0] = match_disp_x
                self.flow_field[i_y:self.block_size+i_y, i_x:self.block_size+i_x, 1] = match_disp_y


        self.flow_field_obtained = True    




    def get_flow_field(self, ref_img, comp_img):
        self.finder(ref_img, comp_img)
        return self.flow_field
    



    def save_flow_fied_img(self):
        if self.flow_field_obtained is None:
            self.finder()
        
        # TO DO: PLOT IMG WITH THE FLOW FIELD AND SAVE IT
    




        

    