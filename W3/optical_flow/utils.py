import cv2


import numpy as np




## Extracted from team 2 2022/2023 
## thx Adrià Molina dear friend 
## The block size must be the same as used to generate the optical fow image
def postprocess(self, optical_flow_image, variance_thr: float = 100, color_diff_thr: float = 8, window_size: int = 3, block_size: int=26):
    new_flow = optical_flow_image.copy()
    # Interpolate the optical flow in border pixels from neighborhood
    # For instance, for each pixel of the left border, we calculate 
    # the optical flow using the pixels of (i, block_size), (i-1, block_size), (i+1, block_size)
    # by interpolation
    for i in range(0, optical_flow_image.shape[0], block_size):
        new_flow[i:i+block_size, :block_size, :] = (optical_flow_image[i-1, block_size, :] + optical_flow_image[i, block_size, :] + optical_flow_image[i+1, block_size, :]) / 3
        new_flow[i:i+block_size, -block_size*2:, :] = (optical_flow_image[i-1, -block_size*2-1, :] + optical_flow_image[i, -block_size*2-1, :] + optical_flow_image[i+1, -block_size*2-1, :]) / 3

    for j in range(0, optical_flow_image.shape[1], block_size):
        new_flow[:block_size, j:j+block_size, :] = (optical_flow_image[block_size, j-1, :] + optical_flow_image[block_size, j, :] + optical_flow_image[block_size, j+1, :]) / 3
        new_flow[-block_size:, j:j+block_size, :] = (optical_flow_image[-block_size-1, j-1, :] + optical_flow_image[-block_size-1, j, :] + optical_flow_image[-block_size-1, j+1, :]) / 3

    # Special corner cases 
    new_flow[:block_size, :block_size, :] = (new_flow[block_size, block_size, :] + new_flow[block_size, 0, :] + new_flow[0, block_size, :]) / 3
    new_flow[:block_size, -block_size:, :] = (new_flow[block_size, -block_size-1, :] + new_flow[block_size, -block_size*2-1, :] + new_flow[0, -block_size-1, :]) / 3
    new_flow[-block_size:, :block_size, :] = (new_flow[-block_size-1, block_size, :] + new_flow[-block_size-1, 0, :] + new_flow[-block_size*2-1, block_size, :]) / 3
    new_flow[-block_size:, -block_size:, :] = (new_flow[-block_size-1, -block_size-1, :] + new_flow[-block_size-1, -block_size*2-1, :] + new_flow[-block_size*2-1, -block_size-1, :]) / 3
    optical_flow_image = new_flow
    new_flow = optical_flow_image.copy()
    hsv = cv2.cvtColor(optical_flow_image, cv2.COLOR_BGR2HSV)
    hue = hsv[:,:,2]
    # save the hue image
    cv2.imwrite("hue.png", hue)

    # Define the window size and threshold for computing the mask
    window_size = block_size*window_size
    mask = np.zeros_like(new_flow[:,:,0], dtype=np.uint8)

    # Compute the var of the flow in each window
    variance = np.zeros_like(new_flow[:,:,0])
    for i in range(0, new_flow.shape[0], block_size):
        for j in range(0, new_flow.shape[1], block_size):
            x_min = max(0, j - window_size)
            x_max = min(new_flow.shape[1], j + window_size)
            y_min = max(0, i - window_size)
            y_max = min(new_flow.shape[0], i + window_size)
            
            # # Compute the variance of the flow in the window only in the masked area
            window = hue[y_min:y_max, x_min:x_max]
            variance[i:i+block_size,j:j+block_size] = np.var(window)

            current_color = np.mean(optical_flow_image[i:i+block_size, j:j+block_size, :], axis=(0, 1))
            
            # Sum all the colors in the neighborhood and divide by 8 to get the mean color
            neighbors_color = []
            for k in range(i-block_size, i+block_size+1, block_size):
                for l in range(j-block_size, j+block_size+1, block_size):
                    if k == i and l == j:
                        continue
                    k = max(0, k)
                    k = min(optical_flow_image.shape[0]-1, k)
                    l = max(0, l)
                    l = min(optical_flow_image.shape[1]-1, l)
                    neighbors_color.append(np.mean(optical_flow_image[k:k+block_size, l:l+block_size, :], axis=(0, 1)))

            neighbors_color_mean = np.mean(neighbors_color, axis=0)

            # If the average color is different from the mean color, we interpolate the optical flow
            if np.linalg.norm(current_color - neighbors_color_mean) > color_diff_thr:
                mask[i:i+block_size, j:j+block_size] = 1

    # Create a binary mask based on the var threshold
    var_mask = (variance > variance_thr).astype(np.uint8)
    var_mask = cv2.dilate(var_mask, np.ones((block_size*3,block_size*3), np.uint8), iterations=1)
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
