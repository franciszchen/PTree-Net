import numpy as np
import math
from skimage.filters.rank import entropy
from skimage.morphology import disk

import torch

def find_max_map(tmp_map):
    """
    local location within tmp_map
    """
    loc_tensor = (tmp_map == torch.max(tmp_map)).nonzero() # find the location of np.array
    row = loc_tensor[0, 0].item() # location in float
    col = loc_tensor[0, 1].item()
    tmp_map[row, col] = -100 # update the map
    return (row, col), tmp_map 

def find_top_focal(heatmap, ratio, attention_patch_size, mask_ratio=3, patch_num_limit=8):
    """
    return : location_list: a list of local locations within heatmap
    params : 
            heatmap: torch tensor
    """
    expected_patch_num = math.ceil(ratio*heatmap.shape[0]*heatmap.shape[1]/(attention_patch_size*attention_patch_size)) # expected number of cropped patches
    expected_patch_num = min(patch_num_limit, expected_patch_num)
    valid_mask = torch.zeros(heatmap.shape, dtype=torch.uint8)
    crop_mask = torch.zeros(heatmap.shape, dtype=torch.uint8)
    location_list = []
    while len(location_list) < expected_patch_num:
        location, heatmap = find_max_map(tmp_map=heatmap) # possible location, and update heatmap        
        if valid_mask[location] == 0: # location is still valid
            location_list.append(location)
            valid_mask[
                int(location[0]-attention_patch_size*mask_ratio/2):int(location[0]+attention_patch_size*mask_ratio/2), 
                int(location[1]-attention_patch_size*mask_ratio/2):int(location[1]+attention_patch_size*mask_ratio/2)] = 1 # add invalid mask
            crop_mask[
                int(location[0]-attention_patch_size/2):int(location[0]+attention_patch_size/2), 
                int(location[1]-attention_patch_size/2):int(location[1]+attention_patch_size/2)] = 1 # add invalid mask
    return location_list, torch.sum(crop_mask).item()/(crop_mask.shape[0]*crop_mask.shape[1])


if __name__ == '__main__':
    x = torch.zeros([10, 9], dtype=torch.float32)
    x[3,6] = 1
    x[4,5] = 2

    print(find_max_map(tmp_map=x))
