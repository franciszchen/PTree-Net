import numpy as np
import openslide
import config

"""
# scale_openslide_dict
# level
0 - bottom
1 - mid
2 - tip
"""

def loc_np2openslide(location):
    return (location[1], location[0])

def loc_openslide2np(location):
    return (location[1], location[0])

def loc_transfer_levels(location_orig, level_orig, level_target): 
    """level global of a single image
    level_orig, level_target: str"""
    return (int((location_orig[0]+1)*config.scale_dict_checked[level_target]/config.scale_dict_checked[level_orig] - 1),
        int((location_orig[1]+1)*config.scale_dict_checked[level_target]/config.scale_dict_checked[level_orig] - 1)
        )

def loc_transfer_levels_list(location_orig_list, level_orig, level_target): 
    """level of several images
    level_orig, level_target: str"""
    location_target_list = []
    for location_orig in location_orig_list:
        location_target_list.append(
            loc_transfer_levels(
                location_orig=location_orig, level_orig=level_orig, 
                level_target=level_target)
        )
    return location_target_list

def get_wsi_handle(filepath):
    wsi_handle = openslide.OpenSlide(filepath)
    return wsi_handle

def load_whole_level(wsi_handle, level): 
    # level=0, means the largest slide
    # wsi_handle = openslide.OpenSlide(filepath)
    slide_level = np.array(
        wsi_handle.read_region((0, 0), 
        config.scale_openslide_dict_checked[level], 
        wsi_handle.level_dimensions[config.scale_openslide_dict_checked[level]]))[:, :, :3] # the 4-th channel is empty
    return slide_level

###############
def slide_loc(location_orig, level_orig):
    slide_loc_0 = int((location_orig[0]+1)*config.scale_dict_checked['slide']/config.scale_dict_checked[level_orig] - 1)
    slide_loc_1 = int((location_orig[1]+1)*config.scale_dict_checked['slide']/config.scale_dict_checked[level_orig] - 1)
    return (slide_loc_0, slide_loc_1)

def slide_loc_list(location_orig_list, level_orig):
    slide_loc_list = []
    for location_orig in location_orig_list:
        slide_loc_list.append(
            slide_loc(location_orig=location_orig, level_orig=level_orig)
        )
    return slide_loc_list

def patchC_loc_list(locationC, psizeS):
    loc_list = []
    loc_list.append((locationC[0]-int(psizeS/2), locationC[1]-int(psizeS/2)))
    loc_list.append((locationC[0]-int(psizeS/2), locationC[1]+int(psizeS/2)))
    loc_list.append((locationC[0]+int(psizeS/2), locationC[1]-int(psizeS/2)))
    loc_list.append((locationC[0]+int(psizeS/2), locationC[1]+int(psizeS/2)))
    return loc_list

def crop_locationCS_patch(wsi_handle, level, locationCS, psizeS, psize):
    """
    level: str
    """
    # locationC should be the position of center, at the largest-resolution layer
    if isinstance(psize, (int, float)):
        psize = (int(psize), int(psize))
    if isinstance(psizeS, (int, float)):
        psizeS = (int(psizeS), int(psizeS))
    locationCS = loc_np2openslide(locationCS) # modify location from np to openslide
    # -> left-top corner
    anchor_top = locationCS[0]-int(psizeS[0]/2)
    anchor_left = locationCS[1]-int(psizeS[1]/2)
    # Slide size
    rows = wsi_handle.level_dimensions[config.scale_openslide_dict_checked['slide']][0]
    cols = wsi_handle.level_dimensions[config.scale_openslide_dict_checked['slide']][1]
    # 
    anchor_top = max(0, anchor_top) #
    anchor_left = max(0, anchor_left)
    anchor_top = min(anchor_top, rows-psizeS[0]) #
    anchor_left = min(anchor_left, cols-psizeS[1])

    patch = np.array(
        wsi_handle.read_region(
            location=(anchor_top, anchor_left), 
            level=level, size=psize)
        )[:, :, :3]
    return patch, loc_openslide2np((anchor_top+int(psizeS[0]/2), anchor_left+int(psizeS[0]/2)))

def crop_locationCS_patches(wsi_handle, level, locationCS_list, psizeS, psize):
    """
    level and loc list of expected patches
    """
    if isinstance(psize, (int, float)):
        psize = (int(psize), int(psize))
    if isinstance(psizeS, (int, float)):
        psizeS = (int(psizeS), int(psizeS))
    patch_list = []
    locationCSR_list = []
    for locationCS in locationCS_list:
        # crop patch
        patch, locationCSR = crop_locationCS_patch(
                    wsi_handle=wsi_handle, 
                    level=level, 
                    locationCS=locationCS, 
                    psizeS=psizeS,
                    psize=psize)
        patch_list.append(patch)
        locationCSR_list.append(locationCSR)
    return patch_list, locationCSR_list


def local2global_loc(local_loc, center_loc, half_side):
    loc0 = local_loc[0] + center_loc[0] + 1 - half_side
    loc1 = local_loc[1] + center_loc[1] + 1 - half_side
    global_loc = (int(loc0), int(loc1))
    return global_loc

def local_list2global_loc(local_loc_list, center_loc, half_side):
    global_loc_list = []
    for local_loc in local_loc_list:
        global_loc = local2global_loc(local_loc, center_loc, half_side)
        global_loc_list.append(global_loc)
    return global_loc_list

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    slide_handle = get_wsi_handle(r"a tiff file")

    patch = crop_location_patch(slide_handle, level=2, locationC=(200, 300), patch_size=(256, 256))
    print(patch.shape)
    plt.figure()
    plt.imshow(patch)
    plt.show()



    
