"""
coding:utf-8
Author: Marwan
Date: 2022/06/16
Time: 05:35:15
"""
import torch

def multibox_prior(data, sizes, ratios):
    """ Generate anchor boxes per pixel for feature maps: Generate anchor boxes with different shapes centered on each pixel
    Note: In practice, we only consider combinations containing either s1 or r1 , i.e. all combinations of sizes[0] and ratios plus all combinations of ratios[0] and sizes
    Remove a repetition, a pixel has a total of s+r-1 anchor boxes, such as s=2, r=3, that is, there are 4 anchor boxes
    :param data: (batch_size, channels, fw, fh) is the feature map in network circulation. Generally, when the anchor frame is preset, batch_size = channels = 1
    :param sizes: A list of scales for different side lengths
    :param ratios: A list to put in different aspect ratios (relative to the original image)
    :return: (fw*fh*(len(s)+len(r)-1), 4), which generates a series of anchor boxes for one "pixel" of each feature map
    """
    in_height, in_width = data.shape[-2:]   # From the penultimate to the last, that is, the last two
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # To move the anchor point to the center of the pixel, the offset needs to be set.
    # Since a pixel has a height of 1 and a width of 1, we choose to offset our center by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height   # Scaled steps in yaxis
    steps_w = 1.0 / in_width    # Scaled steps in xaxis

    # Generate all center points of anchor boxes
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h    # (fh,) 0~1
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w     # (fw, ) 0~1
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')    # (fh, fw), (fh, fw) 0~1
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)     # Pull into one dimension, the two together with zip represent the center coordinates of each anchor frame (0~1)

    # Generate "boxes_per_pixel" height and width, (4,) 0~1
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]), sizes[0] * torch.sqrt(ratio_tensor[1:])))
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]), sizes[0] / torch.sqrt(ratio_tensor[1:])))

    # (fw*fh*bpp, 4), where divide by 2 to get half height and half width -> convert from width to coordinates
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2
    # The meaning of the 4 values inside is, with the origin as the center, xmin xmax ymin ymax of the anchor box, the next step is to put the anchor box in the correct position

    # (fw*fh*bpp, 4)
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    # The meaning of the 4 values in it is that xcenter ycenter xcenter ycenter means that 1, 3 columns are repeated, 2, 4 columns are repeated only for convenience
    '''
    The difference between repeat and repeat_interleave is that, for example, 1, 2, 3, repeat 3 times
    repeat: 1,2,3,1,2,3,1,2,3 [repeat in blocks]
    The difference between repeat and repeat_interleave is: 1,1,1,2,2,2,3,3,3 [internal repeat]
    '''
    output = out_grid + anchor_manipulations
    return output

def generate_anchors(fmp_list, sizes, ratios):
    """ Generate a series of anchor boxes for the feature map
    :param fmp_list: The size of the feature map output by each layer (only a square map), such as [32,16,8,4,1]
    :param sizes: The size of the anchor box of each layer, such as [[0.1, 0.15], [0.25, 0.3], [0.35, 0.4], [0.5, 0.6], [0.7, 0.9]]
    :param ratios: The aspect ratio of the anchor frame of each layer, such as [[1, 2, 0.5]] * 5
    :return: Combine all anchor boxes to form (N, 4), for example tensor.shape = (anchors, 4):
    """
    anchors = [None]*len(fmp_list)
    for i, fmp in enumerate(fmp_list):
        tmp = torch.zeros((1,1,fmp,fmp))
        anchors[i] = multibox_prior(tmp, sizes[i], ratios[i])
    anchors = torch.cat(anchors, dim=0)
    return anchors

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from TinySSD_Banana.utils.tools import *

    r = get_image_size('../model_data/image_size.txt')
    img = Image.open('../VOCdevkit/test.jpg').convert('RGB')
    img = img_preprocessing(img,r)
    fh, fw = 1, 1

    X = torch.rand(size=(1, 1, fh, fw))
    anchor_sizes = get_anchor_info('../model_data/anchor_sizes.txt')
    anchor_ratios = get_anchor_info('../model_data/anchor_ratios.txt')
    anchors_boxes = multibox_prior(X, anchor_sizes[4], anchor_ratios[0])

    index = np.random.choice(fw*fh*4, 50)
    fig = plt.imshow(img.permute(1,2,0))
    show_bboxes(fig.axes, anchors_boxes[index] * r)

    pass
