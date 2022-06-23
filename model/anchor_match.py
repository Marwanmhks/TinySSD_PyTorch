import torch

def box_corner_to_center(boxes):
    """ Convert anchor box from (minx, miny, maxx, maxy) to (centerx, centery, width, height)
    :param boxes: (n, 4)
    :return: (n, 4)
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes


def box_center_to_corner(boxes):
    """ Convert anchor box from (centerx, centery, width, height) to (minx, miny, maxx, maxy)
    :param boxes: (n, 4)
    :return: (n, 4)
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes


def box_iou(boxes1, boxes2):
    """ Calculate the IOU between two sets of boxes
    :param boxes1: (n, 4) n refers to the number of preset anchors (anchors)
    :param boxes2: (m, 4) m refers to the preset maximum number of targets in a picture (o)
    :return: (n, m) Each position represents the intersection ratio of the two boxes corresponding to i, j
    """
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) # Calculate area
    areas1 = box_area(boxes1) # (anchors,)
    areas2 = box_area(boxes2) # (o,)

    # Intersection (using the broadcast mechanism)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0) # There will be a negative value if there is no intersection, set to 0
    inter_areas = inters[:, :, 0] * inters[:, :, 1] # The area of the intersection, [:, :, 0] is relative to the width of the intersection, [:, :, 1] is relative to the height of the intersection

    # Union (using the broadcast mechanism)
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas


def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """ Assign GT to some anchor boxes, if iou is not enough, it will not be assigned
    :param ground_truth: (o, 4) o is the number of targets in this image
    :param anchors: (anchors, 4)
    :param device:
    :param iou_threshold: Indicates the threshold for assigning positive samples to the anchor box
    :return: (anchors,) Indicates the subscript of the gt box assigned to the corresponding anchor box, and -1 is not assigned
    """
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]

    # The element x_ij at row i and column j is the IoU of anchor box i and ground-truth bounding box j
    jaccard = box_iou(anchors, ground_truth) # (anchors, o)

    # Record the subscript of the real target (0~o-1) assigned to each anchor box, if it is not assigned, it is -1
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device) # (anchors,)

    # ------------------------ First assign gt to anchors larger than the IOU threshold ---------------- ------------
    # According to the threshold, decide whether to allocate the real bounding box alloc_gt value (0 ~ o-1) o is how many targets actually exist in the picture, most of which are 0 (because there is no intersection with the target, IOU=0, take max is the first :0)
    max_ious, alloc_gt = torch.max(jaccard, dim=1) # (anchors,) (anchors,)
    active_index = max_ious >= iou_threshold
    anchors_bbox_map[active_index] = alloc_gt[active_index]

    # ------------------------ Then assign each gt the most suitable --------------- -------------
    col_discard = torch.full((num_anchors,), -1) # for replacement
    row_discard = torch.full((num_gt_boxes,), -1) # for replacement
    for idx in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard) # If dim is not specified, return the result of multi-dimensional array in one-dimensional array subscript form
        gt_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = gt_idx
        jaccard[:, gt_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map


def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """ Given the anchor box and the GT assigned to it, calculate the offset value (the offset value you want the network to output)
    :param anchors: (anchors, 4)
    :param assigned_bb: (anchors, 4)
    :param eps:
    :return: (anchors, 4) offset_x offset_y offset_w offset_h
    """
    c_anchors = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anchors[:, :2]) / c_anchors[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anchors[:, 2:])
    offset = torch.cat((offset_xy, offset_wh), axis=1)
    return offset


def multibox_target(anchors, labels):
    """ Mark anchor boxes with ground-truth bounding boxes (match GT for some anchor boxes, return what these anchor boxes should be (anchor box offset, mask, category), which belong to the converted final label)
    :param anchors: (anchors, 4)
    :param labels: (bs, 100, 5) 100 is the maximum number of targets preset in this image 5 is class minx miny maxx maxy
    :return: A tuple with 3 parts:
    bbox_offset: (bs, anchors*4)
    bbox_mask: (bs, anchors*4) The elements in it are either 0 or 1
    class_labels: (bs, anchors) represents the category, 0 is the background
    """
    batch_size = labels.shape[0]
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size): # Take out the batch of img processing one by one
        label = labels[i, :, :]
        label = label[label[:, 0] >= 0] # Take out the label that contains the target in the picture (some lines, may be 0: means no target) (!!!)
        if len(label) == 0: # if there is no target
            batch_offset.append(torch.zeros((num_anchors*4,), dtype=torch.float32, device=device))
            batch_mask.append(torch.zeros((num_anchors*4,), dtype=torch.float32, device=device))
            # indices must be long, byte or bool tensors
            batch_class_labels.append(torch.zeros((num_anchors,), dtype=torch.long, device=device))
        else: # when there is a target
            anchor_map_object = assign_anchor_to_bbox(label[:, 1:], anchors, device) # (anchors,)
            bbox_mask = ((anchor_map_object >= 0).float().unsqueeze(-1)).repeat(1, 4) # (anchors, 4)

            # Initialize [assigned class] and [assigned bounding box coordinates]: so 0 is background class
            assigned_cls = torch.zeros(num_anchors, dtype=torch.long, device=device) # (anchors,)
            assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device) # (anchors, 4)

            # Use the ground-truth bounding box to label the category of the anchor box
            # If an anchor box is not assigned, we mark it as background (value 0)
            active_indices = torch.nonzero(anchor_map_object >= 0) # (active_anchors, 1)
            assigned_object_idx = anchor_map_object[active_indices] # (active_anchors, 1)
            assigned_cls[active_indices] = label[assigned_object_idx, 0].long() + 1 # (anchors,) The background class is 0, and the 0 class in txt becomes 1 (!!!)
            assigned_bb[active_indices] = label[assigned_object_idx, 1:] # (anchors, 4)

            # offset conversion (prediction is: offset of cx cy w h)
            # Only the active_anchors anchor boxes are valid, and they must be multiplied by bbox_mask (!!!)
            offset = offset_boxes(anchors, assigned_bb)

            batch_offset.append(offset.reshape(-1)) # (anchors*4,)
            batch_mask.append(bbox_mask.reshape(-1)) # (anchors*4,)
            batch_class_labels.append(assigned_cls) # (anchors,)

    # (bs, anchors*4) (bs, anchors*4) (bs, anchors)
    return torch.stack(batch_offset), torch.stack(batch_mask), torch.stack(batch_class_labels)
