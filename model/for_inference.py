from TinySSD_Banana.model.anchor_match import *
import torch

def offset_inverse(anchors, offset_preds):
    """ [predicted offset] combined with [anchor frame] to get [predicted target frame]
    :param anchors: (num_anchors, 4)
    :param offset_preds: (num_anchors, 4)
    :return: (num_anchors, 4)
    """
    anc = box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = box_center_to_corner(pred_bbox)
    return predicted_bbox


def nms(boxes, scores, iou_threshold):
    """ Non-Maximum Suppression: Sort the confidence of predicted bounding boxes
    :param boxes: (num_anchors, 4)
    :param scores: (num_anchors,)
    :param iou_threshold:
    :return: the subscript of the remaining boxes
    """
    B = torch.argsort(scores, dim=-1, descending=True) # Return sorted subscripts
    keep = [] # keep the metrics of the predicted bounding box
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break

        # Calculate the iou of [the anchor box with the maximum confidence in the current cycle] and [all anchor boxes in the current set], and the result is converted into a 1-dimensional vector
        iou = box_iou(boxes[i, :].reshape(-1, 4), boxes[B[1:], :].reshape(-1, 4)).reshape(-1) # class-insensitive NMS (!!!)

        # Exclusions greater than the iou threshold, the subscripts of the remaining anchor boxes
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)

        # The reason for adding 1 is because the two boxes sent to the box_iou function are 0 and [1:], but the results sent to the function start from 0
        # Add 1 to restore the position of [1:] after the box_iou function ends
        B = B[inds + 1] # Yes, even if inds is empty, it can be, that is, the subscript can be indexed with an empty list, and the result is also tensor([])
    return torch.tensor(keep, device=boxes.device)


def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5, object_threshold=0.1, max_object_num=100):
    """ Convert the output value of the network into the information we need
    :param cls_probs: (bs, classes, anchors) -> (bs, 1+c, anchors) background + number of classes
    :param offset_preds: (bs, anchors*4) -> (bs, 4 * anchors)
    :param anchors: (anchors, 4)
    :param nms_threshold: NMS threshold
    :param object_threshold: The preset threshold that is considered to be the target
    :param max_object_num: preset maximum number of output objects
    :return: (bs, max_object_num, 10) class conf minx miny maxx maxy aminx aminy amaxx amaxy are 0~1 except for the class
    """
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4) # ((1+c), anchors) (anchors, 4)

        # Take the maximum confidence of each anchor box for different classes, starting from line 1 means no background (line 0), most of conf is a very small number (large numbers are in line 0, referring to the background), if a single class is identified, id All 0s (because there is only 1 row)
        conf, class_id = torch.max(cls_prob[1:], axis=0) # (anchors,) (anchors,)

        # chose_idx refers to the subscript 0~anchors-1
        chose_idx = torch.nonzero(conf>object_threshold).reshape(-1) # (chose1,) first filter

        predicted_bb = offset_inverse(anchors, offset_pred) # (chose1, 4)
        keep = nms(predicted_bb[chose_idx], conf[chose_idx], nms_threshold) # (chose2,)
        chose_idx = chose_idx[keep.long()] # After nms, choose_idx has been sorted in descending order by conf (chose2,) the second filter

        if len(chose_idx) > max_object_num: # Because it is necessary to make up a fixed shape batch output [larger than to truncate]
            chose_idx = chose_idx[:max_object_num]

        class_id = class_id[chose_idx].unsqueeze(1) # (chose2, 1)
        conf = conf[chose_idx].unsqueeze(1) # (chose2, 1)
        predicted_bb = predicted_bb[chose_idx] # (chose2, 4)
        chosen_anchors = anchors[chose_idx] # # (chose2, 4)

        if len(chose_idx) < max_object_num: # Because there is enough to make up a fixed shape batch output [not enough to fill]
            supplement_num = max_object_num - len(chose_idx)
            class_id = torch.cat((class_id, torch.full((supplement_num, 1), -1, device=device)), dim=0) # extra padding -1 (background)
            conf = torch.cat((conf, torch.zeros((supplement_num, 1), device=device)), dim=0)
            predicted_bb = torch.cat((predicted_bb, torch.zeros((supplement_num, 4), device=device)), dim=0)
            chosen_anchors = torch.cat((chosen_anchors, torch.zeros((supplement_num, 4), device=device)), dim=0)

        # assembly result
        pred_info = torch.cat((class_id, conf, predicted_bb, chosen_anchors), dim=1)
        out.append(pred_info)
    return torch.stack(out) # (bs, max_object_num, 10)