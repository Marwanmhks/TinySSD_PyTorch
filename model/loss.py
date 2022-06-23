from torch.nn import functional as F
import torch


def focal_loss(cls_preds, cls_labels, gamma=2):
    """ Calculate class loss
    :param cls_preds: (bs, anchors, 1+c)
    :param cls_labels: (bs, anchors)
    :param gamma: int default->2
    :return: (bs,)
    """
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls_preds = cls_preds.reshape(-1, num_classes)
    cls_labels = cls_labels.reshape(-1)
    cls_preds = F.softmax(cls_preds, dim=1)
    x = cls_preds[torch.arange(cls_labels.shape[0]), cls_labels]
    cls = -(1 - x) ** gamma * torch.log(x)
    return cls.reshape(batch_size, -1).mean(dim=1)


def smooth_l1_loss(bbox_preds, bbox_labels, bbox_masks, sigma=1):
    """ Calculate class border loss
    :param bbox_preds: (bs, anchors*4)
    :param bbox_labels: (bs, anchors*4)
    :param bbox_masks: (bs, anchors*4)
    :param sigma: int default->1
    :return: (bs,)
    """
    bbox_preds = bbox_preds * bbox_masks
    bbox_labels = bbox_labels * bbox_masks
    active_items = bbox_masks.sum(axis=1) + 1e-5
    beta = 1. / (sigma ** 2)
    diff = torch.abs(bbox_preds - bbox_labels)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return torch.sum(loss, dim=1) / active_items


def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks, alpha = 0.01):
    """ Overall loss function
    :param cls_preds: (bs, anchors, 1+c)
    :param cls_labels: (bs, anchors)
    :param bbox_preds: (bs, anchors*4)
    :param bbox_labels: (bs, anchors*4)
    :param bbox_masks: (bs, anchors*4)
    :param alpha: Coefficient to balance class loss and border loss
    :return: int
    """
    cls_loss = focal_loss(cls_preds, cls_labels)
    bbox_loss = smooth_l1_loss(bbox_preds, bbox_labels, bbox_masks)
    return (cls_loss + alpha * bbox_loss).mean()


@torch.no_grad()
def cls_eval(cls_preds, cls_labels):
    """ Category classification loss assessment
    :param cls_preds: (bs, anchors, 1+c)
    :param cls_labels: (bs, anchors)
    :return: int
    """
    return focal_loss(cls_preds, cls_labels).mean()


@torch.no_grad()
def bbox_eval(bbox_preds, bbox_labels, bbox_masks, alpha = 0.01):
    """ Boundary regression loss evaluation
    :param bbox_preds: (bs, anchors*4)
    :param bbox_labels: (bs, anchors*4)
    :param bbox_masks: (bs, anchors*4)
    :param alpha: Coefficient to balance class loss and border loss
    :return: int
    """
    return alpha * smooth_l1_loss(bbox_preds, bbox_labels, bbox_masks).mean()
