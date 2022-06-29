import torch
from torch import nn


# ------------------------------------------------- ---------------
# Take branch for prediction
# ------------------------------------------------- ---------------
def gen_cls_predictor(in_channels, app, cn):
    """ Branch used to form class predictions
    :param in_channels: number of input channels
    :param app: anchors per pixel (anchors per pixel, app)
    :param cn: class number
    :return: nn.Conv
    """
    blk = []
    out_channels = app * (cn + 1)  # Add background class
    # blk.extend([nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
    # nn.Conv2d(in_channels, out_channels, 1, 1, 0),
    # ]) # dw pw
    blk.extend([nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                ])
    return nn.Sequential(*blk)


def gen_bbox_predictor(in_channels, app):
    """ Branch used to form anchor box coordinate offset prediction
    :param in_channels: number of input channels
    :param app: anchors per pixel (anchors per pixel, app)
    :return: nn.Conv
    """
    blk = []
    out_channels = app * 4
    # blk.extend([nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
    # nn.Conv2d(in_channels, out_channels, 1, 1, 0),
    # ])
    blk.extend([nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                ])  # dw pw
    return nn.Sequential(*blk)


def concat_preds(preds):
    """ Make the prediction output continuous according to pixel points and anchor boxes. The reason why the channel dimension is put at the end is that each Fmap pixel prediction is expanded to a continuous value, which is convenient for subsequent processing.
    :param preds: For category predictions: list((bs, bpp*(1+c), fhi, fwi)*5. For bounding box predictions: list((bs, app*4, fhi, fwi))*5
    :return: For class prediction(bs, anchors*(c+1)). For bounding box prediction(bs, anchors*4)
    """

    def flatten_pred(pred):
        return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)  # (bs, -1)

    return torch.cat([flatten_pred(p) for p in preds], dim=1)


# ------------------------------------------------- ---------------
# Attention mechanism
# ------------------------------------------------- ---------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Use 1x1 convolution instead of fully connected
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


# ------------------------------------------------- ---------------
# Build the backbone network (simple convolutional network)
# ------------------------------------------------- ---------------
def down_sample_blk(in_channels, out_channels):
    """ Downsampling block (Conv+BN+Relu)*N + Maxpooling
    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :return: nn.Sequential
    """
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    blk.append(cbam_block(out_channels, 4))  # Add attention mechanism
    return nn.Sequential(*blk)


def base_net():
    """ Feature extraction network backbone. Consists of three downsampling blocks
    :return: nn.Sequential
    """
    blk = []
    num_filters = [3, 16, 24, 48]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i + 1]))
    return nn.Sequential(*blk)


def get_blk(i):
    """ for the convenience of constructing network code
    :param i: index
    :return:
    """
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(48, 64)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1, 1))
    else:
        blk = down_sample_blk(64, 64)
    return blk


def blk_forward(X, blk, cls_predictor, bbox_predictor):
    """ Generate branch output between stages
    :param X: input tensor
    :param blk: each network block blk
    :param cls_predictor: Conv, the category output used to generate this blk output
    :param bbox_predictor: Conv, the bounding box offset output used to generate this blk output
    :return: (backbone output, category branch, bounding box offset branch) of this layer
    """
    Y = blk(X)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return Y, cls_preds, bbox_preds


class TinySSD(nn.Module):
    def __init__(self, app, cn):
        """ Overall network
        :param app: the number of anchor boxes assigned to each pixel
        :param cn: total number of categories (excluding background)
        """
        super(TinySSD, self).__init__()
        self.num_classes = cn
        idx_to_in_channels = [48, 64, 64, 64, 64]
        for i in range(5):
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', gen_cls_predictor(idx_to_in_channels[i], app, cn))
            setattr(self, f'bbox_{i}', gen_bbox_predictor(idx_to_in_channels[i], app))

    def forward(self, X):
        """ Neural network forward propagation
        :param X: (bs, 3, w, h)
        :return: (bs, anchors, 1+c), (bs, anchors*4)
        """
        cls_preds, bbox_preds = [None] * 5, [None] * 5
        for i in range(5):
            X, cls_preds[i], bbox_preds[i] = blk_forward(
                X,
                getattr(self, f'blk_{i}'),
                getattr(self, f'cls_{i}'),
                getattr(self, f'bbox_{i}')
            )
        cls_preds = concat_preds(cls_preds)  # (bs, anchors*4)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)  # (bs, anchors, (1+c))
        bbox_preds = concat_preds(bbox_preds)  # (bs, anchors*4)
        return cls_preds, bbox_preds


if __name__ == '__main__':
    net = TinySSD(4, 1)
    from torchsummary import summary

    summary(net.cuda(), (3, 320, 320))
    # summary(net, (3, 320, 320), device='cpu') # No GPU use this
