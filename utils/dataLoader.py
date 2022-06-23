import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from TinySSD_Banana.utils.tools import cvtColor
import cv2


class MyDataSet(Dataset):
    def __init__(self, content_lines, img_r, mode):
        """
        :param content_lines: Each line of the processed data: the first column is the image name (absolute path + suffix), followed by 4N numbers, indicating that the target sits on the upper and lower right coordinates,
        The same target array is separated by commas, and different targets are separated by spaces
        :param img_r: network input image size: int: 256
        :param mode: 'train' or 'test'
        """
        super(MyDataSet, self).__init__()
        self.content_lines = content_lines
        self.r = img_r
        self.len = len(content_lines)
        self.mode = mode

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        index = index % self.len
        line = self.content_lines[index].split()

        # ------------------------------------------------------#
        #                  Image Processing
        # ------------------------------------------------------#
        img = Image.open(line[0])
        img = cvtColor(img)
        iw, ih = img.size
        scale = min(self.r/iw, self.r/ih)
        nw = round(iw * scale)
        nh = round(ih * scale)
        dx = (self.r - nw) // 2 # one of them is 0
        dy = (self.r - nh) // 2 # one of them is 0

        img = img.resize((nw, nh), Image.BILINEAR) # bilinear difference
        image_data = Image.new('RGB', (self.r, self.r), (128, 128, 128))
        image_data.paste(img, (dx, dy))

        flip = False
        if self.mode.lower() == 'train':
            # ------------------------------------------------------#
            #                  horizontal flip
            # ------------------------------------------------------#
            flip = np.random.rand() > 0.5
            if flip: image_data = image_data.transpose(Image.FLIP_LEFT_RIGHT)

            # ------------------------------------------------------#
            # Gamut distortion
            # ------------------------------------------------------#
            hue = 0.1
            sat = 1.3
            val = 1.3
            hue = self.rand(-hue, hue)
            sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
            val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
            x = cv2.cvtColor(np.array(image_data, np.float32) / 255, cv2.COLOR_RGB2HSV)
            # The three channels of HSV are Hue H, Saturation S, Lightness V H(0~360) S(0~1) V(0~1)
            x[..., 0] += hue * 360
            x[x[..., 0] > 360, 0] = 360
            x[x[..., 0] < 0, 0] = 0
            x[..., 1] *= sat
            x[..., 2] *= val
            x[..., 1:][x[..., 1:] > 1] = 1
            x[x < 0] = 0
            image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)

            # ------------------------------------------------------#
            # channel shuffling
            # ------------------------------------------------------#
            channels = [0, 1, 2]
            np.random.shuffle(channels)
            image_data[:] = image_data[..., channels]


        _transform = transforms. Compose([
            transforms.ToTensor(), # PIL int -> tensor[0~1] floating point is not normalized
        ])
        image_data = _transform(image_data)

        # ------------------------------------------------------#
        # border processing
        # ------------------------------------------------------#
        # (o, 5) xmin xmax ymin ymax class
        boxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
        if len(boxes) > 0:
            np.random.shuffle(boxes)
            boxes = boxes.astype(np.float32)
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + dx
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + dy
            if flip: boxes[:, [0, 2]] = self.r - boxes[:, [2, 0]]
            boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
            boxes[:, 2][boxes[:, 2] > self.r] = self.r
            boxes[:, 3][boxes[:, 3] > self.r] = self.r
            box_w = boxes[:, 2] - boxes[:, 0] # top left, bottom right -> center width and height
            box_h = boxes[:, 3] - boxes[:, 1] # top left, bottom right -> center width and height
            boxes = boxes[np.logical_and(box_w > 1, box_h > 1)] # discard invalid box
            boxes[:, :4] = boxes[:, :4] / self.r # The bounding box size is normalized to 0~1

        # image_data: (3, 320, 320), boxes: (o, 5) or (0,)
        # boxes: cx cy w h c
        # image_data -> tensor boxes -> array
        return image_data, boxes

    @staticmethod
    def rand(a=0., b=1.):
        return np.random.rand() * (b - a) + a


def dataset_collate(batch):
    """ Use collate_fn in DataLoader
    The reason why this parameter needs to be passed in is because the dataloader returns img and boxes, but the boxes of each picture are not fixed, so it is necessary to encapsulate a layer outside to make the shape fixed
    :param batch: list(tuple(tensor(3, 320, 320), ndarray(o, 5) or no target: ndarray([])))
    :return: (bs, 3, 320, 320) (bs, 100, 5) 5: class xmin ymin xmax ymax
    """
    m = 100 # Limit a picture to a maximum of 100 targets
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        if box.ndim == 1: # if there is no target
            bboxes.append(torch.full((m, 5), -1, dtype=torch.float32))
        else: # when there is at least one target
            box[:] = box[:, [4,0,1,2,3]] # put the category in the first one
            bboxes.append(torch.from_numpy(np.pad(box, ((0,m-box.shape[0]),(0,0)), constant_values=-1)))
    return torch.stack(images, 0), torch.stack(bboxes, 0)


if __name__ == '__main__':
    from TinySSD_Banana.utils.tools import get_classes, show_bboxes
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    input_shape = 256
    class_names, _ = get_classes('C:\\Users\Marwan\Downloads\Tiny-SSD-master\Tiny-SSD-master\model_data\\voc_classes.txt')
    with open('C:\\Users\Marwan\Downloads\Tiny-SSD-master\Tiny-SSD-master\\2077_trainval.txt') as f:
        train_lines = f.readlines()

    train_dataset = MyDataSet(train_lines, input_shape, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                              pin_memory=True, drop_last=True, collate_fn=dataset_collate)
    dataiter = iter(train_loader)

    fig = plt.figure(figsize=(16, 8))
    columns = 4
    rows = 2
    inputs, labels = dataiter.next()
    for idx in range(columns * rows):
        ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
        label_boxes = labels[idx, labels[idx][:, 0] >= 0]
        show_bboxes(ax, label_boxes[:,1:]*input_shape, [class_names[int(item)] for item in label_boxes[:,0].tolist()])
        plt.imshow(inputs[idx].permute(1,2,0))
        plt.pause(1)
