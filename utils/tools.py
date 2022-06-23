import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

def get_classes(path):
    """ Get the category and the number of categories
    :param path: file path
    :return: list_classes, num_classes
    """
    with open(path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


def get_anchor_info(path):
    """ Get anchor box information (the network has five scale predictions)
    :param path: file path
    :return: 2D list
    """
    with open(path, encoding='utf-8') as f:
        anchor_infos = f.readlines()
    anchor_infos = [list(map(float, c.strip().split())) for c in anchor_infos]
    return anchor_infos

def get_image_size(path):
    """ Get input image size
    :param path: file path
    :return: int
    """
    with open(path, encoding='utf-8') as f:
        r = f.readlines()
    return int(r[0])

def cvtColor(image):
    """ Convert image to RGB image
    :param image: image
    :return: RGBimage
    """
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def img_preprocessing(_img, img_sz):
    """ is used for inference, Image(H,W,C) -> (C,H,W) 0~1
    :param _img: Image
    :param img_sz: the image side length of the input network
    :return: tensor_img (C,H,W) 0~1
    """
    r = img_sz
    if hasattr(_img, 'size'): iw, ih = _img.size #
    else: iw, ih, _ = _img.shape
    scale = min(r / iw, r / ih)
    nw = round(iw * scale)
    nh = round(ih * scale)
    dx = (r - nw) // 2 # one of them is 0
    dy = (r - nh) // 2 # one of them is 0
    _img = _img.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (r, r), (128, 128, 128))
    new_image.paste(_img, (dx, dy))
    transform = transforms.Compose([
        transforms.ToTensor(), # PIL -> tensor [0~1]
    ])
    return transform(new_image)


def show_bboxes(axes, bboxes, labels=None, colors=None):
    """ Draw anchor box in subplot xmin xmax ymin ymax
    :param axes: the canvas passed in
    :param bboxes: [(num_anchors, 4)]
    :param labels: default = None
    :param colors: default = None
    :return: None
    """
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj
    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0],
                             height=bbox[3] - bbox[1], fill=False, edgecolor=color, linewidth=1)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i], va='center', ha='center',
                      fontsize=9, color=text_color, bbox=dict(facecolor=color, lw=0))


def try_gpu():
    """ try to use GPU, if not, return to CPU
    :return:device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class Timer:
    """ Record multiple running times"""
    def __init__(self):
        self.tik = None
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    # python syntax: magic methods generally start and end with double underscores
    def __getitem__(self, idx):
        return self.data[idx]


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

class Animator:
    """For plotting data in animation.(Incrementally plot multiple lines)"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=0, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(8, 6)):
        if legend is None: legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
            # Use a lambda function to capture arguments
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        plt.pause(0.1)  # Because of single thread, otherwise it will get stuck

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()  # clear subgraph
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()  # The reason why it is written as lambda is that I don't want to pass parameters here
        plt.pause(0.1)  # Because of single thread, otherwise it will get stuck
