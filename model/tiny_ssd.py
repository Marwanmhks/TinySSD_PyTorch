import torch
from torch.nn import functional as F
from TinySSD_Banana.model.anchor_generate import generate_anchors
from TinySSD_Banana.model.for_inference import multibox_detection
from TinySSD_Banana.model.net import TinySSD
from TinySSD_Banana.utils.tools import get_classes, get_anchor_info, try_gpu, img_preprocessing, get_image_size
import os


class Tiny_SSD(object):
    _defaults = {
        "anchor_sizes_path": 'C:\\Users\Marwan\PycharmProjects\TinySSD_Banana\TinySSD_Banana\model_data\\anchor_sizes.txt',
        "anchor_ratios_path": 'C:\\Users\Marwan\PycharmProjects\TinySSD_Banana\TinySSD_Banana\model_data\\anchor_ratios.txt',
        "image_size_path": 'C:\\Users\Marwan\PycharmProjects\TinySSD_Banana\TinySSD_Banana\model_data\\image_size.txt',
        "model_path": 'C:\\Users\Marwan\PycharmProjects\TinySSD_Banana\TinySSD_Banana\model_data\\result.pt',
        "classes_path": 'C:\\Users\Marwan\PycharmProjects\TinySSD_Banana\TinySSD_Banana\model_data\\voc_classes.txt',
        "nms_threshold": 0.3,
    }

    @classmethod
    def get_default(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, isTraining=False, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.sizes = get_anchor_info(self.anchor_sizes_path)
        self.ratios = get_anchor_info(self.anchor_ratios_path)
        self.r = get_image_size(self.image_size_path)
        self.feature_map = [self.r // 8, self.r // 16, self.r // 32, self.r // 64, 1]
        # ---------------------------------------------------------
        # Generate a priori anchor box
        # ---------------------------------------------------------
        if len(self.sizes) != len(self.ratios):
            self.ratios = [self.ratios[0]] * len(self.sizes)
        self.anchors_perpixel = len(self.sizes[0]) + len(self.ratios[0]) - 1
        self.anchors = generate_anchors(self.feature_map, self.sizes, self.ratios)

        # ---------------------------------------------------------
        # load network
        # ---------------------------------------------------------
        self.name_classes, self.num_classes = get_classes(self.classes_path)
        self.device, self.net = try_gpu(), TinySSD(app=self.anchors_perpixel, cn=self.num_classes)
        self.net.load_state_dict(torch.load(self.model_path))

        # Put GPU into evaluation mode
        self.anchors = self.anchors.to(self.device)
        self.net = self.net.to(self.device)
        self.net.eval()

    def inference(self, image):
        """ Pass in an Image and output the prediction result
        :param image: PIL Image
        :return: list -> (chose, 10): class conf bbx bby bbx bby anx any anx any
        """
        iw, ih = image.size  # The return is actually (w, h)
        image = img_preprocessing(image, self.r).unsqueeze(0)  # (1, 3, r, r)
        with torch.no_grad():
            cls_preds, bbox_preds = self.net(image.to(self.device))  # (1, anchors, (1+c)) (1, anchors*4)
            cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)  # (1, (1+c), anchors)
            output = multibox_detection(cls_probs, bbox_preds, self.anchors, self.nms_threshold)  # (1, anchors, 10)
            idx = [i for i, row in enumerate(output[0]) if row[0] >= 0]  # exclude background
            result = output[0, idx]  # (chose, 10): class conf bbx bby bbx bby anx any anx any

            # The following is to restore the prediction box to the original size of the image
            scale = min(self.r / ih, self.r / iw)
            dx = (self.r - round(iw * scale)) // 2  # one of them is 0
            dy = (self.r - round(ih * scale)) // 2  # one of them is 0
            result[:, 2:] *= self.r
            result[:, [2, 4, 6, 8]] -= dx  # bbxmin bbxmax anxmin anxmax
            result[:, [3, 5, 7, 9]] -= dy  # bbymin bbymax anymin anymax
            result[:, 2:] /= scale
            result[result < 0] = 0
            result[result[:, 4] > iw, 4] = iw  # bbxmax
            result[result[:, 5] > ih, 5] = ih  # bbymax
        return result

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")
        output = self.inference(image)  # (chose, 10)
        for i in range(len(output)):
            predicted_class = self.name_classes[int(output[i, 0])]
            if predicted_class not in class_names: continue  # exclude unwanted classes
            score = str(float(output[i, 1]))
            xmin, ymin, xmax, ymax = output[i, 2:6]
            f.write("%s %s %s %s %s %s\n" % (
                predicted_class, score[:6], str(int(xmin)), str(int(ymin)), str(int(xmax)), str(int(ymax))
            ))
        f.close()
        return
