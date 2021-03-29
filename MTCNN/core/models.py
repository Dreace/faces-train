from typing import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import batched_nms

try:
    import MTCNN.core.tools as tools
except:
    import core.tools as tools


class PNet(nn.Module):
    """MTCNN PNet.

    Keyword Arguments:
        pretrained {bool} -- Whether or not to load saved pretrained weights (default: {True})
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3)
        self.prelu2 = nn.PReLU(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.prelu3 = nn.PReLU(32)
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1)
        self.softmax4_1 = nn.Softmax(dim=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        a = self.conv4_1(x)
        a = self.softmax4_1(a)
        b = self.conv4_2(x)
        return b, a


class MTCNN(nn.Module):
    def __init__(self, device=None):
        super().__init__()

        self.p_net = PNet()
        self.p_net.eval()

        self.device = torch.device('cpu')
        if device:
            self.device = device
            self.to(device)

    def load_state(self, p_net_state: OrderedDict[str, torch.Tensor]):
        self.p_net.load_state_dict(p_net_state)

    @torch.no_grad()
    def detect_p_net(self, images):
        if isinstance(images, (np.ndarray, torch.Tensor)):
            if isinstance(images, np.ndarray):
                images = torch.as_tensor(images.copy())

            if isinstance(images, torch.Tensor):
                images = torch.as_tensor(images)

            if len(images.shape) == 3:
                images = images.unsqueeze(0)
        else:
            if not isinstance(images, (list, tuple)):
                images = [images]
            if any(img.size != images[0].size for img in images):
                raise Exception("MTCNN batch processing only compatible with equal-dimension images.")
            images = np.stack([np.uint8(img) for img in images])
            images = torch.as_tensor(images.copy())

        images = images.to(self.device)

        model_data_type = next(self.p_net.parameters()).dtype
        images = images.permute(0, 3, 1, 2).type(model_data_type)

        batch_size = len(images)
        h, w = images.shape[2:4]
        # FIXME minsize=20
        minsize = 20
        m = 12.0 / minsize
        min_length = min(h, w)
        min_length = min_length * m

        # Create scale pyramid
        scale_i = m
        scales = []
        # FIXME factor=0.709
        factor = 0.709
        while min_length >= 12:
            scales.append(scale_i)
            scale_i = scale_i * factor
            min_length = min_length * factor

        # First stage
        boxes = []
        image_indexes = []

        scale_picks = []

        # FIXME thresholds=[0.6, 0.7, 0.7]
        thresholds = [0.6, 0.7, 0.7]
        offset = 0
        for scale in scales:
            im_data = tools.image_resample(images, (int(h * scale + 1), int(w * scale + 1)))
            im_data = (im_data - 127.5) * 0.0078125
            reg, label = self.p_net(im_data)

            boxes_scale, image_indexes_scale = tools.generate_bounding_Box(reg, label[:, 1], scale, thresholds[0])
            boxes.append(boxes_scale)
            image_indexes.append(image_indexes_scale)

            pick = batched_nms(boxes_scale[:, :4], boxes_scale[:, 4], image_indexes_scale, 0.5)
            scale_picks.append(pick + offset)
            offset += boxes_scale.shape[0]

        boxes = torch.cat(boxes, dim=0)
        image_indexes = torch.cat(image_indexes, dim=0)

        scale_picks = torch.cat(scale_picks, dim=0)

        # NMS within each scale + image
        boxes, image_indexes = boxes[scale_picks], image_indexes[scale_picks]

        # NMS within each image
        pick = batched_nms(boxes[:, :4], boxes[:, 4], image_indexes, 0.7)
        boxes, image_indexes = boxes[pick], image_indexes[pick]

        regw = boxes[:, 2] - boxes[:, 0]
        regh = boxes[:, 3] - boxes[:, 1]
        qq1 = boxes[:, 0] + boxes[:, 5] * regw
        qq2 = boxes[:, 1] + boxes[:, 6] * regh
        qq3 = boxes[:, 2] + boxes[:, 7] * regw
        qq4 = boxes[:, 3] + boxes[:, 8] * regh
        boxes = torch.stack([qq1, qq2, qq3, qq4, boxes[:, 4]]).permute(1, 0)
        boxes = tools.rerec(boxes)
        return boxes
