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


class RNet(nn.Module):
    """MTCNN RNet.

    Keyword Arguments:
        pretrained {bool} -- Whether or not to load saved pretrained weights (default: {True})
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 28, kernel_size=3)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=2)
        self.prelu3 = nn.PReLU(64)
        self.dense4 = nn.Linear(576, 128)
        self.prelu4 = nn.PReLU(128)
        self.dense5_1 = nn.Linear(128, 2)
        self.softmax5_1 = nn.Softmax(dim=1)
        self.dense5_2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense4(x.view(x.shape[0], -1))
        x = self.prelu4(x)
        a = self.dense5_1(x)
        a = self.softmax5_1(a)
        b = self.dense5_2(x)
        return b, a


class ONet(nn.Module):
    """MTCNN ONet.

    Keyword Arguments:
        pretrained {bool} -- Whether or not to load saved pretrained weights (default: {True})
    """

    def __init__(self, pretrained=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
        self.prelu4 = nn.PReLU(128)
        self.dense5 = nn.Linear(1152, 256)
        self.prelu5 = nn.PReLU(256)
        self.dense6_1 = nn.Linear(256, 2)
        self.softmax6_1 = nn.Softmax(dim=1)
        self.dense6_2 = nn.Linear(256, 4)
        self.dense6_3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense5(x.view(x.shape[0], -1))
        x = self.prelu5(x)
        a = self.dense6_1(x)
        a = self.softmax6_1(a)
        b = self.dense6_2(x)
        c = self.dense6_3(x)
        return b, c, a


class MTCNN(nn.Module):
    images = torch.Tensor()
    image_indexes = torch.Tensor()

    def __init__(self, device=None):
        super().__init__()

        self.thresholds = [0.6, 0.7, 0.7]

        self.p_net = PNet()
        self.p_net.eval()
        self.r_net = RNet()
        self.r_net.eval()

        self.device = torch.device('cpu')
        if device:
            self.device = device
            self.to(device)

    def load_state(self, p_net_state=None, r_net_state=None):
        if p_net_state:
            self.p_net.load_state_dict(p_net_state)
        if r_net_state:
            self.r_net.load_state_dict(r_net_state)

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

        self.images = images

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

        offset = 0
        for scale in scales:
            im_data = tools.image_resample(images, (int(h * scale + 1), int(w * scale + 1)))
            im_data = (im_data - 127.5) * 0.0078125
            reg, label = self.p_net(im_data)

            boxes_scale, image_indexes_scale = tools.generate_bounding_Box(reg, label[:, 1], scale, self.thresholds[0])
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

        self.image_indexes = image_indexes

        regw = boxes[:, 2] - boxes[:, 0]
        regh = boxes[:, 3] - boxes[:, 1]
        qq1 = boxes[:, 0] + boxes[:, 5] * regw
        qq2 = boxes[:, 1] + boxes[:, 6] * regh
        qq3 = boxes[:, 2] + boxes[:, 7] * regw
        qq4 = boxes[:, 3] + boxes[:, 8] * regh
        boxes = torch.stack([qq1, qq2, qq3, qq4, boxes[:, 4]]).permute(1, 0)
        boxes = tools.rerec(boxes)
        return boxes

    @torch.no_grad()
    def detect_r_net(self, boxes):
        h, w = self.images.shape[2:4]
        y, ey, x, ex = tools.pad(boxes, w, h)
        if len(boxes) > 0:
            im_data = []
            for k in range(len(y)):
                if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                    img_k = self.images[self.image_indexes[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                    im_data.append(tools.image_resample(img_k, (24, 24)))
            im_data = torch.cat(im_data, dim=0)
            im_data = (im_data - 127.5) * 0.0078125

            # This is equivalent to out = rnet(im_data) to avoid GPU out of memory.
            out = self.r_net(im_data)

            out0 = out[0].permute(1, 0)
            out1 = out[1].permute(1, 0)
            score = out1[1, :]
            ipass = score > self.thresholds[1]
            boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
            image_inds = self.image_indexes[ipass]
            mv = out0[:, ipass].permute(1, 0)

            # NMS within each image
            pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
            boxes, image_inds, mv = boxes[pick], image_inds[pick], mv[pick]
            boxes = tools.bbreg(boxes, mv)
            boxes = tools.rerec(boxes)
            return boxes
