import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd.variable import Variable
from torch.nn.functional import interpolate

transform = transforms.ToTensor()


def convert_image_to_tensor(image):
    """convert an image to pytorch tensor

        Parameters:
        ----------
        image: numpy array , h * w * c

        Returns:
        -------
        image_tensor: pytorch.FloatTensor, c * h * w
        """
    # image = image.astype(np.float32)
    return transform(image)
    # return transform(image)


def convert_chwTensor_to_hwcNumpy(tensor):
    """convert a group images pytorch tensor(count * c * h * w) to numpy array images(count * h * w * c)
            Parameters:
            ----------
            tensor: numpy array , count * c * h * w

            Returns:
            -------
            numpy array images: count * h * w * c
            """

    if isinstance(tensor, Variable):
        return np.transpose(tensor.data.numpy(), (0, 2, 3, 1))
    elif isinstance(tensor, torch.FloatTensor):
        return np.transpose(tensor.numpy(), (0, 2, 3, 1))
    else:
        raise Exception("covert b*c*h*w tensor to b*h*w*c numpy error.This tensor must have 4 dimension.")


def compute_accuracy(prob_cls, gt_cls):
    prob_cls = torch.squeeze(prob_cls)
    gt_cls = torch.squeeze(gt_cls)

    # we only need the detection which >= 0
    mask = torch.ge(gt_cls, 0)
    # get valid element
    valid_gt_cls = torch.masked_select(gt_cls, mask)
    valid_prob_cls = torch.masked_select(prob_cls, mask)
    size = min(valid_gt_cls.size()[0], valid_prob_cls.size()[0])
    prob_ones = torch.ge(valid_prob_cls, 0.6).float()
    right_ones = torch.eq(prob_ones, valid_gt_cls).float()

    # if size == 0 meaning that your gt_labels are all negative, landmark or part

    return torch.div(torch.mul(torch.sum(right_ones), float(1.0)),
                     float(size))  # divided by zero meaning that your gt_labels are all negative, landmark or part


def image_resample(img, sz):
    im_data = interpolate(img, size=sz, mode="area")
    return im_data


def generate_bounding_Box(reg, label, scale, thresh: float):
    stride = 2
    cell_size = 12

    reg = reg.permute(1, 0, 2, 3)

    mask = label >= thresh
    mask_inds = mask.nonzero()
    image_inds = mask_inds[:, 0]
    score = label[mask]
    reg = reg[:, mask].permute(1, 0)
    bb = mask_inds[:, 1:].type(reg.dtype).flip(1)
    q1 = ((stride * bb + 1) / scale).floor()
    q2 = ((stride * bb + cell_size - 1 + 1) / scale).floor()
    boundingbox = torch.cat([q1, q2, score.unsqueeze(1), reg], dim=1)
    return boundingbox, image_inds


def rerec(bboxA):
    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]

    l = torch.max(w, h)
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
    bboxA[:, 2:4] = bboxA[:, :2] + l.repeat(2, 1).permute(1, 0)

    return bboxA


class LossFn:
    def __init__(self, cls_factor=1, box_factor=1, landmark_factor=1):
        # loss function
        self.cls_factor = cls_factor
        self.box_factor = box_factor
        self.land_factor = landmark_factor
        self.loss_cls = nn.BCELoss()  # binary cross entropy
        self.loss_box = nn.MSELoss()  # mean square error
        self.loss_landmark = nn.MSELoss()

    def class_loss(self, gt_label, pred_label):
        pred_label = torch.squeeze(pred_label)
        gt_label = torch.squeeze(gt_label)
        # get the mask element which >= 0, only 0 and 1 can effect the detection loss
        mask = torch.ge(gt_label, 0)
        valid_gt_label = torch.masked_select(gt_label, mask)
        valid_pred_label = torch.masked_select(pred_label, mask)
        return self.loss_cls(valid_pred_label, valid_gt_label) * self.cls_factor

    def box_loss(self, gt_label, gt_offset, pred_offset):
        pred_offset = torch.squeeze(pred_offset)
        gt_offset = torch.squeeze(gt_offset)
        gt_label = torch.squeeze(gt_label)

        # get the mask element which != 0
        unmask = torch.eq(gt_label, 0)
        mask = torch.eq(unmask, 0)
        # convert mask to dim index
        chose_index = torch.nonzero(mask.data)
        chose_index = torch.squeeze(chose_index)
        # only valid element can effect the loss
        valid_gt_offset = gt_offset[chose_index, :]
        valid_pred_offset = pred_offset[chose_index, :]
        return self.loss_box(valid_pred_offset, valid_gt_offset) * self.box_factor

    def landmark_loss(self, gt_label, gt_landmark, pred_landmark):
        pred_landmark = torch.squeeze(pred_landmark)
        gt_landmark = torch.squeeze(gt_landmark)
        gt_label = torch.squeeze(gt_label)
        mask = torch.eq(gt_label, -2)

        chose_index = torch.nonzero(mask.data)
        chose_index = torch.squeeze(chose_index)

        valid_gt_landmark = gt_landmark[chose_index, :]
        valid_pred_landmark = pred_landmark[chose_index, :]
        return self.loss_landmark(valid_pred_landmark, valid_gt_landmark) * self.land_factor
