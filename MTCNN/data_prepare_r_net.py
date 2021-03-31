import argparse
import os
import shutil

import cv2
import numpy as np
import torch

from core.image_db import ImageDB
from core.image_reader import TestImageLoader
from core.models import MTCNN
from utils.logger import logger
import utils.tools as tools


def prepare(annotation_file: str, processed_images_path: str, processed_annotation_path: str, p_state_file: str,
            negative_produce=50,
            use_cuda: bool = True):
    device = torch.device('cuda:0' if use_cuda and torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    mtcnn = MTCNN(device=device)
    mtcnn.eval()
    state_dict = torch.load(p_state_file, map_location=torch.device('cpu'))
    # mtcnn.load_state(state_dict['net'])
    mtcnn.load_state(state_dict)
    image_db = ImageDB(annotation_file, mode="validate")
    imdb = image_db.load_imdb()
    image_reader = TestImageLoader(imdb, 1, False)

    predict_boxes = []

    for image in image_reader:
        boxes_align = mtcnn.detect_p_net(image).cpu().numpy()
        if boxes_align is None:
            predict_boxes.append(np.array([]))
            continue
        predict_boxes.append(boxes_align)

    positive_images_path = processed_images_path + "/positive"
    negative_images_path = processed_images_path + '/negative'
    part_images_path = processed_images_path + "/part"

    if os.path.exists(positive_images_path):
        shutil.rmtree(positive_images_path)
    os.mkdir(positive_images_path)

    if os.path.exists(part_images_path):
        shutil.rmtree(part_images_path)
    os.mkdir(part_images_path)

    if os.path.exists(negative_images_path):
        shutil.rmtree(negative_images_path)
    os.mkdir(negative_images_path)

    # store labels of positive, negative, part images
    positive_annotation_file = open(processed_annotation_path + '/24_positive.txt', 'w')
    negative_annotation_file = open(processed_annotation_path + '/24_negative.txt', 'w')
    part_annotation_file = open(processed_annotation_path + '/24_part.txt', 'w')

    image_paths = []
    gt_boxes = []

    with open(annotation_file, 'r') as f:
        annotations = f.readlines()
    total = len(annotations)
    logger.info(f"{total} images")

    for annotation in annotations:
        annotation = annotation.strip().split(' ')

        boxes = list(map(float, annotation[1:]))
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        image_paths.append(annotation[0])
        gt_boxes.append(boxes)

    positive_count = 0  # positive
    negative_count = 0  # negative
    part_count = 0  # dont care
    total_count = 0
    for image_path, boxes_p, boxes_gt in zip(image_paths, predict_boxes, gt_boxes):
        boxes_gt = np.array(boxes_gt, dtype=np.float32).reshape(-1, 4)

        if boxes_p.shape[0] == 0:
            continue

        img = cv2.imread(image_path)

        boxes_p[:, 0:4] = np.round(boxes_p[:, 0:4])
        neg_num = 0
        for box in boxes_p:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # ignore box that is too small or beyond image border
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            # compute intersection over union(IoU) between current box and all gt boxes
            iou = tools.calculate_iou(box, boxes_gt)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_image = cv2.resize(cropped_im, (24, 24),
                                       interpolation=cv2.INTER_LINEAR)

            # save negative images and write label
            # iou with all boxes_gt must below 0.3
            if np.max(iou) < 0.3 and neg_num < 60:
                # save the examples
                image_file = f"{negative_images_path}/{negative_count}.jpg"
                negative_annotation_file.write(image_file + ' 0\n')
                cv2.imwrite(image_file, resized_image)
                negative_count += 1
                neg_num += 1
            else:
                # find gt_box with the highest iou
                idx = np.argmax(iou)
                assigned_gt = boxes_gt[idx]
                x1, y1, x2, y2 = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # save positive and part-face images and write labels
                if np.max(iou) >= 0.65:
                    image_file = f"{positive_images_path}/{positive_count}.jpg"
                    positive_annotation_file.write(
                        image_file + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(image_file, resized_image)
                    positive_count += 1
                elif np.max(iou) >= 0.4:
                    image_file = f"{part_images_path}/{part_count}.jpg"
                    part_annotation_file.write(
                        image_file + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(image_file, resized_image)
                    part_count += 1

        total_count += 1
        if total_count % 100 == 0:
            logger.info(f"{total_count / total * 100:.2f}%({total_count}/{total}) processed")

    positive_annotation_file.close()
    negative_annotation_file.close()
    part_annotation_file.close()
    tools.assemble_data(processed_annotation_path + '/24_all.txt',
                        [processed_annotation_path + '/24_positive.txt', processed_annotation_path + '/24_part.txt',
                         processed_annotation_path + '/24_negative.txt'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', help='raw annotation file', required=True)
    parser.add_argument('--processed-annotation-path', help='path to processed annotation file', required=True)
    parser.add_argument('--processed-images-path', help='path to processed images', required=True)
    parser.add_argument('--p-state-file', help='state file', required=True)
    parser.add_argument('--negative-produce', help='how many negative image need to produce', default=50, type=int)
    parser.add_argument('--use-cuda', help='use cuda', default=True, type=bool)
    args = parser.parse_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()
    prepare(args.annotation_file, args.processed_images_path, args.processed_annotation_path, args.p_state_file,
            args.negative_produce,
            use_cuda=use_cuda)
