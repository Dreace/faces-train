import argparse
import os
import shutil

import cv2
import numpy as np

from logger import logger
from tools import calculate_iou, assemble_data


def prepare(annotation_file: str, processed_images_path: str, processed_annotation_path: str, negative_produce=50):
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
    positive_annotation_file = open(processed_annotation_path + '/12_positive.txt', 'w')
    negative_annotation_file = open(processed_annotation_path + '/12_negative.txt', 'w')
    part_annotation_file = open(processed_annotation_path + '/12_part.txt', 'w')

    # annotation_file: store labels of the wider face training data
    with open(annotation_file, 'r') as f:
        annotations = f.readlines()
    total = len(annotations)
    logger.info(f"{total} images")

    positive_count = 0  # positive
    negative_count = 0  # negative
    part_count = 0  # dont care
    total_count = 0
    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        image_file = annotation[0]
        # logger.info(f"processing {image_file}")
        bbox = list(map(float, annotation[1:]))
        boxes = np.array(bbox, dtype=np.int32).reshape(-1, 4)
        image = cv2.imread(image_file)
        total_count += 1
        if total_count % 100 == 0:
            logger.info(f"{total_count / total * 100:.2f}%({total_count}/{total}) processed")

        height, width, channel = image.shape

        # 每张图裁剪出 50 个负面样本
        negative_crop_count = 0
        while negative_crop_count < negative_produce:
            size = np.random.randint(12, min(width, height) / 2)
            nx = np.random.randint(0, width - size)
            ny = np.random.randint(0, height - size)
            crop_box = np.array([nx, ny, nx + size, ny + size])
            iou = calculate_iou(crop_box, boxes)
            if np.max(iou) < 0.3:
                # iou with all gts must below 0.3
                cropped_image = image[ny: ny + size, nx: nx + size, :]
                resized_image = cv2.resize(cropped_image, (12, 12), interpolation=cv2.INTER_LINEAR)
                image_file = f"{negative_images_path}/{negative_count}.jpg"
                negative_annotation_file.write(image_file + ' 0\n')
                cv2.imwrite(image_file, resized_image)
                negative_count += 1
                negative_crop_count += 1

        for box in boxes:
            # box (x_left, y_top, x_right, y_bottom)
            x1, y1, x2, y2 = box
            # w = x2 - x1 + 1
            # h = y2 - y1 + 1
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            # ignore small faces
            # in case the ground truth boxes of small faces are not accurate
            if min(w, h) <= 0 or max(w, h) < 40 or x1 < 0 or y1 < 0:
                continue

            # generate negative examples that have overlap with gt
            for i in range(5):
                size = np.random.randint(12, min(width, height) / 2)
                # delta_x and delta_y are offsets of (x1, y1)

                delta_x = np.random.randint(max(-size, -x1), w)
                delta_y = np.random.randint(max(-size, -y1), h)
                nx1 = max(0, x1 + delta_x)
                ny1 = max(0, y1 + delta_y)
                if nx1 + size > width or ny1 + size > height:
                    continue

                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                iou = calculate_iou(crop_box, boxes)

                if np.max(iou) < 0.3:
                    # iou with all gts must below 0.3
                    cropped_image = image[ny1: ny1 + size, nx1: nx1 + size, :]
                    resized_image = cv2.resize(cropped_image, (12, 12), interpolation=cv2.INTER_LINEAR)
                    image_file = f"{negative_images_path}/{negative_count}.jpg"
                    negative_annotation_file.write(image_file + ' 0\n')
                    cv2.imwrite(image_file, resized_image)
                    negative_count += 1

            # generate positive examples and part faces
            for i in range(20):
                size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

                # delta here is the offset of box center
                delta_x = np.random.randint(-w * 0.2, w * 0.2)
                delta_y = np.random.randint(-h * 0.2, h * 0.2)

                nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
                ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
                nx2 = nx1 + size
                ny2 = ny1 + size

                if nx2 > width or ny2 > height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])

                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)

                cropped_image = image[int(ny1): int(ny2), int(nx1): int(nx2), :]
                resized_image = cv2.resize(cropped_image, (12, 12), interpolation=cv2.INTER_LINEAR)

                box = box.reshape(1, -1)
                if calculate_iou(crop_box, box) >= 0.65:
                    image_file = f"{positive_images_path}/{positive_count}.jpg"
                    positive_annotation_file.write(
                        image_file + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(image_file, resized_image)
                    positive_count += 1
                elif calculate_iou(crop_box, box) >= 0.4:
                    image_file = f"{part_images_path}/{part_count}.jpg"
                    part_annotation_file.write(
                        image_file + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(image_file, resized_image)
                    part_count += 1

    positive_annotation_file.close()
    negative_annotation_file.close()
    part_annotation_file.close()
    assemble_data(processed_annotation_path + '/12_all.txt',
                  [processed_annotation_path + '/12_positive.txt', processed_annotation_path + '/12_part.txt',
                   processed_annotation_path + '/12_negative.txt'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', help='raw annotation file', required=True)
    parser.add_argument('--processed-annotation-path', help='path to processed annotation file', required=True)
    parser.add_argument('--processed-images-path', help='path to processed images', required=True)
    parser.add_argument('--negative-produce', help='how many negative image need to produce', default=50, type=int)
    args = parser.parse_args()
    prepare(args.annotation_file, args.processed_images_path, args.processed_annotation_path, args.negative_produce)
