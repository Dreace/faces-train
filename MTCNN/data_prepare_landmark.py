import argparse
import os

import cv2
import numpy as np

import utils.tools as tools
from utils.logger import logger


def prepare(annotation_file: str,
            processed_images_path: str,
            processed_annotation_path: str,
            wider_processed_annotation_path: str,
            image_path_prefix: str):
    if not os.path.exists(processed_images_path):
        os.makedirs(processed_images_path)

    landmark_annotation_file = open(processed_annotation_path + '/48_landmark.txt', 'w')

    with open(annotation_file, 'r') as file:
        annotations = file.readlines()

    total_images = len(annotations)
    logger.info(f"{total_images} images")

    valid_count = 0
    total_count = 0

    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        if len(annotation) != 15:
            continue

        im_path = image_path_prefix + '/' + annotation[0].replace('\\', '/')

        gt_box = list(map(float, annotation[1:5]))
        gt_box = np.array(gt_box, dtype=np.int32)

        landmark = list(map(float, annotation[5:]))
        landmark = np.array(landmark, dtype=np.float)

        img = cv2.imread(im_path)

        height, width, channel = img.shape

        x1, x2, y1, y2 = gt_box
        gt_box[1] = y1
        gt_box[2] = x2

        w = x2 - x1 + 1
        h = y2 - y1 + 1
        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue

        for i in range(10):
            bbox_size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
            delta_x = np.random.randint(-w * 0.2, w * 0.2)
            delta_y = np.random.randint(-h * 0.2, h * 0.2)
            nx1 = max(x1 + w / 2 - bbox_size / 2 + delta_x, 0)
            ny1 = max(y1 + h / 2 - bbox_size / 2 + delta_y, 0)

            nx2 = nx1 + bbox_size
            ny2 = ny1 + bbox_size
            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])
            cropped_im = img[int(ny1):int(ny2) + 1, int(nx1):int(nx2) + 1, :]
            resized_im = cv2.resize(cropped_im, (48, 48), interpolation=cv2.INTER_LINEAR)

            offset_x1 = (x1 - nx1) / float(bbox_size)
            offset_y1 = (y1 - ny1) / float(bbox_size)
            offset_x2 = (x2 - nx2) / float(bbox_size)
            offset_y2 = (y2 - ny2) / float(bbox_size)

            offset_left_eye_x = (landmark[0] - nx1) / float(bbox_size)
            offset_left_eye_y = (landmark[1] - ny1) / float(bbox_size)

            offset_right_eye_x = (landmark[2] - nx1) / float(bbox_size)
            offset_right_eye_y = (landmark[3] - ny1) / float(bbox_size)

            offset_nose_x = (landmark[4] - nx1) / float(bbox_size)
            offset_nose_y = (landmark[5] - ny1) / float(bbox_size)

            offset_left_mouth_x = (landmark[6] - nx1) / float(bbox_size)
            offset_left_mouth_y = (landmark[7] - ny1) / float(bbox_size)

            offset_right_mouth_x = (landmark[8] - nx1) / float(bbox_size)
            offset_right_mouth_y = (landmark[9] - ny1) / float(bbox_size)

            iou = tools.calculate_iou(crop_box.astype(np.float), np.expand_dims(gt_box.astype(np.float), 0))
            if iou > 0.65:
                image_file = f"{processed_images_path}/{valid_count}.jpg"
                cv2.imwrite(image_file, resized_im)

                landmark_annotation_file.write(
                    image_file + ' -2 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f \n' % \
                    (offset_x1, offset_y1, offset_x2, offset_y2,
                     offset_left_eye_x, offset_left_eye_y, offset_right_eye_x, offset_right_eye_y, offset_nose_x,
                     offset_nose_y, offset_left_mouth_x, offset_left_mouth_y, offset_right_mouth_x,
                     offset_right_mouth_y))

                valid_count += 1
        total_count += 1
        if total_count % 100 == 0:
            logger.info(f"{total_count / total_images * 100:.2f}%({total_count}/{total_images}) processed")

    landmark_annotation_file.close()
    tools.assemble_data(wider_processed_annotation_path + '/48_all.txt',
                        [processed_annotation_path + '/48_landmark.txt'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', help='raw annotation file', required=True)
    parser.add_argument('--processed-annotation-path', help='path to processed annotation file', required=True)
    parser.add_argument('--wider-processed-annotation-path', help='path to wider processed annotation file',
                        required=True)
    parser.add_argument('--processed-images-path', help='path to processed images', required=True)
    parser.add_argument('--image-path-prefix', help='image path prefix', default='')
    args = parser.parse_args()
    prepare(args.annotation_file,
            args.processed_images_path,
            args.processed_annotation_path,
            args.wider_processed_annotation_path,
            args.image_path_prefix)
