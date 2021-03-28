import argparse
import os

import torch
from torch.autograd import Variable

import core.tools as tools
from core.image_db import ImageDB
from core.image_reader import TrainImageReader
from core.models import PNet
from utils.logger import logger


def train_p_net(model_path, end_epoch, imdb,
                batch_size, frequent=10, base_lr=0.01, use_cuda=True):
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    loss_fn = tools.LossFn()
    net = PNet()
    net.train()

    if use_cuda:
        net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)

    train_data = TrainImageReader(imdb, 12, batch_size, shuffle=True)

    for cur_epoch in range(1, end_epoch + 1):
        accuracy_for_display = 0.0
        class_loss_for_display = 0.0
        all_loss_for_display = 0.0
        box_offset_loss_for_display = 0.0

        train_data.reset()  # shuffle

        for batch_index, (image, (gt_label, gt_bbox, gt_landmark)) in enumerate(train_data):

            image_tensor = [tools.convert_image_to_tensor(image[i, :, :, :]) for i in range(image.shape[0])]
            image_tensor = torch.stack(image_tensor)

            image_tensor = Variable(image_tensor)
            gt_label = Variable(torch.from_numpy(gt_label).float())

            gt_bbox = Variable(torch.from_numpy(gt_bbox).float())
            # gt_landmark = Variable(torch.from_numpy(gt_landmark).float())

            if use_cuda:
                image_tensor = image_tensor.cuda()
                gt_label = gt_label.cuda()
                gt_bbox = gt_bbox.cuda()
                # gt_landmark = gt_landmark.cuda()
            predict_label, predict_box_offset = net(image_tensor)

            # all_loss, cls_loss, offset_loss = loss_fn.loss(gt_label=label_y,gt_offset=bbox_y, pred_label=predict_label,
            # pred_offset=predict_box_offset)
            class_loss = loss_fn.cls_loss(gt_label, predict_label)
            box_offset_loss = loss_fn.box_loss(gt_label, gt_bbox, predict_box_offset)
            # landmark_loss = loss_fn.landmark_loss(gt_label,gt_landmark,landmark_offset_pred)

            all_loss = class_loss * 1.0 + box_offset_loss * 0.5

            if batch_index % frequent == 0:
                accuracy = tools.compute_accuracy(predict_label, gt_label)
                accuracy_for_display = accuracy.data.cpu().numpy()

                class_loss_for_display = class_loss.data.cpu().numpy()
                box_offset_loss_for_display = box_offset_loss.data.cpu().numpy()
                all_loss_for_display = all_loss.data.cpu().numpy()
                logger.info(
                    "epoch: %d, step: %d, accuracy: %s, class loss: %s, box offset loss: %s, all_loss: %s, lr:%s " % (
                        cur_epoch, batch_index, accuracy_for_display, class_loss_for_display,
                        box_offset_loss_for_display, all_loss_for_display, base_lr))

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

        torch.save({
            'net': net.state_dict(),
            'loss': {
                'accuracy': accuracy_for_display,
                'class_loss': class_loss_for_display,
                'box_offset_loss': box_offset_loss_for_display,
                'all_loss': all_loss_for_display
            }
        }, f"{model_path}/p_net_epoch_{cur_epoch}.pt")
        logger.info(f'save to {model_path}/p_net_epoch_{cur_epoch}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train P-Net',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--annotation-file', help='training data annotation file', required=True)
    parser.add_argument('--model-path', help='training model store directory', required=True)
    parser.add_argument('--end-epoch', help='end epoch of training', default=10)
    parser.add_argument('--frequent', help='frequency of logging', default=10, type=int)
    parser.add_argument('--learning-rate', help='learning rate', default=0.01, type=float)
    parser.add_argument('--batch-size', help='train batch size', default=512, type=int)
    parser.add_argument('--use-cuda', help='train with gpu', default=True, type=bool)

    args = parser.parse_args()

    image_db = ImageDB(args.annotation_file)
    gt_imdb = image_db.load_imdb()
    gt_imdb = image_db.append_flipped_images(gt_imdb)
    train_p_net(model_path=args.model_path, end_epoch=args.end_epoch, imdb=gt_imdb,
                batch_size=args.batch_size,
                frequent=args.frequent, base_lr=args.learning_rate, use_cuda=args.use_cuda)
