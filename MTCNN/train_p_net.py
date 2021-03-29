import argparse
import os

import numpy as np
import torch
from torch.autograd import Variable

import core.tools as tools
from core.image_db import ImageDB
from core.image_reader import TrainImageReader
from core.models import PNet
from utils.logger import logger


def train_p_net(model_path, end_epoch, image_db, image_db_validate,
                batch_size, frequent=10, base_lr=0.01, use_cuda=True, resume: str = None):
    device = torch.device('cuda:0' if use_cuda and torch.cuda.is_available() else 'cpu')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    loss_fn = tools.LossFn()
    net = PNet()

    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)

    train_data = TrainImageReader(image_db, 12, batch_size, shuffle=True)
    validate_data = TrainImageReader(image_db_validate, 12, batch_size, shuffle=True)

    start_epoch = 1
    # 恢复训练
    if resume:
        state_dict = torch.load(resume, map_location='cpu')
        start_epoch = state_dict['epoch']
        net.state_dict(state_dict['net'])

    net.to(device)

    for cur_epoch in range(start_epoch, end_epoch + 1):
        net.train()
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

            image_tensor = image_tensor.to(device)
            gt_label = gt_label.to(device)
            gt_bbox = gt_bbox.to(device)
            # gt_landmark = gt_landmark.cuda()
            predict_box_offset, predict_label = net(image_tensor)

            # all_loss, cls_loss, offset_loss = loss_fn.loss(gt_label=label_y,gt_offset=bbox_y, pred_label=predict_label,
            # pred_offset=predict_box_offset)
            class_loss = loss_fn.class_loss(gt_label, predict_label[:, 1])
            box_offset_loss = loss_fn.box_loss(gt_label, gt_bbox, predict_box_offset)
            # landmark_loss = loss_fn.landmark_loss(gt_label,gt_landmark,landmark_offset_pred)

            all_loss = class_loss * 1.0 + box_offset_loss * 0.5

            if batch_index % frequent == 0:
                accuracy = tools.compute_accuracy(predict_label[:, 1], gt_label)
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

        # 计算验证数据集损失
        net.eval()
        with torch.no_grad():
            class_losses = []
            box_offset_losses = []
            accuracies = []
            validate_data.reset()
            for batch_index, (image, (gt_label, gt_bbox, gt_landmark)) in enumerate(validate_data):

                image_tensor = [tools.convert_image_to_tensor(image[i, :, :, :]) for i in range(image.shape[0])]
                image_tensor = torch.stack(image_tensor)

                image_tensor = Variable(image_tensor)
                gt_label = Variable(torch.from_numpy(gt_label).float())

                gt_bbox = Variable(torch.from_numpy(gt_bbox).float())

                image_tensor = image_tensor.to(device)
                gt_label = gt_label.to(device)
                gt_bbox = gt_bbox.to(device)

                predict_box_offset, predict_label = net(image_tensor)

                class_loss = loss_fn.class_loss(gt_label, predict_label[:, 1])
                box_offset_loss = loss_fn.box_loss(gt_label, gt_bbox, predict_box_offset)
                accuracy = tools.compute_accuracy(predict_label[:, 1], gt_label)

                class_losses.append(class_loss.data.cpu().numpy())
                box_offset_losses.append(box_offset_loss.data.cpu().numpy())
                accuracies.append(accuracy.data.cpu().numpy())
        logger.info(
            "validate, accuracy: %s, class loss: %s, box offset loss: %s, all_loss: %s" % (
                np.mean(accuracies), np.mean(class_losses),
                np.mean(box_offset_losses), np.mean(class_losses) * 0.1 +
                np.mean(box_offset_losses) * 0.5))
        torch.save({
            'epoch': cur_epoch,
            'net': net.state_dict(),
            'train_loss': {
                'accuracy': accuracy_for_display,
                'class_loss': class_loss_for_display,
                'box_offset_loss': box_offset_loss_for_display,
                'all_loss': all_loss_for_display
            },
            'validate_loss': {
                'accuracy': np.mean(accuracies),
                'class_loss': np.mean(class_losses),
                'box_offset_loss': np.mean(box_offset_losses),
                'all_loss': np.mean(class_losses) * 0.1 + np.mean(box_offset_losses) * 0.5
            }
        }, f"{model_path}/p_net_epoch_{cur_epoch}.pt")
        logger.info(f'save to {model_path}/p_net_epoch_{cur_epoch}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train P-Net',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--annotation-file', help='training data annotation file', required=True)
    parser.add_argument('--validate-annotation-file', help='validate data annotation file', required=True)
    parser.add_argument('--model-path', help='training model store directory', required=True)
    parser.add_argument('--end-epoch', help='end epoch of training', default=10, type=int)
    parser.add_argument('--frequent', help='frequency of logging', default=10, type=int)
    parser.add_argument('--learning-rate', help='learning rate', default=0.01, type=float)
    parser.add_argument('--batch-size', help='train batch size', default=512, type=int)
    parser.add_argument('--use-cuda', help='train with gpu', default=True, type=bool)
    # parser.add_argument('--start-epoch', help='start epoch', default=1, type=int)
    parser.add_argument('--resume', help='resume train')

    args = parser.parse_args()

    image_db = ImageDB(args.annotation_file)
    gt_image_db = image_db.load_imdb()
    gt_image_db = image_db.append_flipped_images(gt_image_db)

    image_db_validate = ImageDB(args.validate_annotation_file)
    gt_image_db_validate = image_db_validate.load_imdb()

    use_cuda = args.use_cuda and torch.cuda.is_available()
    train_p_net(model_path=args.model_path, end_epoch=args.end_epoch, image_db=gt_image_db,
                image_db_validate=gt_image_db_validate,
                batch_size=args.batch_size,
                frequent=args.frequent,
                base_lr=args.learning_rate,
                use_cuda=use_cuda,
                resume=args.resume)
