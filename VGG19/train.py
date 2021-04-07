import argparse

import torch
import tqdm
from torchvision import transforms

from datasets.fer_2013 import FER2013
from models.vgg import VGG

parser = argparse.ArgumentParser()
parser.add_argument('--save-prefix', type=str, required=True,
                    help="(REQUIRED) where to save durable data"
                    )
parser.add_argument('--batch-size', type=int, default=128, required=True)
parser.add_argument('--epochs', type=int, default=50, required=True)
parser.add_argument('--learning-rate', help='learning rate', default=0.01, type=float)
parser.add_argument('--learning-rate-step', help='learning rate step', default=1, type=int)
parser.add_argument('--learning-rate-gamma', help='learning rate gamma', default=0.8, type=float)
parser.add_argument('--resume', help='resume train')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def main():
    epochs = args.epochs
    save_prefix = args.save_prefix
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    learning_rate_step = args.learning_rate_step
    learning_rate_gamma = args.learning_rate_gamma
    resume = args.resume

    start_epoch = 1
    model = VGG()
    if resume:
        state_dict = torch.load(resume, map_location='cpu')
        model.load_state_dict(state_dict['net'])
        start_epoch = state_dict['epoch']
        learning_rate = state_dict['lr']
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_step, gamma=learning_rate_gamma)

    train_loader = torch.utils.data.DataLoader(FER2013('train', transform=transforms.Compose([
        transforms.RandomCrop(44),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])), batch_size=batch_size, shuffle=True, num_workers=1)

    public_test_loader = torch.utils.data.DataLoader(FER2013('public_test', transform=transforms.Compose([
        transforms.CenterCrop(44),
        transforms.ToTensor()
    ])), batch_size=batch_size, shuffle=False, num_workers=1)

    private_test_loader = torch.utils.data.DataLoader(FER2013('private_test', transform=transforms.Compose([
        transforms.CenterCrop(44),
        transforms.ToTensor()
    ])), batch_size=batch_size, shuffle=False, num_workers=1)

    for epoch in range(start_epoch, epochs + 1):
        print(f'epoch {epoch}')
        # 训练
        total = 0
        correct = 0
        for batch_index, (images, labels) in enumerate(tqdm.tqdm(train_loader)):
            images = images.to(device)
            labels = labels.to(device)

            model.train()
            optimizer.zero_grad()

            predict_labels = model(images)

            loss = criterion(predict_labels, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(predict_labels, 1)
            total += labels.shape[0]
            correct += (predicted == labels).data.cpu().sum()
            if batch_index % 10 == 0:
                print(
                    f'train, loss {loss.cpu().data.numpy():.3}  accuracy {correct / total * 100:0.3}%  lr {get_learning_rate(optimizer)[0]}')
                with open(f'{save_prefix}/train.txt', 'a+') as train_log_file:
                    train_log_file.write(
                        f'{epoch}-{batch_index} {loss.cpu().data.numpy():.3} {correct / total * 100:0.3} {get_learning_rate(optimizer)[0]}\n')
        scheduler.step()
        torch.save({
            'epoch': epoch,
            'lr': learning_rate,
            'net': model.state_dict(),
        }, f'{save_prefix}/epoch_{epoch}.pt')

        model.eval()
        with torch.no_grad():
            # public test
            public_test_total = 0
            public_test_correct = 0
            for batch_index, (images, labels) in enumerate(tqdm.tqdm(public_test_loader)):
                images = images.to(device)
                labels = labels.to(device)

                predict_labels = model(images)

                loss = criterion(predict_labels, labels)

                _, predicted = torch.max(predict_labels, 1)
                public_test_total += labels.shape[0]
                public_test_correct += (predicted == labels).data.cpu().sum()
            print(
                f'public test, loss {loss.cpu().data.numpy():.3}  accuracy {public_test_correct / public_test_total * 100:0.3}%')
            with open(f'{save_prefix}/public_test.txt', 'a+') as public_test_log_file:
                public_test_log_file.write(
                    f'{loss.cpu().data.numpy():.3} {public_test_correct / public_test_total * 100:0.3}\n')
            # private test
            private_test_total = 0
            private_test_correct = 0
            for _, (images, labels) in enumerate(tqdm.tqdm(private_test_loader)):
                images = images.to(device)
                labels = labels.to(device)

                predict_labels = model(images)

                loss = criterion(predict_labels, labels)

                _, predicted = torch.max(predict_labels, 1)
                private_test_total += labels.shape[0]
                private_test_correct += (predicted == labels).data.cpu().sum()
            print(
                f'public test, loss {loss.cpu().data.numpy():.3}  accuracy {private_test_correct / private_test_total * 100:0.3}%')
            with open(f'{save_prefix}/private_test.txt', 'a+') as private_test_log_file:
                private_test_log_file.write(
                    f'{loss.cpu().data.numpy():.3} {private_test_correct / private_test_total * 100:0.3}\n')


if __name__ == '__main__':
    main()
