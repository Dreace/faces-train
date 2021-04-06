import torch
import tqdm
from torchvision import transforms
from models.vgg import VGG

from datasets.fer_2013 import FER2013

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    train_data_set = FER2013('train', transform=transforms.Compose([
        transforms.RandomCrop(44),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]))
    model = VGG().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=128, shuffle=True, num_workers=1)
    for (images, labels) in tqdm.tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        predict_labels = model(images)
        loss = criterion(predict_labels, labels)
        loss.backward()
        optimizer.step()

        print(loss.cpu().data.numpy())


if __name__ == '__main__':
    main()
