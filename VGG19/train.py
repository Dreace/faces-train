import torch
import tqdm
from torchvision import transforms

from datasets.fer_2013 import FER2013


def main():
    train_data_set = FER2013('train', transform=transforms.Compose([
        transforms.RandomCrop(44),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]))
    train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=2, shuffle=True, num_workers=1)
    for (images, labels) in tqdm.tqdm(train_loader):
        print(images)
        print(labels)
        break


if __name__ == '__main__':
    main()
