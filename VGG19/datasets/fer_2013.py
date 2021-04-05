import numpy as np
import torch.utils.data as data
from PIL import Image


class FER2013(data.Dataset):
    def __init__(self, data_type='train', transform=None):
        self.transform = transform
        fer_data = []
        fer_label = []
        with open(f'./data/{data_type}.csv', 'r') as data_file:
            for line in data_file.readlines():
                line = line.strip().split(',')
                image = np.fromstring(line[1], sep=' ', dtype=np.uint8).reshape(48, 48)
                image = image[:, :, np.newaxis]
                image = np.concatenate((image, image, image), axis=2)
                fer_data.append(Image.fromarray(image))
                # fer_data.append(list([lambda x: int(x), line[1].split(' ')]))
                fer_label.append(int(line[0]))
        self.data = fer_data
        self.label = fer_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        if self.transform:
            image = self.transform(image)
        return image, self.label[index]
