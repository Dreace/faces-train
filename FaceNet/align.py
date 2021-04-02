import glob
import os

import cv2
import torch
from tqdm import tqdm

from MTCNN.core.models import MTCNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(device=device)
mtcnn.eval()

p_state_dict = torch.load('MTCNN/checkpoint/pnet.pt', map_location=torch.device('cpu'))
r_state_dict = torch.load('MTCNN/checkpoint/rnet.pt', map_location=torch.device('cpu'))
o_state_dict = torch.load('MTCNN/checkpoint/onet.pt', map_location=torch.device('cpu'))

mtcnn.load_state(p_state_dict, r_state_dict, o_state_dict)

processed_path = 'FaceNet/data/VGGFace2/processed'
if not os.path.exists(processed_path):
    os.mkdir(processed_path)

files = glob.glob("FaceNet/data/VGGFace2/test/*/*")
progress_bar = tqdm(files)

for file_name in progress_bar:
    image = cv2.imread(file_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes, _, _ = mtcnn.detect_face(image_rgb)
    if (boxes is None) or boxes.shape[0] > 1:
        continue
    box = boxes[0]
    width, height = box[2] - box[0], box[3] - box[1]
    max_length = int(max(width, height))
    x = int(box[0] - (max_length - width) / 2)
    y = int(box[1] - (max_length - height) / 2)
    x = max(0, x)
    y = max(0, y)
    x1 = min(x + max_length, image.shape[1])
    y1 = min(y + max_length, image.shape[0])
    # image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0))
    # image = cv2.rectangle(image, (x, y), (x + max_length, y + max_length), (0, 255, 0))
    image = image[y:y1, x:x1]
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    face_id = os.path.basename(file_name).split('.')[0]
    face_label = os.path.basename(os.path.dirname(file_name))
    if not os.path.exists(f'{processed_path}/{face_label}'):
        os.mkdir(f'{processed_path}/{face_label}')
    cv2.imwrite(f'{processed_path}/{face_label}/{face_id}.jpg', image)

if __name__ == '__main__':
    pass
