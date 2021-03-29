import torch
import cv2

from core.models import MTCNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
mtcnn = MTCNN(device=device)
mtcnn.eval()

p_state_dict = torch.load('./checkpoint/pnet.pt', map_location=torch.device('cpu'))
r_state_dict = torch.load('./checkpoint/rnet.pt', map_location=torch.device('cpu'))

mtcnn.load_state(p_state_dict, r_state_dict)

image = cv2.imread('test.jpg')
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

boxes = mtcnn.detect_p_net(image)
print(boxes)
boxes = mtcnn.detect_r_net(boxes)
print(boxes)

if __name__ == '__main__':
    pass
