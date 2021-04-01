import cv2
import torch

from core.models import MTCNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
mtcnn = MTCNN(device=device)
mtcnn.eval()

p_state_dict = torch.load('./checkpoint/pnet.pt', map_location=torch.device('cpu'))
r_state_dict = torch.load('./checkpoint/rnet.pt', map_location=torch.device('cpu'))
o_state_dict = torch.load('./checkpoint/o_net_epoch_50.pt', map_location=torch.device('cpu'))

mtcnn.load_state(p_state_dict, r_state_dict, o_state_dict['net'])

image = cv2.imread('test.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(mtcnn.detect_face(image))
exit(-1)
video = cv2.VideoCapture()
video.open(0)
while True:
    code, frame = video.read()

    boxes, _, points = mtcnn.detect_face(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if boxes is not None:
        for box, point in zip(boxes, points):
            frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0))
            for p in point:
                frame = cv2.circle(frame, (p[0], p[1]), 1, (255, 0, 0))
    cv2.imshow("", frame)
    cv2.waitKey(10)

if __name__ == '__main__':
    pass
