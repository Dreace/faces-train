import onnx
import torch
from torch.autograd import Variable
from core.models import PNet, ONet, RNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net_state = torch.load('./checkpoint/rnet.pt')

net = RNet()
net.load_state_dict(net_state)

net = net.eval().to(device)

dummy_input = Variable(torch.randn(1, 3, 24, 24))

input_names = ["input_1"]
output_names = ["dense5_2", "softmax5_1"]

torch.onnx.export(net, dummy_input, 'r_net.onnx', verbose=True,
                  input_names=input_names, output_names=output_names)
if __name__ == '__main__':
    pass
