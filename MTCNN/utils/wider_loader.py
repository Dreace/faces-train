from typing import List, Tuple

from scipy.io import loadmat


class WIDERData:
    def __init__(self, image_name: str, boxes: List[Tuple[int]]):
        self.image_name = image_name
        self.boxes = boxes


class WIDER(object):
    def __init__(self, file_to_label, path_to_image=None):
        self.file_to_label = file_to_label
        self.path_to_image = path_to_image

        self.f = loadmat(file_to_label)
        self.event_list = self.f['event_list']
        self.file_list = self.f['file_list']
        self.face_bbx_list = self.f['face_bbx_list']

    def next(self):
        for event_idx, event in enumerate(self.event_list):
            # fix error of "can't not .. bytes and strings"
            e = str(event[0][0].encode('utf-8'))[2:-1]
            for file, bbx in zip(self.file_list[event_idx][0],
                                 self.face_bbx_list[event_idx][0]):
                f = file[0][0].encode('utf-8')
                # print(type(e), type(f))  # bytes, bytes
                # fix error of "can't not .. bytes and strings"
                f = str(f)[2:-1]
                # path_of_image = os.path.join(self.path_to_image, str(e), str(f)) + ".jpg"
                path_of_image = self.path_to_image + '/' + e + '/' + f + ".jpg"
                # print(path_of_image)

                boxes = []
                bbx0 = bbx[0]
                for i in range(bbx0.shape[0]):
                    x, y, width, height = bbx0[i]
                    boxes.append((int(x), int(y), int(x + width), int(y + height)))
                yield WIDERData(path_of_image, boxes)
