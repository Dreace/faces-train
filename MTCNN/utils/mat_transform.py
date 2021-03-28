import argparse

from wider_loader import WIDER


def transform(images_path: str, mat_file: str, output_file: str):
    wider = WIDER(mat_file, images_path)

    with open(output_file, 'w+') as f:
        for data in wider.next():
            line = [str(data.image_name)]
            for box in data.boxes:
                for bvalue in box:
                    line.append(str(bvalue))

            line_str = ' '.join(line)
            f.write(line_str)
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-path', required=True, help='path to WIDER dataset')
    parser.add_argument('--mat-file', required=True, help='mat file')
    parser.add_argument('--output-file', required=True, help='output file')
    args = parser.parse_args()
    transform(args.images_path, args.mat_file, args.output_file)
