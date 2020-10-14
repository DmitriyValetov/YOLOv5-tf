import cv2
import numpy as np

from nets import nn
from utils import util


def draw_bbox(image, boxes):
    for box in boxes:
        coordinate = np.array(box[:4], dtype=np.int32)
        c1, c2 = (coordinate[0], coordinate[1]), (coordinate[2], coordinate[3])
        cv2.rectangle(image, c1, c2, (255, 0, 0), 1)

    return image


def main():
    model = nn.build_model(training=False)
    model.load_weights("weights/model26.h5")

    image = cv2.imread('../Dataset/VOC2012/IMAGES/2007_000027.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_data, dw, dh, scale = util.resize(image)
    image_data = image_data.astype(np.float32) / 255.0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_data -= mean
    image_data /= std

    boxes, score, label = model.predict(image_data[np.newaxis, ...])

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / scale

    image = draw_bbox(image, boxes)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('result.png', image)


if __name__ == '__main__':
    main()
