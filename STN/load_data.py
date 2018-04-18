import numpy as np
import h5py
from PIL import Image
from sklearn.utils import shuffle


def read_dataset(mode):
    f = h5py.File('./dataset/{}/digitStruct.mat'.format(mode), 'r')

    name_list = f['digitStruct/name']
    bbox_list = f['digitStruct/bbox']

    img_list = []
    crop_list = []
    annotation_list = []
    label_list = []

    for name_obj in name_list:
        name_obj = f[name_obj[0]]
        str1 = ''.join(chr(i) for i in name_obj[:])
        img_list.append(str1)

    for bbox_obj in bbox_list:
        bbox_obj = f[bbox_obj[0]]
        bbox_len = len(bbox_obj['top'])
        bbox_list = []
        tmp_label = []
        crop_val = [9999, 9999, 0, 0]
        for i in range(bbox_len):
            # print(i)
            if isinstance(bbox_obj['top'][i][0], np.float64):
                bbox_top = bbox_obj['top'][i][0]
                bbox_height = bbox_obj['height'][i][0]
                bbox_left = bbox_obj['left'][i][0]
                bbox_width = bbox_obj['width'][i][0]
                bbox_label = bbox_obj['label'][i][0]
            else:
                bbox_top = f[bbox_obj['top'][i][0]][0, 0]
                bbox_height = f[bbox_obj['height'][i][0]][0, 0]
                bbox_left = f[bbox_obj['left'][i][0]][0, 0]
                bbox_width = f[bbox_obj['width'][i][0]][0, 0]
                bbox_label = f[bbox_obj['label'][i][0]][0, 0]
            # print(bbox_top)
            if bbox_top < crop_val[1]:
                crop_val[1] = bbox_top
            if bbox_left < crop_val[0]:
                crop_val[0] = bbox_left
            if crop_val[3] < bbox_top + bbox_height:
                crop_val[3] = bbox_top + bbox_height
            if crop_val[2] < bbox_left + bbox_width:
                crop_val[2] = bbox_left + bbox_width
            bbox_list.append([bbox_top, bbox_height, bbox_left, bbox_width])
            tmp_label.append(bbox_label)

        crop_list.append(crop_val)
        annotation_list.append(bbox_list)
        label_list.append(tmp_label)
    return img_list, annotation_list, label_list, crop_list


class Data():
    def __init__(self, mode):
        self.img, self.bbox, self.label, self.crop = read_dataset(mode)
        self.img, self.bbox, self.label, self.crop = shuffle(self.img, self.bbox, self.label, self.crop)
        self.index = 0
        self.dir = 'dataset/{}/'.format(mode)
        self.length = len(self.img)
        self.end = False

    def next(self, size):
        if self.index + size > self.length:
            size = self.length - self.index
            self.end = True
        img_list = self.img[self.index:self.index + size]
        crop_list = self.crop[self.index:self.index + size]
        label_list = self.label[self.index:self.index + size]

        slice_bbox = self.bbox[self.index:self.index + size]
        slice_img, slice_label = self.load(img_list, label_list, crop_list, slice_bbox)

        self.index += size

        return slice_img, slice_bbox, slice_label

    def load(self, imgs, labels, crops, bboxes):
        img_batch = []
        label_batch = []

        for filename, label, crop, bbox in zip(imgs, labels, crops, bboxes):
            if crop[2] - crop[0] > 64 or crop[3] - crop[1] > 64:
                continue
            center_x = (crop[0] + crop[2]) // 2
            if (crop[0] + crop[2]) % 2 == 1:
                center_x += 1
            center_y = (crop[1] + crop[3]) // 2
            if (crop[1] + crop[3]) % 2 == 1:
                center_y += 1
            sub = (center_x - 32, center_y - 32, center_x + 32, center_y + 32)

            size_l = center_x - 32
            size_r = center_x + 32
            val = [10 for _ in range(5)]
            chk_label = False
            if len(label) > 5:
                continue
            for idx, box in enumerate(bbox):
                val[idx] = label[idx]
            """
            for lab, box in zip(label, bbox):
                center_x = box[2] + box[3] / 2
                if center_x < size_l + 12.8:
                    if val[0] != 10:
                        chk_label = True
                    val[0] = lab
                elif center_x < size_l + 25.6:
                    if val[1] != 10:
                        chk_label = True
                    val[1] = lab
                elif center_x < size_l + 38.4:
                    if val[2] != 10:
                        chk_label = True
                    val[2] = lab
                elif center_x < size_l + 51.2:
                    if val[3] != 10:
                        chk_label = True
                    val[3] = lab
                else:
                    if val[4] != 10:
                        chk_label = True
                    val[4] = lab
            if chk_label:
                continue
            """

            label_batch.append(val)

            img = Image.open(self.dir + filename).convert('RGB')
            img = img.crop(sub)
            img = np.array(img, dtype=np.float32) / 255.0
            img_batch.append(img)

        return np.array(img_batch), np.array(label_batch)

    def shuffle(self):
        self.img, self.bbox, self.label = shuffle(self.img, self.bbox, self.label)
        self.index = 0
        self.end = False

if __name__ == '__main__':
    # Data('train')
    read_dataset('train')
