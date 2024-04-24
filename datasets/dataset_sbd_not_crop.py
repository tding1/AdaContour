from collections import namedtuple

import cv2
import scipy.io
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from libs.utils import *

Corner = namedtuple("Corner", "x1 y1 x2 y2")
BBox = Corner
Center = namedtuple("Center", "x y w h")


class SBD(Dataset):
    def __init__(self, cfg, datalist="train"):
        self.cfg = cfg
        self.height = cfg.height
        self.width = cfg.width

        # truth = load_pickle("data/{}_seg".format(datalist))
        # self.truth = truth

        truth_bb = load_pickle("data/{}_bb".format(datalist))
        self.truth_bb = truth_bb

        self.datalist = sorted(list(self.truth_bb.keys()))
        # self.datalist = load_pickle('datalist')

        # image transform
        self.transform = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)
        self.new_shape = (cfg.height, cfg.width)

        self.imw, self.imh = self.cfg.crop_w, self.cfg.crop_h
        self.color_grey = (114, 114, 114)

        self.exemplar_size = 127
        self.search_size = 511
        self.context_amount = 0.5

    def transform(self, img, interpolation):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        img = cv2.resize(img, (self.width, self.height), interpolation=interpolation)
        return img

    def get_image(self, idx):
        img = cv2.imread(self.cfg.img_dir + self.datalist[idx] + ".jpg")[
            :, :, [2, 1, 0]
        ]

        shape = img.shape[:2]
        self.org_width, self.org_height = int(shape[1]), int(shape[0])

        # padding
        rh, rw = self.new_shape[0] / self.org_height, self.new_shape[1] / self.org_width
        new_unpad = int(round(self.org_width * rw)), int(round(self.org_height * rh))
        # wh padding
        dw, dh = self.new_shape[1] - new_unpad[0], self.new_shape[0] - new_unpad[1]
        ratio = rh, rw
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if (self.org_width, self.org_height) != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color_grey
        )  # add borde

        self.ratio = ratio
        self.padding = (dw, dh)

        # transform
        img = self.transform(img)

        return img, ratio, (dw, dh)

    def rescale_pts_padding(self, datas):
        for data in datas:
            data[:, 0] = data[:, 0] * self.ratio[1] + self.padding[0]
            data[:, 1] = data[:, 1] * self.ratio[0] + self.padding[1]
            # data[:, 1] += self.crop_size

        # data[:, 0] = data[:, 0] / (self.org_width - 1) * (self.cfg.r_width - 1)
        # data[:, 1] = data[:, 1] / (self.org_height - 1) * (self.cfg.r_height - 1)

        return datas

    def get_label(self, idx, flip=0):
        # self.seg_label = self.truth[self.datalist[idx]]
        self.bb_label = self.truth_bb[self.datalist[idx]]

        poly_label = []
        cen_label = []
        id_label = []
        bbox_label = []

        # for i in range(len(self.seg_label)):
        #     id_label.append(int(self.seg_label[i][0]))
        #     poly_pts = np.array(self.seg_label[i][1:]).reshape(-1, 2)
        #     poly_label.append(poly_pts)

        for j in range(len(self.bb_label)):
            id_label.append(int(self.bb_label[j][-1]))
            bb_pts = np.array(self.bb_label[j][:-1]).reshape(-1, 2)
            bbox_label.append(bb_pts)

        # self.poly_label = self.rescale_pts_padding(poly_label)
        self.poly_label = None
        self.bbox_label = self.rescale_pts_padding(bbox_label)

        for bbox in self.bbox_label:
            cen = np.mean(bbox, axis=0).reshape(1, -1)
            cen_label.append(cen)

        self.cen_label = cen_label

        return self.poly_label, self.cen_label, self.bbox_label, id_label

    def make_polygon_mask(self, label):
        img = Image.new("L", (self.cfg.width, self.cfg.height))
        ImageDraw.Draw(img).polygon(
            np.round(label).astype(np.float32), fill=1, outline=True
        )
        mask = np.array(img).astype(np.uint8)

        return mask

    def mat2mask(self, mat_file):
        key = "GTinst"
        mat = scipy.io.loadmat(
            mat_file, mat_dtype=True, squeeze_me=True, struct_as_record=False
        )
        category = np.array(mat[key].Categories).reshape(-1).tolist()
        segmentation = mat[key].Segmentation
        masks = []
        for i in range(len(category)):
            mask = (segmentation == i + 1).astype(np.uint8)
            shape = mask.shape[:2]

            org_width, org_height = int(shape[1]), int(shape[0])

            # padding
            rh, rw = self.new_shape[0] / org_height, self.new_shape[1] / org_width
            new_unpad = int(round(org_width * rw)), int(round(org_height * rh))
            # wh padding
            dw, dh = self.new_shape[1] - new_unpad[0], self.new_shape[0] - new_unpad[1]
            dw /= 2  # divide padding into 2 sides
            dh /= 2

            if (org_width, org_height) != new_unpad:  # resize
                mask = cv2.resize(mask, new_unpad, interpolation=cv2.INTER_LINEAR)

            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            mask = cv2.copyMakeBorder(
                mask,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=self.color_grey,
            )  # add borde

            masks.append(mask)
        return masks

    def get_mask(self, idx):
        filename = self.datalist[idx]
        masks = self.mat2mask(self.cfg.img_dir[:-4] + "inst/" + filename + ".mat")
        return masks

    def get_cropped_data(self, img, masks, cen, bbox, id):
        outs = []

        for i in range(len(masks)):
            out = {}
            img_norm = self.normalize(img)
            mask = masks[i]

            out["cropped_img_rgb"] = img
            out["cropped_img"] = img_norm
            out["seg_mask"] = mask
            out["crop_h"], out["crop_w"] = self.cfg.height, self.cfg.width
            out["bbox_center"] = bbox[i].mean(axis=0).astype(np.float32)
            out["bbox"] = bbox[i].reshape(-1).astype(np.float32)
            out["id"] = id[i]
            outs.append(out)

        return outs

    def __getitem__(self, idx):
        img, _, _ = self.get_image(idx)
        _, cen, bbox, id = self.get_label(idx)
        masks = self.get_mask(idx)
        outputs = self.get_cropped_data(img, masks, cen, bbox, id)
        return {"output": outputs, "img_name": self.datalist[idx]}

    def __len__(self):
        return len(self.datalist)
