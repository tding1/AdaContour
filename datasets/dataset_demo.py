import cv2
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
from collections import namedtuple
from libs.utils import *

Corner = namedtuple('Corner', 'x1 y1 x2 y2')
BBox = Corner
Center = namedtuple('Center', 'x y w h')


class Dataset_demo(Dataset):
    def __init__(self, cfg, datalist='train'):
        self.cfg = cfg
        self.height = cfg.height
        self.width = cfg.width

        self.datalist = ['image%d.png' % i for i in range(1, 11)]
        self.masklist = ['image%d-mask.png' % i for i in range(1, 11)]

        # image transform
        self.transform = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)
        self.new_shape = (cfg.height, cfg.width)

        self.imw, self.imh = self.cfg.crop_w, self.cfg.crop_h

    def transform(self, img, interpolation):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        img = cv2.resize(img, (self.width, self.height),
                         interpolation=interpolation)
        return img

    def get_image(self, idx):
        img = cv2.imread(self.cfg.img_dir + self.datalist[idx])[..., ::-1].transpose(2, 0, 1)
        mask = cv2.imread(self.cfg.img_dir + self.masklist[idx]).transpose(2, 0, 1)[0].astype(bool).astype(np.uint8)

        self.h, self.w, _ = img.shape

        img = torch.from_numpy(img.copy())
        mask = torch.from_numpy(mask)
        return img, mask

    def make_polygon_mask(self, label):
        img = Image.new("L", (self.w, self.h))
        ImageDraw.Draw(img).polygon(
            np.round(label).astype(np.float32), fill=1, outline=True)
        mask = np.array(img).astype(np.uint8)

        return mask

    def __getitem__(self, idx):
        img, mask = self.get_image(idx)
        outs = []
        label = {}
        label["cropped_img"] = self.normalize(img.float()/255.0)
        label["seg_mask"] = mask

        # placeholder for id and bbox since they are not used here
        label['id'] = 0
        label['bbox'] = torch.tensor([[0,0,1,1]], dtype=torch.float)

        outs.append(label)

        return {'output': outs,
                'img_name': self.datalist[idx]}

    def __len__(self):
        return len(self.datalist)
