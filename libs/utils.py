import os
import math
import pickle

import numpy as np
import random
import torch
import cv2

global global_seed


global_seed = 123
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)
torch.cuda.manual_seed_all(global_seed)
np.random.seed(global_seed)
random.seed(global_seed)


def _init_fn(worker_id):

    seed = global_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return


# convertor
def to_tensor(data):
    return torch.from_numpy(data).cuda()


def to_tensor2(data):
    return torch.from_numpy(data)


def to_np(data):
    return data.cpu().numpy()


def to_np2(data):
    return data.detach().cpu().numpy()


def to_3D_np(data):
    return np.repeat(np.expand_dims(data, 2), 3, 2)


def logger(text, LOGGER_FILE):  # write log
    with open(LOGGER_FILE, "a") as f:
        f.write(text)
        f.close()


# directory & file
def mkdir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)


def rmfile(path):
    if os.path.exists(path):
        os.remove(path)


# pickle
def save_pickle(dir_name, file_name, data):
    """
    :param file_path: ...
    :param data:
    :return:
    """
    os.makedirs(dir_name, exist_ok=True)
    with open(os.path.join(dir_name, file_name + ".pickle"), "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path):
    with open(file_path + ".pickle", "rb") as f:
        data = pickle.load(f)
    return data


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type="avg"):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        total = torch.FloatTensor([self.sum, self.count])
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is None:
            fmtstr = ""
        elif self.summary_type == "avg":
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type == "sum":
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type == "count":
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)
    

def draw_seg(result, mask):
    info = result['hierarchy_list']
    center_list = result['center_list']
    tmp = 0
    for i in info:
        _, ww = i.shape
        img1 = i[:,:ww//2]
        img2 = i[:,ww//2:]
        img1_boundary = _seg2bmap(img1)
        img2_boundary = _seg2bmap(img2)

        img1_dil = cv2.dilate(img1_boundary.astype(np.uint8),disk(2).astype(np.uint8))
        img2_dil = cv2.dilate(img2_boundary.astype(np.uint8),disk(2).astype(np.uint8))

        inter = (img1_dil + img2_dil) == 2
        tmp += inter

    tmp = tmp.astype(np.uint8)
    tmp = (np.stack((tmp,)*3, axis=-1)*255)
    tmp[...,0] = 0
    tmp[...,2] = 0

    line_mask = tmp > 0

    mask_ = (np.stack((mask,)*3, axis=-1)*255)
    mask_[line_mask] = 0

    for i, cen in enumerate(center_list):
        mask_ = cv2.circle(mask_, (int(cen[0]), int(cen[1])), 1, (0,0,255), 8)
    
    return mask_

def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries. The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.

    # Arguments
        seg: Segments labeled from 1..k.
        width:	Width of desired bmap  <= seg.shape[1]
        height:	Height of desired bmap <= seg.shape[0]

    # Returns
        bmap (ndarray):	Binary boundary map.

    David Martin <dmartin@eecs.berkeley.edu>
    January 2003
    """

    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width > w | height > h | abs(ar1 - ar2) >
                0.01), "Can't convert %dx%d seg to %dx%d bmap." % (w, h, width,
                                                                   height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


def disk(radius, dtype=np.uint8):
    """Generates a flat, disk-shaped footprint.

    A pixel is within the neighborhood if the Euclidean distance between
    it and the origin is no greater than radius.

    Parameters
    ----------
    radius : int
        The radius of the disk-shaped footprint.

    Other Parameters
    ----------------
    dtype : data-type
        The data type of the footprint.

    Returns
    -------
    footprint : ndarray
        The footprint where elements of the neighborhood are 1 and 0 otherwise.
    """
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    return np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
