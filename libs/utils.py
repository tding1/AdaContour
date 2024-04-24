import os
import pickle

import numpy as np
import random
import torch

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
