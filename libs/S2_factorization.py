import os

import h5py
import torch
import numpy as np
import torch.linalg as LA

from libs.utils import *
from RSR_solvers.FMS.FMS import *


class Factorization(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg
        self.dataloader = dict_DB["dataloader"]
        self.visualize = dict_DB["visualize"]
        self.height = cfg.height
        self.width = cfg.width
        self.size = np.float32([cfg.height, cfg.width])

        self.center = np.array([(self.width - 1) / 2, (self.height - 1) / 2])
        self.datalist = []

    def load_contour_component(self):
        self.datalist = h5py.File(os.path.join(self.cfg.output_dir, "S1.h5"), "r")

        cnt = 0
        for i, (name, metas) in enumerate(self.datalist.items()):
            for j in range(len(metas)):
                meta = metas["meta" + str(j)]
                r = meta["r"][()]
                cnt += r.shape[0]

        print(cnt)

        # TD: pre-allocate space to speed up
        self.mat = np.empty((self.cfg.node_num, cnt))

        n = 0
        for i, (name, metas) in enumerate(self.datalist.items()):
            for j in range(len(metas)):
                meta = metas["meta" + str(j)]
                r = meta["r"][()]
                self.mat[:, n : n + r.shape[0]] = r.T
                n += r.shape[0]

            if i % 1000 == 0:
                print("%d done!" % i)

        if self.cfg.save:
            save_pickle(dir_name=self.cfg.output_dir, file_name="matrix", data=self.mat)

    def do_SVD(self):
        print("start SVD")
        self.mat = torch.from_numpy(load_pickle(self.cfg.output_dir + "matrix"))

        U, S, _ = LA.svd(self.mat / self.cfg.max_dist, full_matrices=False)

        if self.cfg.save:
            save_pickle(dir_name=self.cfg.output_dir, file_name="U_svd", data=U)
            save_pickle(dir_name=self.cfg.output_dir, file_name="S_svd", data=S)

    def do_FMS(self, d):
        print("start FMS")
        self.mat = torch.from_numpy(load_pickle(self.cfg.output_dir + "matrix"))

        fms = FMS(verbose=True)
        U_fms = fms.run(self.mat.t() / self.cfg.max_dist, d)

        if self.cfg.save:
            save_pickle(
                dir_name=self.cfg.output_dir, file_name="U_fms_" + str(d), data=U_fms
            )

    def run(self):
        print("start")
        self.load_contour_component()
        if self.cfg.fms:
            self.do_FMS(self.cfg.dim)
        else:
            self.do_SVD()
