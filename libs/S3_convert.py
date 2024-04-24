import math

import cv2
import h5py
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
from davisinteractive.metrics import batched_f_measure

from libs.utils import *
from libs.hierarchy_encoding import comp_iou, grow_mask


class Convert(object):
    def __init__(self, cfg, dict_DB):

        self.cfg = cfg
        self.dataloader = dict_DB["dataloader"]
        self.visualize = dict_DB["visualize"]
        self.height = cfg.height
        self.width = cfg.width
        self.size = np.float32([cfg.height, cfg.width])

        self.center = np.array([(self.width - 1) / 2, (self.height - 1) / 2])
        self.datalist = []

        self.r_coord = np.linspace(0, 360, self.cfg.node_num, endpoint=False)
        self.r_coord_x = np.cos(self.r_coord * 2 * math.pi / 360).reshape(-1, 1)
        self.r_coord_y = ((-1) * np.sin(self.r_coord * 2 * math.pi / 360)).reshape(
            -1, 1
        )
        self.r_coord_xy = np.concatenate(
            (self.r_coord_x, self.r_coord_y), axis=1
        ).astype(np.float32)

    def approximate_contour(self):
        out = self.data
        self.img = self.label_all[0]["cropped_img"]
        self.c, self.h, self.w = self.img[0].shape

        dir_ = os.path.join(self.cfg.output_dir, "display_approx", self.img_name)

        for i in range(len(out.keys())):
            meta = out["meta" + str(i)]
            contour_center = meta["contour_center"][()]
            r = meta["r"][()]
            seg_gt = to_np(self.label_all[i]["seg_mask"][0])

            U = self.U[:, : self.cfg.dim]
            U_t = (U.T).copy()

            c = r @ U / self.cfg.max_dist
            r_ap = c @ U_t * self.cfg.max_dist

            contour_center = (
                contour_center[:, np.newaxis, :]
                .repeat(self.cfg.node_num, axis=1)
                .astype(np.float32)
            )

            idx = np.linspace(0, 359, self.cfg.node_num).astype(np.int32)
            theta_list = np.flip(idx, axis=0).astype(np.float32).reshape((1, -1))

            r = r.astype(np.float32)
            r_ap = r_ap.astype(np.float32)

            x, y = cv2.polarToCart(
                r_ap, theta_list.repeat(r.shape[0], 0), angleInDegrees=True
            )  # 360    360
            xy_ap = np.concatenate(
                [
                    x.reshape((-1, self.cfg.node_num, 1)),
                    y.reshape((-1, self.cfg.node_num, 1)),
                ],
                axis=2,
            )
            x, y = cv2.polarToCart(
                r, theta_list.repeat(r.shape[0], 0), angleInDegrees=True
            )  # 360    360
            xy = np.concatenate(
                [
                    x.reshape((-1, self.cfg.node_num, 1)),
                    y.reshape((-1, self.cfg.node_num, 1)),
                ],
                axis=2,
            )

            polygon_pts_ap = contour_center + xy_ap
            polygon_pts = contour_center + xy

            mask_ap_reduced = []
            for j in range(polygon_pts_ap.shape[0]):
                img = Image.new("L", (self.w, self.h))
                ImageDraw.Draw(img).polygon(
                    polygon_pts_ap[j, :, :].astype(np.float32), fill=1, outline=True
                )
                mask_ap = np.array(img)
                mask_ap_reduced.append(mask_ap)

                img = Image.new("L", (self.w, self.h))
                ImageDraw.Draw(img).polygon(
                    polygon_pts[j, :, :].astype(np.float32), fill=1, outline=True
                )
                mask = np.array(img)

                self.iou_ap.update(comp_iou(mask, mask_ap))
                f_measure = batched_f_measure(
                    mask[np.newaxis],
                    mask_ap[np.newaxis],
                    average_over_objects=True,
                    nb_objects=None,
                    bound_th=self.cfg.bound_th,
                )
                self.f_ap.update(f_measure.item())

            mask_ap = grow_mask(mask_ap_reduced)
            self.iou_gt.update(comp_iou(seg_gt, mask_ap))
            f_measure = batched_f_measure(
                seg_gt[np.newaxis],
                mask_ap[np.newaxis],
                average_over_objects=True,
                nb_objects=None,
                bound_th=self.cfg.bound_th,
            )
            self.f_gt.update(f_measure.item())

            if self.cfg.save:
                """h5 data structure:
                |-- S1.h5
                    |-- img_name
                        |-- meta0                  # meta data for each object in an img
                            |-- id                 # class No.
                            |-- bbox               # bbox 1 x 4
                            |-- r                  # radius N x 360
                            |-- pts_ap             # boundary pts N x 360 x 2
                            |-- mask_ap            # approx mask N x h x w
                            |-- contour_center     # centers N x 2
                            |-- id_xyxy            # bbox N x 5
                            |-- c                  # coef N x dim
                            |-- r_reduced          # reduced radius N x 360
                            |-- pts_reduced        # reduced boundary pts N x 360 x 2
                            |-- mask_per_contour_reduced  # reduced approx mask per contour N x h x w
                            |-- mask_ap_reduced    # reduced approx mask h x w
                        |-- meta1
                            ...
                """
                if "c" in meta.keys():
                    del meta["c"]
                meta.create_dataset("c", data=c)

                if "r_reduced" in meta.keys():
                    del meta["r_reduced"]
                meta.create_dataset("r_reduced", data=r_ap)

                if "pts_reduced" in meta.keys():
                    del meta["pts_reduced"]
                meta.create_dataset("pts_reduced", data=polygon_pts_ap)

                if "mask_per_contour_reduced" in meta.keys():
                    del meta["mask_per_contour_reduced"]
                meta.create_dataset(
                    "mask_per_contour_reduced", data=np.stack(mask_ap_reduced, axis=0)
                )

                if "mask_ap_reduced" in meta.keys():
                    del meta["mask_ap_reduced"]
                meta.create_dataset("mask_ap_reduced", data=mask_ap)

            print(self.iou_ap.summary())
            print(self.f_ap.summary())
            print(self.iou_gt.summary())
            print(self.f_gt.summary())

            self.tmp = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            self.visualize.update_image(self.img[0], normalized=True)
            self.visualize.update_image_name(self.img_name)

            self.visualize.show["ap_mask"] = np.copy(self.visualize.show["img"])
            self.visualize.show["gt_mask"] = np.copy(self.visualize.show["img"])

            img = Image.new("L", (self.w, self.h))
            self.visualize.draw_mask_cv_2(
                data=mask_ap, name="ap_mask", ref_name="ap_mask", color=(0, 0, 255)
            )

            img = Image.new("L", (self.w, self.h))
            self.visualize.draw_mask_cv_2(
                data=seg_gt, name="gt_mask", ref_name="gt_mask", color=(0, 255, 0)
            )

            if self.cfg.display:
                mkdir(dir_)
                self.visualize.display_saveimg_v2(
                    dir_name=dir_,
                    file_name=str(self.cfg.dim) + "_" + str(i) + ".jpg",
                    list=["img", "ap_mask", "gt_mask"],
                )

    def compare_contour(self):
        out = self.data

        results = []
        r_coord = torch.FloatTensor([])
        r_coord_id = []
        r_coord_cen = []
        check = np.full(len(out), True, dtype=np.bool)

        for i in range(len(out)):
            r_coord_tmp = out[i]["r"]
            if len(r_coord_tmp) == 0:
                check[i] = False
            else:
                r_coord = torch.cat(
                    (
                        r_coord,
                        torch.tensor(r_coord_tmp).type(torch.float32).view(-1, 1),
                    ),
                    dim=1,
                )
                r_coord_id.append(out[i]["id_xyxy"][-1])
                r_coord_cen.append(np.array(out[i]["center"]))

        dir_ = os.path.join(self.cfg.output_dir, "display_approx", self.img_name)

        n = 0
        for i in range(len(out)):
            if check[i] == False:
                result = {}
                result["r"] = np.array([])
                result["c"] = np.array([])
                result["center"] = np.array([])
                result["id_xyxy"] = np.array([])
                results.append(result)
                continue

            U = self.U[:, : self.cfg.dim]
            U_t = U.clone().permute(1, 0)
            r_coord_ = r_coord[:, n : n + 1].type(torch.float)
            c_ = torch.matmul(U_t, r_coord_) / self.cfg.max_dist
            r_coord_ap_ = torch.matmul(U, c_) * self.cfg.max_dist

            U = self.U_fms[:, : self.cfg.dim]
            U_t = U.clone().permute(1, 0)
            r_coord_ = r_coord[:, n : n + 1].type(torch.float)
            c_ = torch.matmul(U_t, r_coord_) / self.cfg.max_dist
            r_coord_ap_fms = torch.matmul(U, c_) * self.cfg.max_dist

            U = self.U_dpcp[:, : self.cfg.dim]
            U_t = U.clone().permute(1, 0)
            r_coord_ = r_coord[:, n : n + 1].type(torch.float)
            c_ = torch.matmul(U_t, r_coord_) / self.cfg.max_dist
            r_coord_ap_dpcp = torch.matmul(U, c_) * self.cfg.max_dist

            result = {}
            result["r"] = to_np(r_coord_)[:, 0]
            result["c"] = to_np(c_)[:, 0]
            result["center"] = r_coord_cen[n]
            result["id_xyxy"] = out[i]["id_xyxy"]

            self.cen = np.repeat(
                r_coord_cen[n].reshape(1, -1), self.cfg.node_num, 0
            ).astype(np.float32)

            idx = np.linspace(0, 359, self.cfg.node_num).astype(np.int32)
            theta_list = np.flip(idx, axis=0).astype(np.float32)
            x, y = cv2.polarToCart(
                to_np(r_coord_ap_.T)[0], theta_list, angleInDegrees=True
            )  # 360    360
            xy_ap = np.concatenate((x, y), axis=1)
            x, y = cv2.polarToCart(
                to_np(r_coord_ap_fms.T)[0], theta_list, angleInDegrees=True
            )  # 360    360
            xy_ap_fms = np.concatenate((x, y), axis=1)
            x, y = cv2.polarToCart(
                to_np(r_coord_ap_dpcp.T)[0], theta_list, angleInDegrees=True
            )  # 360    360
            xy_ap_dpcp = np.concatenate((x, y), axis=1)
            x, y = cv2.polarToCart(
                to_np(r_coord_.T)[0], theta_list, angleInDegrees=True
            )  # 360    360
            xy = np.concatenate((x, y), axis=1)

            polygon_pts_ap = self.cen + xy_ap
            polygon_pts_ap_fms = self.cen + xy_ap_fms
            polygon_pts_ap_dpcp = self.cen + xy_ap_dpcp
            polygon_pts = self.cen + xy

            self.img = self.label_all[i]["cropped_img"]
            self.c, self.h, self.w = self.img[0].shape
            self.tmp = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            self.visualize.update_image(self.img[0])
            self.visualize.update_image_name(self.img_name)

            self.visualize.show["ap_mask"] = np.copy(self.visualize.show["img"])
            self.visualize.show["ap_mask_fms"] = np.copy(self.visualize.show["img"])
            self.visualize.show["ap_mask_dpcp"] = np.copy(self.visualize.show["img"])
            self.visualize.show["gt_mask"] = np.copy(self.visualize.show["img"])

            img = Image.new("L", (self.w, self.h))
            ImageDraw.Draw(img).polygon(
                polygon_pts_ap.astype(np.float32), fill=1, outline=True
            )
            mask_ap = np.array(img)
            self.visualize.draw_mask_cv_2(
                data=mask_ap, name="ap_mask", ref_name="ap_mask", color=(0, 0, 255)
            )
            # self.visualize.draw_polyline_cv(data=polygon_pts_ap, name='ap_mask', ref_name='ap_mask', color=(0, 0, 255))

            img = Image.new("L", (self.w, self.h))
            ImageDraw.Draw(img).polygon(
                polygon_pts_ap_fms.astype(np.float32), fill=1, outline=True
            )
            mask_ap_fms = np.array(img)
            self.visualize.draw_mask_cv_2(
                data=mask_ap_fms,
                name="ap_mask_fms",
                ref_name="ap_mask_fms",
                color=(0, 0, 255),
            )
            # self.visualize.draw_polyline_cv(data=polygon_pts_ap_fms, name='ap_mask_fms', ref_name='ap_mask_fms', color=(0, 0, 255))

            img = Image.new("L", (self.w, self.h))
            ImageDraw.Draw(img).polygon(
                polygon_pts_ap_dpcp.astype(np.float32), fill=1, outline=True
            )
            mask_ap_dpcp = np.array(img)
            self.visualize.draw_mask_cv_2(
                data=mask_ap_dpcp,
                name="ap_mask_dpcp",
                ref_name="ap_mask_dpcp",
                color=(0, 0, 255),
            )
            # self.visualize.draw_polyline_cv(data=polygon_pts_ap_dpcp, name='ap_mask_dpcp', ref_name='ap_mask_dpcp', color=(0, 0, 255))

            img = Image.new("L", (self.w, self.h))
            ImageDraw.Draw(img).polygon(
                polygon_pts.astype(np.float32), fill=1, outline=True
            )
            mask = np.array(img)
            # mask = to_np(self.label_all[i]['seg_mask'][0])
            self.visualize.draw_mask_cv_2(
                data=mask, name="gt_mask", ref_name="gt_mask", color=(0, 255, 0)
            )
            # self.visualize.draw_polyline_cv(data=polygon_pts, name='gt_mask', ref_name='gt_mask', color=(0, 255, 0))

            f_measure = batched_f_measure(
                mask[np.newaxis],
                mask_ap[np.newaxis],
                average_over_objects=True,
                nb_objects=None,
                bound_th=self.cfg.bound_th,
            )

            f_measure_fms = batched_f_measure(
                mask[np.newaxis],
                mask_ap_fms[np.newaxis],
                average_over_objects=True,
                nb_objects=None,
                bound_th=self.cfg.bound_th,
            )

            f_measure_dpcp = batched_f_measure(
                mask[np.newaxis],
                mask_ap_dpcp[np.newaxis],
                average_over_objects=True,
                nb_objects=None,
                bound_th=self.cfg.bound_th,
            )

            mask_overlap = mask + mask_ap
            non_lap = (mask_overlap == 1).astype(np.uint8)
            over_lap = (mask_overlap == 2).astype(np.uint8)
            iou_ = over_lap.sum() / (over_lap.sum() + non_lap.sum())

            self.iou.append(torch.tensor(iou_).type(torch.float32))
            self.F.append(torch.tensor(f_measure).type(torch.float32))
            results.append(result)
            n += 1

            if f_measure_fms > f_measure_dpcp + 0.2 and f_measure_dpcp > f_measure:
                mkdir(dir_)
                print(dir_)
                print(f_measure_fms, f_measure_dpcp, f_measure)
                self.visualize.display_saveimg_v2(
                    dir_name=dir_,
                    file_name=str(self.cfg.dim) + "_" + str(i) + ".jpg",
                    list=["img", "ap_mask", "ap_mask_dpcp", "ap_mask_fms", "gt_mask"],
                )

                # exit()

        return results

    def run_compare(self):
        print("start")
        self.make_dict()

        self.U_fms = load_pickle(self.cfg.U_dir + "U_fms_" + str(self.cfg.dim))
        self.U_dpcp = load_pickle(self.cfg.U_dir + "U_dpcp_" + str(self.cfg.dim))
        self.U = load_pickle(self.cfg.U_dir + "U")

        for i, batch in enumerate(tqdm(self.dataloader)):
            print(i)
            self.label_all = batch["output"]
            self.img_name = batch["img_name"][0]

            err_data = load_pickle(self.cfg.output_dir + "pickle/datalist_error")
            if self.img_name in err_data:
                continue

            self.data = load_pickle(self.cfg.output_dir + "pickle/" + self.img_name)[0]
            out_f = list()
            out_f.append(self.compare_contour())

    def make_dict(self):
        self.iou_ap = AverageMeter("iou_ap")
        self.iou_gt = AverageMeter("iou_gt")
        self.f_ap = AverageMeter("f_ap")
        self.f_gt = AverageMeter("f_gt")

    def run(self):
        print("start")
        self.make_dict()

        method = self.cfg.args.method  # ['svd', 'dpcp', 'fms']
        if method == "svd":
            self.U = load_pickle(self.cfg.U_dir + "U_svd").numpy()
        else:
            self.U = load_pickle(
                self.cfg.U_dir + "U_" + method + "_" + str(self.cfg.dim)
            ).numpy()

        h = h5py.File(os.path.join(self.cfg.output_dir, "S1.h5"), "a")

        center_num = {}
        for i, batch in enumerate(tqdm(self.dataloader)):
            print(i)
            self.label_all = batch["output"]
            self.img_name = batch["img_name"][0]
            if self.img_name in h.keys():
                self.data = h[self.img_name]
                self.approximate_contour()
        h.close()
