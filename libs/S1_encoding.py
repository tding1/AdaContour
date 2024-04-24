import os
import math

import h5py
import numpy as np
from tqdm import tqdm
from davisinteractive.metrics import batched_f_measure

from libs.utils import AverageMeter, to_np
from libs.hierarchy_encoding import *


class Generate_AdaContour(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg
        self.dataloader = dict_DB["dataloader"]
        self.visualize = dict_DB["visualize"]
        self.size = np.float32([cfg.height, cfg.width])

        self.datalist = []
        self.datalist_error = []

        self.r_coord = np.linspace(0, 360, self.cfg.node_num, endpoint=False)
        self.r_coord_x = np.cos(self.r_coord * 2 * math.pi / 360)[:, np.newaxis]
        self.r_coord_y = np.sin(self.r_coord * 2 * math.pi / 360)[:, np.newaxis]
        self.r_coord_xy = np.concatenate((self.r_coord_x, self.r_coord_y), axis=1)

        self.iou = AverageMeter("iou")
        self.F = AverageMeter("F")

    def generate_shape(self):
        results = []
        self.error = False

        for i, label in enumerate(self.label_all):

            self.img = label["cropped_img"]
            self.c, self.h, self.w = self.img[0].shape
            self.tmp = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            self.visualize.update_image(self.img[0])
            self.visualize.update_image_name(self.img_name)

            self.visualize.show["polygon_mask"] = np.copy(self.tmp)
            self.visualize.show["ap_mask"] = np.copy(self.tmp)

            mask = to_np(label["seg_mask"][0])

            result = {
                "id_xyxy_list": [],
                "r_list": [],
                "pts_list": [],
                "center_list": [],
                "mask_ap_list": [],
                "hierarchy_list": [],
                "error": True,
            }

            if (
                self.cfg.max_depth == 0
            ):  # original eigencontour, no need to isolate disjoint parts
                hierarchy_encoding(
                    label,
                    result,
                    self.cfg.node_num,
                    self.r_coord_xy,
                    self.cfg.process_mode,
                    max_depth=self.cfg.max_depth,
                )
            else:
                output = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

                label_cc = []
                num_labels = output[0]
                stats = output[2]
                areas = stats[:, -1]
                keep = areas > 100  # remove isolate area less than 100
                for jj in range(num_labels):
                    if jj > 0 and keep[jj]:
                        mask_cc = output[1] == jj
                        bbox_cc = torch.FloatTensor(
                            [
                                stats[jj, 0],
                                stats[jj, 1],
                                stats[jj, 0] + stats[jj, 2],
                                stats[jj, 1] + stats[jj, 3],
                            ]
                        ).view(1, -1)
                        b_cen_cc = torch.FloatTensor(
                            [
                                stats[jj, 0] + stats[jj, 2] / 2,
                                stats[jj, 1] + stats[jj, 3] / 2,
                            ]
                        ).view(1, -1)

                        label_ = label.copy()
                        label_["seg_mask"] = torch.from_numpy(mask_cc)[None, :, :]
                        label_["bbox"] = bbox_cc
                        label_["bbox_center"] = b_cen_cc
                        label_cc.append(label_)

                for ll in label_cc:
                    hierarchy_encoding(
                        ll,
                        result,
                        self.cfg.node_num,
                        self.r_coord_xy,
                        self.cfg.process_mode,
                        max_depth=self.cfg.max_depth,
                    )

            if result["error"]:
                self.error = True
            else:
                self.error = False
                mask_ap = grow_mask(result["mask_ap_list"])
                iou = comp_iou(mask, mask_ap)
                f_measure = batched_f_measure(
                    mask.squeeze()[np.newaxis],
                    mask_ap[np.newaxis],
                    average_over_objects=True,
                    nb_objects=None,
                    bound_th=self.cfg.bound_th,
                )
                self.iou.update(iou)
                self.F.update(f_measure.item())
                print(self.iou.summary(), self.F.summary())

            self.error_all.append(self.error)
            results.append(result)

            if self.cfg.display and not self.error:
                self.visualize.draw_mask_cv(
                    data=mask,
                    name="polygon_mask",
                    ref_name="polygon_mask",
                    color=(0, 0, 255),
                )
                self.visualize.draw_mask_cv(
                    data=mask_ap, name="ap_mask", ref_name="ap_mask", color=(0, 255, 0)
                )
                dir_name = self.cfg.output_dir + "display/"
                file_name = self.img_name + "_{}".format(str(i))
                self.visualize.display_saveimg(
                    dir_name=dir_name,
                    file_name=file_name,
                    list=["img", "polygon_mask", "ap_mask"],
                )

        return results

    def run(self):
        print("start")
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        outfile = "S1.h5" if self.cfg.save else "S1_tmp.h5"
        with h5py.File(os.path.join(self.cfg.output_dir, outfile), "w") as h:
            for i, batch in enumerate(tqdm(self.dataloader)):

                self.error_all = []
                self.label_all = batch["output"]
                self.img_name = batch["img_name"][0]

                out_f = self.generate_shape()

                error_all = (~np.array(self.error_all)).sum()
                if error_all != 0 and self.cfg.save:
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
                            |-- meta1
                                ...
                    """
                    h_name = h.create_group(self.img_name)
                    k = 0
                    for j in range(len(out_f)):
                        error = out_f[j]["error"]
                        if not error:
                            h_meta = h_name.create_group("meta" + str(k))

                            id = self.label_all[j]["id"].item()
                            bbox = self.label_all[j]["bbox"].squeeze()
                            h_meta.create_dataset("id", data=id)
                            h_meta.create_dataset("bbox", data=bbox)

                            out = out_f[j]

                            contour_center = np.concatenate(
                                out["center_list"], axis=0
                            ).reshape(-1, 2)
                            h_meta.create_dataset("contour_center", data=contour_center)

                            r = np.concatenate(out["r_list"], axis=0).reshape(-1, 360)
                            h_meta.create_dataset("r", data=r)

                            pts_ap = np.concatenate(out["pts_list"], axis=0).reshape(
                                -1, 360, 2
                            )
                            h_meta.create_dataset("pts_ap", data=pts_ap)

                            mask_ap = np.concatenate(
                                out["mask_ap_list"], axis=0
                            ).reshape(-1, self.cfg.height, self.cfg.width)
                            h_meta.create_dataset("mask_ap", data=mask_ap)

                            id_xyxy = np.concatenate(
                                out["id_xyxy_list"], axis=0
                            ).reshape(-1, 5)
                            h_meta.create_dataset("id_xyxy", data=id_xyxy)
                            k += 1

                print(
                    "image %d ===> %s clear, error_all %d"
                    % (i, self.img_name, error_all)
                )

        h.close()
