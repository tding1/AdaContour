import os

import h5py
import scipy.io
import numpy as np
from tqdm import tqdm
from PIL import Image as PILImage

from libs.utils import save_pickle


input_path = "./data"
output_path = "./data"

# Mat to png conversion for http://www.cs.berkeley.edu/~bharath2/codes/SBD/download.html
# 'GTcls' key is for class segmentation
# 'GTinst' key is for instance segmentation
def mat2pklcountour(mat_file):
    key = "GTinst"
    mat = scipy.io.loadmat(
        mat_file, mat_dtype=True, squeeze_me=True, struct_as_record=False
    )
    inst = mat[key].Boundaries
    category = np.array(mat[key].Categories).reshape(-1).tolist()
    segmentation = mat[key].Segmentation
    bboxs = list()
    contours_masks = {"contours": [], "masks": []}
    for i in range(len(category)):
        id = category[i]
        boundary = inst[i].toarray() if len(category) != 1 else inst.toarray()
        pts = np.argwhere(boundary == 1)
        h, w = pts[:, 0:1], pts[:, 1:2]
        pts = np.concatenate([w, h], axis=1)
        pts = pts.reshape(-1).tolist()
        pts = [id] + pts
        contours_masks["contours"].append(pts)
        min_w, min_h = np.min(w), np.min(h)
        max_w, max_h = np.max(w), np.max(h)
        bbox = [min_w, min_h, max_w, max_h, id]
        bboxs.append(bbox)
        mask = (segmentation == i + 1).astype(np.uint8)
        contours_masks["masks"].append(mask)
    return contours_masks, bboxs


def convert_mat2h5(mat_files, output_path, mode="train"):
    """h5 data structure:
    |-- raw_train.h5
        |-- filename
            |-- img
            |-- contour
                |-- 0
                |-- 1
                |-- ...
            |-- mask
                |-- 0
                |-- 1
                |-- ...
            |-- bbox
                |-- 0
                |-- 1
                |-- ...
    """
    if not mat_files:
        help("Input directory does not contain any Matlab files!\n")

    with h5py.File(os.path.join(output_path, "raw_" + mode + ".h5"), "w") as h:
        for mat in tqdm(mat_files):
            filename = mat.split(".")[1].split("/")[-1]
            image = np.array(
                PILImage.open(mat.replace("inst", "img").replace("mat", "jpg"))
            )
            contour_mask, bbox = mat2pklcountour(mat)
            contours, masks, bboxs = (
                contour_mask["contours"],
                contour_mask["masks"],
                bbox,
            )
            h_name = h.create_group(filename)
            h_name.create_dataset("img", data=image)
            h_contour = h_name.create_group("contour")
            h_mask = h_name.create_group("mask")
            h_bbox = h_name.create_group("bbox")
            for i, (contour, mask, bbox) in enumerate(zip(contours, masks, bboxs)):
                h_mask.create_dataset(str(i), data=np.array(mask))
                h_bbox.create_dataset(str(i), data=np.array(bbox))
                h_contour.create_dataset(str(i), data=np.array(contour))
    h.close()


def save_h5(input_path, output_path):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if os.path.isdir(input_path) and os.path.isdir(output_path):
        train_datalist_path = os.path.join(input_path, "train.txt")
        test_datalist_path = os.path.join(input_path, "val.txt")
        with open(train_datalist_path, "r") as f:
            train_mat_names = f.readlines()
        f.close()

        train_mat_files = []
        for mat_name in train_mat_names:
            train_mat_files.append(
                os.path.join(input_path, "inst", mat_name.split("\n")[0] + ".mat")
            )
        convert_mat2h5(train_mat_files, output_path, mode="train")

        with open(test_datalist_path, "r") as f:
            test_mat_names = f.readlines()
        f.close()

        test_mat_files = []
        for mat_name in test_mat_names:
            test_mat_files.append(
                os.path.join(input_path, "inst", mat_name.split("\n")[0] + ".mat")
            )
        convert_mat2h5(test_mat_files, output_path, mode="val")


def main(input_path, output_path):
    train_datalist_path = os.path.join(input_path, "train.txt")
    with open(train_datalist_path, "r") as f:
        train_mat_names = f.readlines()
    f.close()

    train_mat_files = []
    for mat_name in train_mat_names:
        train_mat_files.append(
            os.path.join(input_path, "inst", mat_name.split("\n")[0] + ".mat")
        )

    train_seg_dict = dict()
    train_bb_dict = dict()
    for mat in tqdm(train_mat_files):
        filename = mat.split(".")[1].split("/")[-1]

        contour_mask, bbox = mat2pklcountour(mat)
        contours, masks, bboxs = (
            contour_mask["contours"],
            contour_mask["masks"],
            bbox,
        )

        train_seg_dict[filename] = []
        train_bb_dict[filename] = []
        for (contour, mask, bbox) in zip(contours, masks, bboxs):
            train_seg_dict[filename].append(contour)
            train_bb_dict[filename].append(bbox)

    test_datalist_path = os.path.join(input_path, "val.txt")
    with open(test_datalist_path, "r") as f:
        test_mat_names = f.readlines()
    f.close()

    test_mat_files = []
    for mat_name in test_mat_names:
        test_mat_files.append(
            os.path.join(input_path, "inst", mat_name.split("\n")[0] + ".mat")
        )

    val_seg_dict = dict()
    val_bb_dict = dict()
    for mat in tqdm(test_mat_files):
        filename = mat.split(".")[1].split("/")[-1]

        contour_mask, bbox = mat2pklcountour(mat)
        contours, masks, bboxs = (
            contour_mask["contours"],
            contour_mask["masks"],
            bbox,
        )

        val_seg_dict[filename] = []
        val_bb_dict[filename] = []
        for (contour, mask, bbox) in zip(contours, masks, bboxs):
            val_seg_dict[filename].append(contour)
            val_bb_dict[filename].append(bbox)

    save_pickle(output_path, "train_seg", train_seg_dict)
    save_pickle(output_path, "train_bb", train_bb_dict)
    save_pickle(output_path, "val_seg", val_seg_dict)
    save_pickle(output_path, "val_bb", val_bb_dict)


if __name__ == "__main__":
    main(input_path, output_path)
