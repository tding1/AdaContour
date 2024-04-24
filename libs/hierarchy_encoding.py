import cv2
import numpy as np
from PIL import Image, ImageDraw

from label_utils.label_centerdeg_custom import *
from label_utils.decoding import decoding_theta
from libs.utils import *


def reorg_cc(part1, part2):
    output = cv2.connectedComponentsWithStats(part1.astype(np.uint8), 4, cv2.CV_32S)
    num_labels = output[0]
    stats = output[2]

    loc = []
    for i in range(num_labels):
        loc.append(output[1] == i)

    areas = stats[:, -1]
    ind = np.argmax(areas[1:])
    ind += 1

    keep = np.zeros(num_labels)
    keep[0] = 1
    keep[ind] = 1

    for i in range(num_labels):
        if keep[i] == 0:
            part2[loc[i]] = 1
            part1[loc[i]] = 0

    return part1, part2


def f(x, y, x1, x2, y1, y2):
    # line equation give (x1, y1), (x2, y2)
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)


def remove_isolate_points(img, cc_output):
    output = cc_output

    num_labels = output[0]
    stats = output[2]

    loc = []
    for i in range(num_labels):
        loc.append(output[1] == i)

    areas = stats[:, -1]
    ind = np.argmax(areas[1:])
    ind += 1

    keep = np.zeros(num_labels)
    keep[0] = 1
    keep[ind] = 1

    for i in range(num_labels):
        if keep[i] == 0:
            img[loc[i]] = 0

    return img, ind


def grow_mask(mask_ap_list):
    if len(mask_ap_list) == 0:
        return None
    elif len(mask_ap_list) == 1:
        return mask_ap_list[0]

    # only grow if more than one mask
    mask_ap = 0
    for mm in mask_ap_list:
        if len(mm) == 0:  # skip leaf node with no center found
            continue
        mask_ap = (mask_ap + mm).clip(0, 1).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask_ap = cv2.morphologyEx(mask_ap, cv2.MORPH_CLOSE, kernel)
    return mask_ap


def comp_iou(mask1, mask2):
    mask_overlap = mask1.astype(np.uint8) + mask2.astype(np.uint8)
    non_lap = (mask_overlap == 1).astype(np.uint8)
    over_lap = (mask_overlap == 2).astype(np.uint8)
    iou = over_lap.sum() / (over_lap.sum() + non_lap.sum())
    return iou


def seg2maskap(label, node_num, r_coord_xy, mode):
    result = {}

    label_id = int(label["id"])
    img = label["cropped_img"]
    c, h, w = img[0].shape

    out = dict()
    out["categ_id"] = label_id
    out.update(runOneImage(label["bbox"], label["seg_mask"][0], mode))
    error = out["check"]

    if error:
        print("*************************error***************************")
        raise ValueError

    out.update(decoding_theta(out, node_num))
    idx = np.linspace(0, 360, node_num, endpoint=False).astype(np.int32)
    # TD: choose num of points corres to the node_num instead of 360 (all of them)
    result["r"] = np.float32(out["r"])[idx]
    # result['r'] = out['r']
    result["pts"] = out["contour_pts"]
    result["center"] = list(out["center"])
    polygon_pts_ap = np.array(result["pts"], dtype=np.float32)
    mask = to_np(label["seg_mask"][0])

    cen = np.repeat(np.array(result["center"])[:, np.newaxis], node_num, 1).T
    xy = np.flip(np.array(result["r"]))[:, np.newaxis] * r_coord_xy
    polygon_pts_ap = cen + xy

    img = Image.new("L", (w, h))
    ImageDraw.Draw(img).polygon(
        np.round(polygon_pts_ap).astype(np.float32), fill=1, outline=True
    )
    mask_ap = np.array(img)
    iou = comp_iou(mask, mask_ap)

    return (
        mask,
        mask_ap,
        iou,
        result["center"],
        result["r"],
        result["pts"],
        label["bbox"],
    )


def hierarchy_encoding(label, result, node_num, r_coord_xy, mode, max_depth=1):
    mask = to_np(label["seg_mask"][0].squeeze())
    if np.sum(mask) == 0:  # avoid some bad case where no mask at all
        return result
    img = label["cropped_img"]
    c, h, w = img[0].shape
    props = measure.regionprops(measure.label(mask))
    sdt = props[0].solidity
    area = props[0].area

    if max_depth == 0 or sdt > 0.9 or area < 0.001 * w * h:
        try:
            (
                mask_part,
                mask_part_ap,
                iou_part,
                center_part,
                r_part,
                pts_part,
                bbox_part,
            ) = seg2maskap(label, node_num, r_coord_xy, mode=mode)
            result["mask_ap_list"].append(mask_part_ap)
            result["center_list"].append(center_part)
            result["r_list"].append(r_part)
            result["pts_list"].append(pts_part)
            tmp = np.zeros(5, dtype=np.int32)
            tmp[:-1] = to_np(bbox_part).reshape(-1)
            tmp[-1] = int(label["id"])
            result["id_xyxy_list"].append(tmp)
            result["error"] = False
            return result
        except:
            # result["mask_ap_list"].append([])
            # result["center_list"].append([])
            # result["r_list"].append([])
            # result["pts_list"].append([])
            # result["id_xyxy_list"].append([])
            return result

    y, x = np.where(mask == 1)
    data = np.array([(x[k], y[k]) for k in range(len(y))])

    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data.astype(np.float32), mean)
    cntr = (int(mean[0, 0]), int(mean[0, 1]))
    p2 = cntr - 0.1 * eigenvectors[1, :] * eigenvalues[1, 0]

    # cntr_ = props[0].centroid
    # cntr = cntr_[1], cntr_[0]
    # inertia = props[0].inertia_tensor
    # p2 = cntr - 0.1 * inertia[0,:]

    x1, y1 = cntr
    x2, y2 = p2

    part1 = np.zeros_like(mask)
    part2 = np.zeros_like(mask)
    for d in data:
        if f(d[0], d[1], x1, x2, y1, y2) >= 0:
            part1[d[1], d[0]] = 1
        elif f(d[0], d[1], x1, x2, y1, y2) < 0:
            part2[d[1], d[0]] = 1

    output = cv2.connectedComponentsWithStats(part1.astype(np.uint8), 4, cv2.CV_32S)
    num_labels = output[0]
    if num_labels > 2:
        part1, part2 = reorg_cc(part1, part2)

    output = cv2.connectedComponentsWithStats(part2.astype(np.uint8), 4, cv2.CV_32S)
    num_labels = output[0]
    if num_labels > 2:
        part1, part2 = reorg_cc(part2, part1)

    label_1 = label.copy()
    label_2 = label.copy()

    output = cv2.connectedComponentsWithStats(part1.astype(np.uint8), 4, cv2.CV_32S)
    num_labels = output[0]
    assert num_labels == 2
    # if num_labels > 2:
    #     part1, ind = remove_isolate_points(part1, output)
    # else:
    ind = 1
    stats = output[2]
    bbox = torch.FloatTensor(
        [
            stats[ind, 0],
            stats[ind, 1],
            stats[ind, 0] + stats[ind, 2],
            stats[ind, 1] + stats[ind, 3],
        ]
    ).view(1, -1)
    b_cen = torch.FloatTensor(
        [stats[ind, 0] + stats[ind, 2] / 2, stats[ind, 1] + stats[ind, 3] / 2]
    ).view(1, -1)
    label_1["seg_mask"] = torch.from_numpy(part1)[None, :, :]
    label_1["bbox"] = bbox
    label_1["bbox_center"] = b_cen

    output = cv2.connectedComponentsWithStats(part2.astype(np.uint8), 4, cv2.CV_32S)
    num_labels = output[0]
    assert num_labels == 2
    # if num_labels > 2:
    #     part2, ind = remove_isolate_points(part2, output)
    # else:
    ind = 1
    stats = output[2]
    bbox = torch.FloatTensor(
        [
            stats[ind, 0],
            stats[ind, 1],
            stats[ind, 0] + stats[ind, 2],
            stats[ind, 1] + stats[ind, 3],
        ]
    ).view(1, -1)
    b_cen = torch.FloatTensor(
        [stats[ind, 0] + stats[ind, 2] / 2, stats[ind, 1] + stats[ind, 3] / 2]
    ).view(1, -1)
    label_2["seg_mask"] = torch.from_numpy(part2)[None, :, :]
    label_2["bbox"] = bbox
    label_2["bbox_center"] = b_cen

    result["hierarchy_list"].append(np.hstack((part1, part2)))

    hierarchy_encoding(label_1, result, node_num, r_coord_xy, mode, max_depth - 1)
    hierarchy_encoding(label_2, result, node_num, r_coord_xy, mode, max_depth - 1)
