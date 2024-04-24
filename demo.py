import os
import cv2
import math
import torch
import argparse
import numpy as np
from skimage import measure

from libs.utils import to_np, draw_seg
from libs.hierarchy_encoding import hierarchy_encoding, grow_mask, comp_iou


def generate_shape(label, max_depth, mode):
    node_num = 360
    r_coord = np.linspace(0, 360, node_num, endpoint=False)
    r_coord_x = np.cos(r_coord * 2 * math.pi / 360)[:, np.newaxis]
    r_coord_y = np.sin(r_coord * 2 * math.pi / 360)[:, np.newaxis]
    r_coord_xy = np.concatenate((r_coord_x, r_coord_y), axis=1)

    img = label["cropped_img"]
    _, c, h, w = img.shape
    mask_gt = to_np(label['seg_mask'][0])

    result = {
                "id_xyxy_list": [],
                "r_list": [],
                "pts_list": [],
                "center_list": [],
                "mask_ap_list": [],
                "hierarchy_list": [],
                "error": True,
            }
    
    if (max_depth == 0):  # original eigencontour, no need to isolate disjoint parts
        hierarchy_encoding(
            label,
            result,
            node_num,
            r_coord_xy,
            mode='ese_ori',
            max_depth=max_depth,
        )
        mask_ap = grow_mask(result['mask_ap_list'])
        iou = comp_iou(mask_gt, mask_ap)
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
                node_num,
                r_coord_xy,
                mode=mode,
                max_depth=max_depth,
            )

        mask_ap = grow_mask(result['mask_ap_list'])
        iou = comp_iou(mask, mask_ap)

    for i, cen in enumerate(result['center_list']):
        if i == 0:
            vis = cv2.circle(np.stack((mask_ap,)*3, axis=-1)*255, (int(cen[0]), int(cen[1])), 1, (0,0,255), 8)
        else:
            vis = cv2.circle(vis, (int(cen[0]), int(cen[1])), 1, (0,0,255), 8)
    
    if max_depth == 0:
        seg_info = None
    else:
        seg_info = draw_seg(result, mask_ap)

    num_centers = len(result['center_list'])

    return vis, seg_info, iou, num_centers


parser = argparse.ArgumentParser()
parser.add_argument('--image_id', type=int, default=1)
args = parser.parse_args()

image_id = args.image_id
image = cv2.imread("sample_data/image"+str(image_id)+".png").transpose(2, 0, 1)
mask = cv2.imread("sample_data/image"+str(image_id)+"-mask.png").transpose(2, 0, 1)[0].astype(bool).astype(np.uint8)
h, w = mask.shape

label = {}
label["cropped_img"] = torch.from_numpy(image).unsqueeze(0)
label["seg_mask"] = torch.from_numpy(mask).unsqueeze(0)

# placeholder for id and bbox since they are not used here
label['id'] = 0
label['bbox'] = torch.tensor([[0,0,1,1]], dtype=torch.float)

props = measure.regionprops(measure.label(mask))
sdt = props[0].solidity

os.makedirs('output', exist_ok=True)

print('=================================================================')
print('image id = %d' % image_id)
print('solidtiy = %.2f' % sdt)
print('eccentricity = %.2f' % props[0].eccentricity)
print('area ratio = %.2f (%d / %d)' % (float(props[0].area / (w*h)), props[0].area, w*h ))
print('=================================================================')

for max_depth in range(6):
    vis, seg_info, iou, num_centers = generate_shape(label, max_depth, mode='hybrid')
    print('depth = %d, iou = %.2f, # of centers = %d' % (max_depth, iou, num_centers))
    cv2.imwrite('output/image_id_%d_depth_%d.png' % (image_id, max_depth), vis)
    if seg_info is not None:
        cv2.imwrite('output/image_id_%d_depth_%d_seg.png' % (image_id, max_depth), seg_info)

print('Done! All results saved in output/')
print('=================================================================')
