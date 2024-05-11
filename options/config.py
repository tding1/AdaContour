from ast import arg
import os
import math


class Config(object):
    def __init__(self, args):
        self.head_path = os.path.dirname(os.getcwd()) + "/"

        # dataset dir
        self.dataset = args.dataset  # ['coco', 'sbd']
        self.datalist = args.datalist  # ['train', 'test', 'val']
        self.stage = args.stage  # ['encoding', 'factorization', 'convert']
        self.mode = args.mode  # ['hierarchy_encoding', 'encoding']

        if self.dataset == "sbd":
            self.img_dir = self.head_path + "Preprocess_SBD/data/img/"
        elif self.dataset == "coco":
            self.img_dir = self.head_path + "Preprocessing/data/coco/{}2017/".format(
                self.datalist
            )
        elif self.dataset == "demo":
            self.img_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data/")

        self.label_path_seg = "{}_seg.txt".format(self.datalist)
        self.label_path_bb = "{}_bb.txt".format(self.datalist)

        self.node_num = args.node_num

        self.output_dir = (
            self.head_path if self.dataset != "demo" else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output/")
            + "output_"
            + self.dataset
            + "_"
            + self.datalist
            + "_v1_node_{}".format(self.node_num)
            + "_iouth_"
            + str(args.thresd_iou)
            + "_mode_"
            + args.mode
            + "_D_"
            + str(args.max_depth)
            + "_center_"
            + args.process_mode
            + "/"
        )

        if args.val_U:
            self.U_dir = (
                self.head_path if self.dataset != "demo" else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output/")
                + "output_"
                + self.dataset
                + "_val"
                + "_v1_node_{}".format(self.node_num)
                + "_iouth_"
                + str(args.thresd_iou)
                + "_mode_"
                + args.mode
                + "_D_"
                + str(args.max_depth)
                + "_center_"
                + args.process_mode
                + "/"
            )
        else:
            self.U_dir = (
                self.head_path
                + "output_"
                + self.dataset
                + "_train"
                + "_v1_node_{}".format(self.node_num)
                + "_iouth_"
                + str(args.thresd_iou)
                + "_mode_"
                + args.mode
                + "_D_"
                + str(args.max_depth)
                + "_center_"
                + args.process_mode
                + "/"
            )

        if self.dataset == "demo":
            self.U_dir = (
                self.head_path if self.dataset != "demo" else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output/")
                + "output_demo_demo"
                + "_v1_node_{}".format(self.node_num)
                + "_iouth_"
                + str(args.thresd_iou)
                + "_mode_"
                + args.mode
                + "_D_"
                + str(args.max_depth)
                + "_center_"
                + args.process_mode
                + "/"
            )

        # other setting
        self.process_mode = args.process_mode
        self.bound_th = args.bound_th
        self.fms = args.fms

        # dataloader
        self.gpu_id = args.gpu_id
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size

        # constant
        self.height = 416
        self.width = 416
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.crop_w, self.crop_h = 256, 256
        self.max_dist = math.sqrt(
            self.width / 2 * self.width / 2 + self.height / 2 * self.height / 2
        )

        self.dim = args.dim
        self.max_depth = args.max_depth
        self.thresd_iou = args.thresd_iou

        # visualization
        self.display = args.display

        # save
        self.save = args.save

        # get everything from args
        self.args = args
