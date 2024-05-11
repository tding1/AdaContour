from datasets.dataset_coco_not_crop import *
from datasets.dataset_sbd_not_crop import *
from datasets.dataset_demo import *
from visualizes.visualize import Visualize


def prepare_dataloader(cfg, dict_DB):
    if cfg.dataset == "coco":
        dataset = Dataset_coco(cfg=cfg, datalist=cfg.datalist)
    elif cfg.dataset == "sbd":
        dataset = SBD(cfg=cfg, datalist=cfg.datalist)
    elif cfg.dataset == "demo":
        dataset = Dataset_demo(cfg=cfg, datalist=cfg.datalist)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )

    dict_DB["dataloader"] = dataloader

    return dict_DB


def prepare_visualization(cfg, dict_DB):
    dict_DB["visualize"] = Visualize(cfg)
    return dict_DB
