import os

import torch.nn.parallel
import torch.optim

from options.config import Config
from options.args import *
from libs.prepare import *
from libs.S1_encoding import *
from libs.S2_factorization import *
from libs.S3_convert import *


def run_hierarchy_encoding(cfg, dict_DB):
    ada_contour_generator = Generate_AdaContour(cfg, dict_DB)
    ada_contour_generator.run()


def run_factorization(cfg, dict_DB):
    factorization = Factorization(cfg, dict_DB)
    factorization.run()


def run_convert(cfg, dict_DB):
    convertor = Convert(cfg, dict_DB)
    convertor.run()


def main():
    # option
    args = parse_args()
    cfg = Config(args)

    # gpu setting
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    torch.backends.cudnn.deterministic = True

    # prepare
    dict_DB = dict()
    dict_DB = prepare_visualization(cfg, dict_DB)
    dict_DB = prepare_dataloader(cfg, dict_DB)

    # run
    if cfg.stage == "encoding":
        if cfg.mode == "hierarchy_encoding":
            run_hierarchy_encoding(cfg, dict_DB)
        else:
            raise NotImplementedError
    elif cfg.stage == "factorization":
        run_factorization(cfg, dict_DB)
    elif cfg.stage == "convert":
        run_convert(cfg, dict_DB)
    else:
        print("Please mode check!")


if __name__ == "__main__":
    main()
