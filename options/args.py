import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Hello")
    parser.add_argument("--root", default="./raw_data", type=str)
    parser.add_argument(
        "--stage",
        default="encoding",
        type=str,
        choices=["encoding", "factorization", "convert"],
    )
    parser.add_argument(
        "--mode",
        default="hierarchy_encoding",
        type=str,
        choices=["hierarchy_encoding", "encoding"],
    )
    parser.add_argument("--node_num", default=360, type=int, help="node_num")
    parser.add_argument(
        "--max_depth", default=5, type=int, help="maximum depth for hierarchy encoding"
    )
    parser.add_argument("--dim", default=36, type=int, help="dim")
    parser.add_argument("--display", action="store_true", help="visualization")
    parser.add_argument("--fms", action="store_true", help="using FMS instead of SVD")
    parser.add_argument("--save", action="store_true", help="save file")
    parser.add_argument(
        "--dataset", default="sbd", type=str, choices=["coco", "sbd"], help="dataset"
    )
    parser.add_argument("--datalist", default="train", type=str, help="datalist")
    parser.add_argument("--gpu_id", default="0", type=str, help="gpu_id")
    parser.add_argument("--num_workers", default=0, type=int, help="num_workers")
    parser.add_argument("--batch_size", default=1, type=int, help="batch_size")
    parser.add_argument("--thresd_iou", default=0.0, type=float, help="thresd_iou")
    parser.add_argument("--bound_th", default=0.0, type=float, help="bound_th")
    parser.add_argument(
        "--method", default="svd", type=str, choices=["svd", "dpcp", "fms"]
    )
    parser.add_argument(
        "--val_U", action="store_true", help="use U trained on val data"
    )
    parser.add_argument(
        "--process_mode",
        default="ese_ori",
        type=str,
        choices=["ese_ori", "ese_box_center", "centroid", "hybrid", "inscribed_circle"],
    )

    args = parser.parse_args()

    return args
