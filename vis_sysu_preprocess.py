from __future__ import absolute_import

import argparse
import os
import os.path as osp
import re
import shutil


PID_CAM_RE = re.compile(r"([-\d]+)_c(\d)")
PIXEL_MEAN = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]


def _scan_prepared_records(data_dir, modal):
    if modal == "rgb":
        image_dir = osp.join(data_dir, "rgb_modify", "bounding_box_train")
    else:
        image_dir = osp.join(data_dir, "ir_modify", "bounding_box_train")

    if not osp.isdir(image_dir):
        raise RuntimeError("SYSU prepared folder is not available: {}".format(image_dir))

    records = []
    for name in sorted(os.listdir(image_dir)):
        path = osp.join(image_dir, name)
        if not osp.isfile(path) or osp.splitext(name)[1].lower() != ".jpg":
            continue
        match = PID_CAM_RE.search(name)
        if match is None:
            continue
        pid = int(match.group(1))
        camid = int(match.group(2))
        if pid == -1:
            continue
        records.append((path, pid, camid))

    if not records:
        raise RuntimeError("No SYSU prepared images found in {}".format(image_dir))
    return records


def _tokenmatcher_stage1_transforms(height, width):
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode

    from ChannelAug import ChannelAdapGray, ChannelExchange, ChannelRandomErasing

    normalizer = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)
    rgb_view1 = T.Compose([
        T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        normalizer,
        ChannelRandomErasing(probability=0.5),
    ])
    rgb_view2 = T.Compose([
        T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        T.ToTensor(),
        normalizer,
        ChannelRandomErasing(probability=0.5),
        ChannelExchange(gray=2),
    ])
    ir = T.Compose([
        T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        ChannelRandomErasing(probability=0.5),
        ChannelAdapGray(probability=0.5),
    ])
    return {"rgb": (rgb_view1, rgb_view2), "ir": ir}


def _tokenmatcher_stage2_transforms(height, width):
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode

    from ChannelAug import ChannelAdapGray, ChannelExchange, ChannelRandomErasing

    color_aug = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    normalizer = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)
    rgb_view1 = T.Compose([
        color_aug,
        T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        normalizer,
        ChannelRandomErasing(probability=0.5),
    ])
    rgb_view2 = T.Compose([
        color_aug,
        T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        normalizer,
        ChannelRandomErasing(probability=0.5),
        ChannelExchange(gray=2),
    ])
    ir = T.Compose([
        color_aug,
        T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        ChannelRandomErasing(probability=0.5),
        ChannelAdapGray(probability=0.5),
    ])
    return {"rgb": (rgb_view1, rgb_view2), "ir": ir}


def _remove_existing(output_dir, project, stage, modal, vis_id):
    pid_name = "pid{:04d}".format(int(vis_id))
    base_dir = osp.join(output_dir, project, stage, modal, pid_name)
    if osp.isdir(base_dir):
        shutil.rmtree(base_dir)


def _normalize_modal(modal):
    if modal == "thermal":
        return "ir"
    return modal


def _modal_list(modal):
    modal = _normalize_modal(modal)
    if modal == "both":
        return ["rgb", "ir"]
    return [modal]


def _stage_list(stage):
    if stage == "both":
        return ["stage1", "stage2"]
    return [stage]


def _default_output_dir():
    return osp.join(osp.dirname(osp.abspath(__file__)), "logs", "vis_only")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save TokenMatcher SYSU preprocessing visualizations without training."
    )
    parser.add_argument("--data-dir", required=True, help="SYSU-MM01 root containing rgb_modify/ir_modify")
    parser.add_argument("--vis-id", "--vis_id", dest="vis_id", type=int, required=True)
    parser.add_argument("--vis-camid", "--vis_camid", dest="vis_camid", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", choices=["stage1", "stage2", "both"], default="both")
    parser.add_argument("--modal", choices=["rgb", "ir", "thermal", "both"], default="rgb")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--output-dir", default=_default_output_dir())
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    from preprocess_vis import save_fixed_pid_preprocess_visualization

    stages = _stage_list(args.stage)
    modals = _modal_list(args.modal)
    stage_transforms = {
        "stage1": _tokenmatcher_stage1_transforms(args.height, args.width),
        "stage2": _tokenmatcher_stage2_transforms(args.height, args.width),
    }

    for stage in stages:
        for modal in modals:
            records = _scan_prepared_records(args.data_dir, modal)
            if args.overwrite:
                _remove_existing(args.output_dir, "TokenMatcher", stage, modal, args.vis_id)
            result = save_fixed_pid_preprocess_visualization(
                records,
                stage_transforms[stage][modal],
                save_dir=args.output_dir,
                project="TokenMatcher",
                stage=stage,
                modal=modal,
                target_id=args.vis_id,
                target_camid=args.vis_camid,
                root=None,
                seed=args.seed,
                mean=PIXEL_MEAN,
                std=PIXEL_STD,
            )
            if result is None:
                print(
                    "[vis] skipped TokenMatcher {} {} vis_id={} camid={}".format(
                        stage, modal, args.vis_id, args.vis_camid
                    )
                )


if __name__ == "__main__":
    main()
