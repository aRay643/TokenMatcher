from __future__ import annotations

import os
import os.path as osp
import random
import re
from contextlib import contextmanager
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont


PID_CAM_RE = re.compile(r"(?P<pid>-?\d+)_c(?P<cam>\d+)")
LEADING_PID_RE = re.compile(r"(?P<pid>-?\d+)")


@contextmanager
def _seeded_rng(seed: Optional[int]):
    if seed is None:
        yield
        return

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _safe_component(value) -> str:
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._-") or "vis"


def _resolve_path(path: str, root: Optional[str]) -> str:
    if osp.isabs(path) or root is None:
        return path
    return osp.join(root, path)


def _extract_pid_cam_from_name(path: str) -> Tuple[Optional[int], Optional[int]]:
    name = osp.basename(str(path))
    match = PID_CAM_RE.search(name)
    if match is not None:
        return int(match.group("pid")), int(match.group("cam"))

    match = LEADING_PID_RE.match(name)
    if match is not None:
        return int(match.group("pid")), None

    return None, None


def _record_meta(record):
    if isinstance(record, (list, tuple)) and record:
        path = str(record[0])
        pid, cam = _extract_pid_cam_from_name(path)
        if pid is None and len(record) >= 2 and isinstance(record[1], (int, np.integer)):
            pid = int(record[1])
        if cam is None and len(record) >= 3 and isinstance(record[2], (int, np.integer)):
            cam = int(record[2])
        return path, pid, cam

    path = str(record)
    pid, cam = _extract_pid_cam_from_name(path)
    return path, pid, cam


def _select_record(
    records: Sequence,
    target_pid: int,
    target_camid: Optional[int] = None,
):
    candidates = []
    for record in records:
        _, raw_pid, raw_camid = _record_meta(record)
        if raw_pid != target_pid:
            continue
        if target_camid is not None and raw_camid != target_camid:
            continue
        candidates.append(record)

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda item: _record_meta(item)[0])
    return candidates[0]


def _to_pil_image(img, mean, std):
    if isinstance(img, Image.Image):
        return img.convert("RGB")

    if torch.is_tensor(img):
        tensor = img.detach().cpu().float()
        if tensor.dim() == 4:
            tensor = tensor[0]
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.size(0) >= 3:
            mean_t = torch.tensor(mean[: tensor.size(0)], dtype=tensor.dtype).view(-1, 1, 1)
            std_t = torch.tensor(std[: tensor.size(0)], dtype=tensor.dtype).view(-1, 1, 1)
            tensor = tensor * std_t + mean_t
        tensor = tensor.clamp(0.0, 1.0)
        return TF.to_pil_image(tensor)

    raise TypeError("Unsupported image type for visualization: {}".format(type(img)))


def _apply_transforms(orig_img: Image.Image, transform, seed: Optional[int]):
    if isinstance(transform, (list, tuple)):
        transformed = []
        for idx, sub_transform in enumerate(transform):
            with _seeded_rng(None if seed is None else seed + idx):
                transformed.append(sub_transform(orig_img.copy()))
        return transformed

    with _seeded_rng(seed):
        return transform(orig_img.copy())


def _panel_with_title(img: Image.Image, title: str, width: int, height: int, header_h: int = 24):
    canvas = Image.new("RGB", (width, height + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((6, 6), title, fill="black", font=font)
    resized = img.convert("RGB").resize((width, height), Image.BICUBIC)
    canvas.paste(resized, (0, header_h))
    return canvas


def _save_image(img: Image.Image, path: str):
    os.makedirs(osp.dirname(path), exist_ok=True)
    img.save(path)


def _try_mark_done(marker_path: str) -> bool:
    os.makedirs(osp.dirname(marker_path), exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(marker_path, flags)
    except FileExistsError:
        return False
    else:
        os.close(fd)
        return True


def save_fixed_pid_preprocess_visualization(
    records: Sequence,
    transform,
    save_dir: str,
    project: str,
    stage: str,
    modal: str,
    target_pid: int,
    target_camid: Optional[int] = None,
    root: Optional[str] = None,
    seed: Optional[int] = 0,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
):
    """Save the raw and transformed views for one fixed pid."""
    if target_pid is None or int(target_pid) <= 0:
        return None

    record = _select_record(records, int(target_pid), target_camid)
    if record is None:
        print(
            "[vis] {} {} {}: no sample found for pid={} camid={}".format(
                project, stage, modal, target_pid, target_camid
            )
        )
        return None

    path, raw_pid, raw_camid = _record_meta(record)
    abs_path = _resolve_path(path, root)
    raw_img = Image.open(abs_path).convert("RGB")
    transformed = _apply_transforms(raw_img, transform, seed)

    if isinstance(transformed, (list, tuple)):
        transformed_imgs = [_to_pil_image(img, mean, std) for img in transformed]
    else:
        transformed_imgs = [_to_pil_image(transformed, mean, std)]

    project_name = _safe_component(project)
    stage_name = _safe_component(stage)
    modal_name = _safe_component(modal)
    pid_name = "pid{:04d}".format(int(raw_pid) if raw_pid is not None and raw_pid >= 0 else int(target_pid))
    cam_name = "cam{}".format(raw_camid) if raw_camid is not None else "camx"
    base_dir = osp.join(save_dir, project_name, stage_name, modal_name, pid_name)
    os.makedirs(base_dir, exist_ok=True)
    marker_path = osp.join(base_dir, ".done")
    if not _try_mark_done(marker_path):
        return None

    raw_path = osp.join(base_dir, "{}_{}_{}_raw.png".format(project_name, stage_name, modal_name))
    _save_image(raw_img, raw_path)

    transformed_paths = []
    for idx, img in enumerate(transformed_imgs):
        view_path = osp.join(
            base_dir,
            "{}_{}_{}_view{:02d}.png".format(project_name, stage_name, modal_name, idx + 1),
        )
        _save_image(img, view_path)
        transformed_paths.append(view_path)

    display_width = transformed_imgs[0].width
    display_height = transformed_imgs[0].height
    panels = [
        _panel_with_title(raw_img, "original", display_width, display_height),
    ]
    for idx, img in enumerate(transformed_imgs):
        panels.append(_panel_with_title(img, "view{}".format(idx + 1), display_width, display_height))

    gap = 8
    total_width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    total_height = max(panel.height for panel in panels)
    grid = Image.new("RGB", (total_width, total_height), "white")
    x = 0
    for panel in panels:
        grid.paste(panel, (x, 0))
        x += panel.width + gap

    grid_path = osp.join(
        base_dir,
        "{}_{}_{}_comparison_{}_{}.png".format(project_name, stage_name, modal_name, pid_name, cam_name),
    )
    _save_image(grid, grid_path)

    print("[vis] saved {}".format(grid_path))
    return {
        "record_path": abs_path,
        "raw_path": raw_path,
        "transformed_paths": transformed_paths,
        "grid_path": grid_path,
    }
