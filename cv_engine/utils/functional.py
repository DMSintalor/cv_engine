import os.path

from glob import glob
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from tensorboardX import SummaryWriter
from PIL import Image
from typing import Iterable


def prepare_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def prepare_tb(dir_path):
    return SummaryWriter(log_dir=dir_path)


def get_filename(file_path: str):
    return file_path.split(os.sep)[-1].split('.')[0]


def check_dir(args):
    save_dir = Path(args.save_dir)
    target_dir = save_dir / args.filename
    if os.path.exists(target_dir):
        dirs = glob(f'{target_dir}*')
        dirs = [int('0' + it.split('/')[-1].strip(args.filename).strip('_')) for it in dirs]
        dirs.sort()
        args.filename = f'{args.filename}_{dirs[-1] + 1}'
    target_dir = save_dir / args.filename
    prepare_dir(target_dir)
    return target_dir


def load_image(path: str):
    img = Image.open(path)
    img = img.convert("RGB")
    img = TF.resize(img, (224, 224))
    img = TF.to_tensor(img)
    img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)
    return img


def resume_img(
        img: torch.Tensor,
        dataset_mean: Iterable[float],
        dataset_std: Iterable[float],
):
    _img = list()
    for channel_id, (s, m) in enumerate(zip(dataset_std, dataset_mean)):
        channel = img[channel_id] * s + m
        _img.append(channel)
    img = torch.stack(_img, dim=0)
    img.clamp_(0, 1)
    img = TF.to_pil_image(img)
    return img
