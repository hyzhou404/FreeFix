import os
import json
from tqdm import tqdm
from typing import Any, Dict, List, Optional

import cv2
from PIL import Image
import imageio.v2 as imageio
import numpy as np
import torch
import open3d as o3d


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def _resize_image_folder(image_dir: str, resized_dir: str, factor: int) -> str:
    """Resize image folder."""
    print(f"Downscaling images by {factor}x from {image_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)

    image_files = _get_rel_paths(image_dir)
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        resized_path = os.path.join(
            resized_dir, os.path.splitext(image_file)[0] + ".png"
        )
        if os.path.isfile(resized_path):
            continue
        image = imageio.imread(image_path)[..., :3]
        resized_size = (
            int(round(image.shape[1] / factor)),
            int(round(image.shape[0] / factor)),
        )
        resized_image = np.array(
            Image.fromarray(image).resize(resized_size, Image.BICUBIC)
        )
        imageio.imwrite(resized_path, resized_image)
    return resized_dir


class Parser:
    """Seva parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 10,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.scene_scale = 10.0

        with open(os.path.join(data_dir, "meta_data.json"), "r") as f:
            self.meta_data = json.load(f)

        points_path = os.path.join(data_dir, "points3d.ply")
        pcd = o3d.io.read_point_cloud(points_path)
        self.points = np.asarray(pcd.points)
        self.points_rgb = np.asarray(pcd.colors)

        # Load infos
        self.image_names = []
        self.image_paths = []
        self.camtoworlds = []
        self.camera_ids = []
        self.imsize_dict = {}
        self.Ks_dict = {}

        index = 0
        for frame in tqdm(self.meta_data["frames"]):
            cam_name = os.path.dirname(frame["rgb_path"])
            self.image_names.append(os.path.basename(frame["rgb_path"]))
            self.image_paths.append(os.path.join(self.data_dir, frame["rgb_path"]))
            c2w = np.array(frame["camtoworld"]).astype(np.float32)
            # OpenGL to OpenCV
            # c2w = c2w @ torch.tensor(np.diag([1, -1, -1, 1])).float()
            self.camtoworlds.append(c2w)
            self.camera_ids.append(cam_name)
            self.Ks_dict[cam_name] = np.array(frame['intrinsics'])[:3, :3].astype(np.float32)
            self.imsize_dict[cam_name] = (frame['width'], frame['height'])

            index += 1

        self.camtoworlds = np.stack(self.camtoworlds)


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
        partition_file: str = "partition.json",
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        with open(partition_file, "r") as f:
            self.partition = json.load(f)
        self.indices = self.partition[split]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        cam_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[cam_id]
        camtoworlds = self.parser.camtoworlds[index]

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
            "image_path": self.parser.image_paths[index],
            "image_name": self.parser.image_names[index],
            "image_size": self.parser.imsize_dict[cam_id],
        }

        return data


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()