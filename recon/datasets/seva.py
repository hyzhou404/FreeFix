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
from depth_model import Metric3dModel, unproject_depth
from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


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


class SevaParser:
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

        # Load transforms.json
        with open(os.path.join(data_dir, "transforms_all.json"), "r") as f:
            self.meta_data = json.load(f)

        # Prepare depth model

        # Load infos
        self.indices = []
        self.camera_ids = []
        self.image_names = []
        self.image_paths = []
        self.camtoworlds = []
        self.Ks_dict = {}
        self.imsize_dict = {}
        self.mask_dict = {}
        self.params_dict = {}


        points_path = os.path.join(data_dir, "points.ply")
        if os.path.exists(points_path):
            pcd = o3d.io.read_point_cloud(points_path)
            self.points = np.asarray(pcd.points)
            self.points_rgb = np.asarray(pcd.colors)
        else:
            metric3d = Metric3dModel()
            self.points = []
            self.points_rgb = []

        index = 0
        for frame in tqdm(self.meta_data["frames"]):
            if frame['train'] == False:
                continue
            if not os.path.exists(os.path.join(self.data_dir, frame["file_path"])):
                continue
            self.indices.append(index)
            self.image_names.append(os.path.basename(frame["file_path"]))
            self.image_paths.append(os.path.join(self.data_dir, frame["file_path"]))
            c2w = np.eye(4)
            c2w[:3, :4] = np.array(frame["transform_matrix"])[:3, :4]
            # OpenGL to OpenCV
            c2w = c2w @ np.diag([1, -1, -1, 1])
            self.camtoworlds.append(c2w)
            self.mask_dict[index] = None

            # only consider one camera case for now
            self.camera_ids.append(0)
            fx, fy, cx, cy = frame["fl_x"], frame["fl_y"], frame["cx"], frame["cy"]
            if 0 not in self.Ks_dict:
                K = np.eye(3)
                K[0, 0] = fx
                K[1, 1] = fy
                K[0, 2] = cx
                K[1, 2] = cy
                self.Ks_dict[0] = K
                im = imageio.imread(os.path.join(self.data_dir, frame["file_path"]))
                self.imsize_dict[0] = (im.shape[1], im.shape[0])
                self.params_dict[0] = []

            # infer depth
            if not os.path.exists(points_path):
                intrinsic = [fx, fy, cx, cy]

                im, depth = metric3d.forward(self.image_paths[index], intrinsic)
                xyzs, rgbs = unproject_depth(im, depth, intrinsic, c2w, sample_per_frame=5000)
                if xyzs is not None:
                    self.points.append(xyzs)
                    self.points_rgb.append(rgbs)

            index += 1

        if not os.path.exists(points_path):
            self.points = np.concatenate(self.points, axis=0)
            self.points_rgb = np.concatenate(self.points_rgb, axis=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)
            pcd.colors = o3d.utility.Vector3dVector(self.points_rgb)
            o3d.io.write_point_cloud(points_path, pcd)


class SevaDataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: SevaParser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        elif split == 'val':
            self.indices = indices[indices % self.parser.test_every == 0]
        elif split == 'all':
            self.indices = indices
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id]
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]

        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        return data


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = SevaParser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = SevaDataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")
    import ipdb; ipdb.set_trace()

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()