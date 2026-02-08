import argparse
import os
import json
import torch
import numpy as np
from ours.utils import read_images, query_warp, project_warp, eval
from torchvision.utils import save_image
from recon.refiner import Refiner, Config
from omegaconf import OmegaConf
import argparse
from recon.trainer import save_depth_map_visualization
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms.functional import to_pil_image

def eval(cfg, load_step, eval_test=False, test_from_train=False):

    with open(os.path.join(cfg.base_dir, cfg.gs_cfg_file), "r") as f:
        config = Config(**json.load(f))
    refiner = Refiner(
        config, 
        load_step=load_step,
        test_split=cfg.test_split, 
        test_trans=cfg.test_trans, 
        test_rots=cfg.test_rots, 
        c_exp_index=cfg.c_exp_index,
        hessian_attr=cfg.hessian_attr,
        test_len = cfg.refine_end_idx - cfg.refine_start_idx,
        data_type=cfg.data_type,
    )

    output_dir = os.path.join(cfg.base_dir, cfg.exp_name)
    os.makedirs(f'{output_dir}/eval/{load_step}_test', exist_ok=True)
    os.makedirs(f'{output_dir}/eval/{load_step}_train', exist_ok=True)

    # eval test images
    test_eval_results = {
        "psnr": [],
        "ssim": [],
        "lpips": [],
    }
    # test_images = []
    for i in range(cfg.refine_start_idx, cfg.refine_end_idx):
        if test_from_train:
            rgb, _, _, _, _, eval_results = refiner.render(i, split='train', eval=True, trans=True)
        else:
            rgb, _, _, _, _, eval_results = refiner.render(i, split='test', eval=True, trans=True)
        if eval_test:
            test_eval_results["psnr"].append(eval_results["psnr"])
            test_eval_results["ssim"].append(eval_results["ssim"])
            test_eval_results["lpips"].append(eval_results["lpips"])
        save_image(rgb.permute(2,0,1), f'{output_dir}/eval/{load_step}_test/{i:03d}.jpg')
        
    #     fid_img = to_pil_image(rgb.permute(2,0,1))
    #     # fid_img = fid_img.resize((299, 299), Image.LANCZOS)
    #     fid_img = np.array(fid_img)
    #     test_images.append(torch.from_numpy(fid_img))
    # test_images = torch.stack(test_images).permute(0, 3, 1, 2).to(rgb.device)
    if eval_test:
        for k in test_eval_results:
            test_eval_results[k] = np.mean(test_eval_results[k])
        with open(f"{output_dir}/eval/{load_step}_test.json", "w") as f:
            json.dump(test_eval_results, f, indent=2)

    # eval train images
    train_eval_results = {
        "psnr": [],
        "ssim": [],
        "lpips": [],
    }
    # train_images = []
    for i in range(cfg.train_start_idx, cfg.train_end_idx):
        if test_from_train:
            rgb, _, _, _, _, eval_results = refiner.render(i, split='train', eval=True, trans=False)
        else:
            rgb, _, _, _, _, eval_results = refiner.render(i, split='train', eval=True, trans=False)
        train_eval_results["psnr"].append(eval_results["psnr"])
        train_eval_results["ssim"].append(eval_results["ssim"])
        train_eval_results["lpips"].append(eval_results["lpips"])
        save_image(rgb.permute(2,0,1), f'{output_dir}/eval/{load_step}_train/{i:03d}.jpg')

    #     fid_img = to_pil_image((refiner.train_dataset[i]['image'] / 255.).numpy())
    #     # fid_img = fid_img.resize((299, 299), Image.LANCZOS)
    #     fid_img = np.array(fid_img)
    #     train_images.append(torch.from_numpy(fid_img))
    # train_images = torch.stack(train_images).permute(0, 3, 1, 2).to(rgb.device)
    for k in train_eval_results:
        train_eval_results[k] = np.mean(train_eval_results[k])
    with open(f"{output_dir}/eval/{load_step}_train.json", "w") as f:
        json.dump(train_eval_results, f, indent=2)

    # fid = FrechetInceptionDistance(feature=64).to(refiner.device)
    # fid.reset()
    # fid.update(train_images, real=True)
    # fid.update(test_images, real=False)
    # fid_score = fid.compute()
    # with open(f"{output_dir}/eval/{load_step}_test_fid.json", "w") as f:
    #         json.dump({"fid": fid_score.item()}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_cfg', type=str, required=True, help='exp cfg path')
    parser.add_argument('--base_cfg', type=str, default="exp_cfg/base.yaml", help='base cfg path')
    parser.add_argument('--eval_test', action='store_true', help='eval test images')
    parser.add_argument('--test_from_train', action='store_true', help='test from train images')
    args = parser.parse_args()
    base_cfg = OmegaConf.load(args.base_cfg)
    exp_cfg = OmegaConf.load(args.exp_cfg)
    cfg = OmegaConf.merge(base_cfg, exp_cfg)
    eval(cfg, 29999, args.eval_test, args.test_from_train)
    eval(cfg, cfg.exp_name, args.eval_test, args.test_from_train)
