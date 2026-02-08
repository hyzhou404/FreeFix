import argparse
import os
import json
import torch
import numpy as np
import imageio
from ours.utils import align_depth, splat_pixels
from ours.pipelines.flux_pipeline import FluxPipeline
from ours.schedulers.flow_match_euler_discrete_scheduler import FlowMatchEulerDiscreteScheduler
from torchvision.utils import save_image
import re
import fire
from recon.refiner import Refiner, Config
from omegaconf import OmegaConf
import argparse
from recon.trainer import save_depth_map_visualization

def refine(cfg):
    with open(os.path.join(cfg.base_dir, cfg.gs_cfg_file), "r") as f:
        config = Config(**json.load(f))
    refiner = Refiner(
        config, 
        load_step=cfg.load_step,
        test_split=cfg.test_split, 
        test_trans=cfg.test_trans, 
        test_rots=cfg.test_rots, 
        c_exp_index=cfg.c_exp_index,
        hessian_attr=cfg.hessian_attr,
        test_len = cfg.refine_end_idx - cfg.refine_start_idx,
        data_type=cfg.data_type,
    )

    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

    output_dir = os.path.join(cfg.base_dir, cfg.exp_name)
    os.makedirs(f'{output_dir}/before_refine', exist_ok=True)
    os.makedirs(f'{output_dir}/after_refine', exist_ok=True)
    os.makedirs(f'{output_dir}/refine/render', exist_ok=True)
    os.makedirs(f'{output_dir}/refine/gen', exist_ok=True)
    os.makedirs(f'{output_dir}/refine/depth', exist_ok=True)
    for c_exp in cfg.c_exp_index:
        os.makedirs(f'{output_dir}/refine/masks/{c_exp}', exist_ok=True)
    before_refine_writer = imageio.get_writer(f'{output_dir}/before_refine.mp4', fps=12)
    gen_writer = imageio.get_writer(f'{output_dir}/refine/gen.mp4', fps=12)
    after_refine_writer = imageio.get_writer(f'{output_dir}/after_refine.mp4', fps=12)

    generator = torch.manual_seed(64)
    infer_steps = int(cfg.num_inference_steps * cfg.strength)
    mask_scheduler=[int(infer_steps * cfg.c_scheduler[i]) for i in range(len(cfg.c_scheduler))]

    # render test images before refine
    for i in range(cfg.refine_start_idx, cfg.refine_end_idx):
        rgb, _, _, _, _, _ = refiner.render(i)
        save_image(rgb.permute(2,0,1), f'{output_dir}/before_refine/{i:03d}.jpg')
        before_refine_writer.append_data((rgb.detach().cpu().numpy() * 255).astype(np.uint8))
    before_refine_writer.close()

    # refine
    train_cams = [refiner.train_dataset[j] for j in range(cfg.train_start_idx, cfg.train_end_idx)]
    train_prob = [1 for _ in range(cfg.train_start_idx, cfg.train_end_idx)]
    prev_refine_cams = []
    for i in range(cfg.refine_start_idx, cfg.refine_end_idx):
        rgb, masks, alpha, depth, cam_param, _ = refiner.render(i)
        masks = torch.stack(masks)
        rgb_to_refine = rgb.permute(2,0,1).to(pipe.device) # (3, H, W)
        masks = masks.to(pipe.device)
        H, W = rgb_to_refine.shape[1], rgb_to_refine.shape[2]

        save_image(rgb_to_refine, f'{output_dir}/refine/render/{i:03d}.jpg')
        save_depth_map_visualization(depth[..., 0].cpu().numpy(), f'{output_dir}/refine/depth/{i:03d}.jpg')
        for j in range(masks.shape[0]):
            save_image(masks[j:j+1][None, ...], f'{output_dir}/refine/masks/{cfg.c_exp_index[j]}/{i:03d}.jpg')
        
        if i == cfg.refine_start_idx:
            warp_until = -1
            warp_mask = None
            refine_steps = cfg.refine_steps * 2
        else:
            warp_until = infer_steps*cfg.warp_ratio
            warp_mask = alpha
            refine_steps = cfg.refine_steps

        refined_image = pipe(
            cfg.prompt,
            negative_prompt=cfg.negative_prompt if "negative_prompt" in cfg else None,
            image=rgb_to_refine,
            mask=masks,
            mask_scheduler=mask_scheduler,
            guide_until=infer_steps*cfg.guide_ratio,
            warp_image=rgb_to_refine,
            warp_until=warp_until,
            warp_mask=warp_mask,
            height=H,
            width=W,
            guidance_scale=3.5,
            num_inference_steps=cfg.num_inference_steps,
            generator=generator,
            strength=cfg.strength,
        ).images[0]

        refined_image = refined_image.resize((W, H))
        torch_refined_image = torch.from_numpy(np.array(refined_image))
        ixt = cam_param["K"]
        c2w = cam_param["c2w"]
        refine_cams = [{
            "image": torch_refined_image,
            "camtoworld": c2w,
            "K": ixt,
            "Gen": True,
            "image_id": f"gen_{i - cfg.refine_start_idx}",
        }]

        refined_image.save(f'{output_dir}/refine/gen/image_{i:03d}.jpg')
        gen_writer.append_data(np.array(refined_image))

        refiner.refine(refine_cams, train_cams, train_prob, max_steps=refine_steps, use_affine=cfg.affine)

        train_cams.append(refine_cams[0])
        train_prob.append(cfg.gen_prob)

    gen_writer.close()

    # render test images after refine
    for i in range(cfg.refine_start_idx, cfg.refine_end_idx):
        rgb, _, _, _, _, _ = refiner.render(i)
        save_image(rgb.permute(2,0,1), f'{output_dir}/after_refine/{i:03d}.jpg')
        after_refine_writer.append_data((rgb.detach().cpu().numpy() * 255).astype(np.uint8))
    after_refine_writer.close()

    refiner.save(name=f"ckpt_{cfg.exp_name}")   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_cfg', type=str, required=True, help='exp cfg path')
    parser.add_argument('--base_cfg', type=str, default="exp_cfg/base.yaml", help='base cfg path')
    args = parser.parse_args()
    base_cfg = OmegaConf.load(args.base_cfg)
    exp_cfg = OmegaConf.load(args.exp_cfg)
    cfg = OmegaConf.merge(base_cfg, exp_cfg)
    refine(cfg)
