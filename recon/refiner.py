import json
import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torch.nn.functional as F
from recon import nerfview
import viser
import tqdm
import numpy as np
import roma

from recon.trainer import Config, soft_sigmoid

from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy
from einops import reduce
from ours.utils import neighbor_L1_loss

class Refiner:
    def __init__(
        self, 
        cfg: Config, 
        load_step=29999, 
        test_split="test",
        test_trans=[0, 0, 0],
        test_rots=[0, 0, 0],
        c_exp_index=[0.001, 0.01, 0.1],
        hessian_attr=["mean"],
        test_len=10,
        data_type="colmap",
    ):
        self.cfg = cfg
        self.device = "cuda"

        if data_type == "colmap":
            from recon.datasets.colmap import Dataset, Parser
        elif data_type == "hugsim":
            from recon.datasets.hugsim import Dataset, Parser
        else:
            raise NotImplementedError

        # Load the original dataset 
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=True,
            test_every=cfg.test_every,
        )
        self.train_dataset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
            partition_file=cfg.partition
        )
        self.test_dataset = Dataset(
            self.parser,
            split=test_split,
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
            partition_file=cfg.partition
        )
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale

        self.hessian_attr = hessian_attr
        self.c_exp_index = c_exp_index
        self.test_trans = test_trans
        self.test_rots = test_rots

        self.total_step = 0

        # Load the refine dataset
        # self.refine_dataset = Refine_Dataset(os.path.join(cfg.result_dir, "to_refine"))

        # Load the pre-optimized gaussian splats
        ckpt_path = os.path.join(cfg.result_dir, "ckpts", f"ckpt_{load_step}.pt")
        self.splats = torch.nn.ParameterDict(torch.load(ckpt_path, map_location=self.device)["splats"])

        affines = {}
        for i in range(test_len):
            affines[f"gen_{i}"] = torch.nn.Parameter(torch.eye(4)[:3, :].to(self.device)) # 3x4
        self.affines = torch.nn.ParameterDict(affines)

        # init the optimizers
        self._init_optimizer()

        # Densification Strategy
        self.strategy = DefaultStrategy(
            verbose=True,
            # scene_scale=self.scene_scale,
            prune_opa=cfg.prune_opa,
            grow_grad2d=cfg.grow_grad2d,
            grow_scale3d=cfg.grow_scale3d,
            prune_scale3d=cfg.prune_scale3d,
            refine_start_iter=100,
            refine_stop_iter=5000,
            reset_every=1500,
            refine_every=200,
            absgrad=cfg.absgrad,
            revised_opacity=cfg.revised_opacity,
        )
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state()

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            # c2ws = np.concatenate([self.parser.camtoworlds, self.refine_dataset.interp_c2ws], axis=0)
            # Ks = [self.parser.Ks_dict[camera_id].copy() for camera_id in self.parser.camera_ids] + \
            #      [self.refine_dataset.closest_K]*len(self.refine_dataset)
            # img_whs = [self.parser.imsize_dict[camera_id] for camera_id in self.parser.camera_ids] + \
            #           [self.refine_dataset.img_wh]*len(self.refine_dataset)
            c2ws = self.parser.camtoworlds
            Ks = [self.parser.Ks_dict[camera_id].copy() for camera_id in self.parser.camera_ids]
            img_whs = [self.parser.imsize_dict[camera_id] for camera_id in self.parser.camera_ids]
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="refining",
                c2ws=c2ws,
                Ks=Ks,
                img_whs=img_whs,
                scene_scale=self.scene_scale,
                result_dir=cfg.result_dir,
            )

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb_refine")

    def _init_optimizer(self):
        for param in self.splats.values():
            param.requires_grad = True

        self.optimizers = {
            'means': torch.optim.Adam([{"params": self.splats['means'], "lr": 1e-4 * self.scene_scale}]),
            'scales': torch.optim.Adam([{"params": self.splats['scales'], "lr": 5e-3}]),
            'quats': torch.optim.Adam([{"params": self.splats['quats'], "lr": 1e-3}]),
            'opacities': torch.optim.Adam([{"params": self.splats['opacities'], "lr": 5e-2}]),
            'sh0': torch.optim.Adam([{"params": self.splats['sh0'], "lr": 2.5e-3}]),
            'shN': torch.optim.Adam([{"params": self.splats['shN'], "lr": 2.5e-3/20}]),
        }

        self.affine_optimizers = {
            f"gen_{i}": torch.optim.Adam([{"params": self.affines[f"gen_{i}"], "lr": 1e-2}])
            for i in range(len(self.affines))
        }

    def add_splats(self, new_splats):
        for name, add_params in new_splats.items():
            param = self.splats[name]
            new_param = torch.nn.Parameter(torch.cat([param, add_params], dim=0), requires_grad=True)
            self.splats[name] = new_param
            optimizer = self.optimizers[name]
            for i in range(len(optimizer.param_groups)):
                param_state = optimizer.state[param]
                del optimizer.state[param]
                for key in param_state.keys():
                    if key != "step":
                        v = param_state[key]
                        param_state[key] = torch.cat([v, torch.zeros((len(add_params), *v.shape[1:]), device=v.device)])
                optimizer.param_groups[i]["params"] = [new_param]
                optimizer.state[new_param] = param_state
            
        for k, v in self.strategy_state.items():
            if isinstance(v, torch.Tensor):
                self.strategy_state[k] = torch.cat((v, torch.zeros((len(add_params), *v.shape[1:]), device=v.device)))

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        override_color: Tensor=None,
        affine: Tensor=None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors if override_color is None else override_color,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=self.cfg.absgrad,
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            **kwargs,
        )
        if affine is not None:
            render_colors = render_colors @ affine[:3, :3] + affine[:3, 3]
        return render_colors, render_alphas, info
    
    def rasterize_splats_w_certainty(
            self, 
            camtoworlds: Tensor, 
            Ks: Tensor, 
            width: int, 
            height: int
        ):
        rgbs, alphas, _ = self.rasterize_splats(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=self.cfg.sh_degree,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane, 
            render_mode="RGB+ED",
        ) 
        depths = rgbs[..., 3:4].detach()[0]
        colors = torch.clamp(rgbs[..., :3], 0.0, 1.0).detach()[0]
        alphas = alphas.detach()[0, ..., 0]
        
        # render uncertainty
        rgbs[..., :3].backward(gradient=torch.ones_like(rgbs[..., :3]))
        H_per_gaussian = [self.splats[k].grad.detach() ** 2 for k in self.hessian_attr]
        # H_per_gaussian = [self.splats[k].grad.detach() ** 2 for k in ["means", "quats", "scales"]]
        H_per_gaussian = torch.cat(H_per_gaussian, dim=-1)
        self.splats['means'].grad = None
        self.splats['quats'].grad = None
        self.splats['scales'].grad = None
        self.splats['opacities'].grad = None
        self.splats['sh0'].grad = None
        self.splats['shN'].grad = None
        multi_certainties = []
        for exp_index in self.c_exp_index:
            inv_H_gaussian = torch.exp(-exp_index * H_per_gaussian)
            certainties, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                override_color=inv_H_gaussian,
                sh_degree=None,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
            )  # [1, H, W, 3]
            certainties = certainties[0].detach()
            certainties = reduce(certainties, "h w c -> h w", "mean").detach()
            certainties = (alphas * certainties).clamp(0,1)
            certainties = soft_sigmoid(certainties - 0.5, soft=10.0)
            multi_certainties.append(certainties)
        return colors, multi_certainties, alphas, depths
    
    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, render_alphas, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy(), render_alphas.squeeze().cpu().numpy()
    
    def render(self, idx, split="test", eval=False, trans=True):
        device = self.device
        if split == "test":
            data = self.test_dataset[idx]
        elif split == "train":
            data = self.train_dataset[idx]
        else:
            raise ValueError
        if trans:
            mat = torch.eye(4)
            mat[:3, :3] = roma.euler_to_rotmat('xyz', torch.tensor(self.test_rots).float(), degrees=True)
            mat[:3, 3] = torch.tensor(self.test_trans).float()
            c2w = data["camtoworld"].float() @ mat
        else:
            c2w = data["camtoworld"].float()
        c2w = c2w[None, ...].to(device)
        Ks = data["K"][None, ...].to(device)
        height, width = data["image"].shape[:2]
        colors, multi_certainties, alphas, depths = self.rasterize_splats_w_certainty(
            camtoworlds=c2w,
            Ks=Ks,
            width=width,
            height=height,
        )
        cam_param = {
            "c2w": c2w[0],
            "K": Ks[0],
        }

        eval_results = None
        if eval:
            psnr = self.psnr(colors, data["image"].to(device) / 255.0).item()
            ssim = self.ssim(colors.permute(2,0,1)[None, ...], data["image"].permute(2,0,1)[None, ...].to(device) / 255.0).item()
            lpips = self.lpips(colors.permute(2,0,1)[None, ...], data["image"].permute(2,0,1)[None, ...].to(device) / 255.0).item()
            eval_results = {
                'psnr': psnr,
                'ssim': ssim,
                'lpips': lpips,
            }

        # #debug
        # torch.save(colors.detach().cpu(), f"dbg/evaluation/renders_{idx}.pt")
        # torch.save(data["image"] / 255. , f"dbg/evaluation/gts_{idx}.pt")

        return colors, multi_certainties, alphas, depths, cam_param, eval_results
            

    def refine(self, refine_cams, train_cams, train_prob, max_steps=100, gen_loss_weight=0.2, use_affine=True):
        cfg = self.cfg
        device = self.device
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]

        # Training loop.
        densification = True
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:

            if step<=max_steps*1/3:
                is_refine_step = step % 3 == 1
            elif step<=max_steps*2/3:
                is_refine_step = step % 5 == 1
            else:
                is_refine_step = step % 8 == 1

            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            # get data
            affine = None
            if is_refine_step:
                idx = np.random.randint(0, len(refine_cams))
                data = refine_cams[idx]
            else:
                # idx = np.random.randint(0, len(train_cams))
                normalized_prob = train_prob / np.sum(train_prob)
                idx = np.random.choice(len(train_cams), 1, p=normalized_prob).item()
                data = train_cams[idx]
            if data.get('Gen', False):
                affine = self.affines[data['image_id']]

            camtoworlds = data["camtoworld"][None, ...].to(device)  # [1, 4, 4]
            Ks = data["K"][None, ...].to(device)  # [1, 3, 3]
            pixels = data["image"][None, ...].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )

            height, width = pixels.shape[1:3]
            
            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB",
                affine=affine if use_affine else None,
            )
            colors = renders.clip(0,1)

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            if densification:
                self.strategy.step_pre_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                )

            # loss
            ssimloss = 1.0 - self.ssim(
                pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
            )
            if data.get('Gen', False):
                # loss = l1loss * 0.2 + ssimloss * 0.4 + lpips_loss * 0.4
                l1loss = F.l1_loss(colors, pixels)
                # l1loss = neighbor_L1_loss(colors, pixels)
                # loss = ssimloss * 0.1 + l1loss * 0.2
                loss = l1loss * gen_loss_weight
            else:
                l1loss = F.l1_loss(colors, pixels)
                loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            loss.backward()

            if densification:
                self.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                )

            # logging
            desc = f"loss={loss.item():.3f}"
            pbar.set_description(desc)
            if cfg.tb_every > 0 and self.total_step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), self.total_step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), self.total_step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), self.total_step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), self.total_step)
                self.writer.add_scalar("train/mem", mem, self.total_step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, self.total_step)
                self.writer.flush()

            self.total_step += 1

            # optimize
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if affine is not None:
                self.affine_optimizers[f"{data['image_id']}"].step()
                self.affine_optimizers[f"{data['image_id']}"].zero_grad(set_to_none=True)

            for scheduler in schedulers:
                scheduler.step()


            # update viewer
            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(self.total_step, num_train_rays_per_step)

    def save(self, name="ckpt_refined"):
        # Save checkpoint
        torch.save(
            {
                "step": -1,
                "splats": self.splats.state_dict(),
            },
            f"{self.cfg.result_dir}/ckpts/{name}.pt",
        )


    @ torch.no_grad()
    def render_refined_video(self):
        pass
        # frames = []
        # for idx in range(len(self.refine_dataset.all_image_paths)):
        #     camtoworlds = torch.from_numpy(self.refine_dataset.all_interp_c2ws[idx]).float()[None,...].to(self.device)
        #     Ks = torch.from_numpy(self.refine_dataset.closest_K).float()[None,...].to(self.device)
        #     img_wh = self.refine_dataset.img_wh

        #     colors, alphas, info = self.rasterize_splats(
        #         camtoworlds=camtoworlds,
        #         Ks=Ks,
        #         width=img_wh[0],
        #         height=img_wh[1],
        #         sh_degree=self.cfg.sh_degree,
        #         near_plane=self.cfg.near_plane,
        #         far_plane=self.cfg.far_plane,
        #         image_ids=idx,
        #         render_mode="RGB",
        #     )
        #     frames.append(np.clip(colors[0].cpu().numpy(),0,1))

        # save_dir = f"{self.cfg.result_dir}/to_refine"
        # writer = imageio.get_writer(f"{save_dir}/refined_gs_render.mp4", fps=6)
        # for frame in frames:
        #     writer.append_data((frame*255).astype(np.uint8))
        # writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="the path to the config file"
                        ,default='results/bike/cfg.json')
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        config = Config(**json.load(f))
    
    refiner = Refiner(config)
    refiner.train()

    if not config.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)

    
