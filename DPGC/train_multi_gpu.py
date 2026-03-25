import cv2
import os
import argparse
import numpy as np
from collections import OrderedDict
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from dpvo.data_readers.factory import dataset_factory
from dpvo.lietorch import SE3
from dpvo.logger import Logger
import torch.nn.functional as F
from dpvo.net import VONet
from evaluate_tartan import evaluate as validate
from tqdm import tqdm

####multigpu
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

####multigpu
def setup(rank, i):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

####multigpu
def cleanup():
    dist.destroy_process_group()

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def image2gray(image):
    image = image.mean(dim=0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def kabsch_umeyama(A, B):
    n, m = A.shape
    EA = torch.mean(A, axis=0)
    EB = torch.mean(B, axis=0)
    VarA = torch.mean((A - EA).norm(dim=1)**2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = torch.svd(H)

    c = VarA / torch.trace(torch.diag(D))
    return c

def train(args):
    """ main training loop """

    # legacy ddp code
    # rank = 0
    
    ####multigpu
    rank = int(os.environ["LOCAL_RANK"])

    db = dataset_factory(['tartan'], datapath="datasets/TartanAir", n_frames=args.n_frames)
    
    ####multigpu
    train_sampler = DistributedSampler(db, num_replicas=args.world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        db,               # 训练数据集
        batch_size=1,               # 每个GPU的批次大小
        sampler=train_sampler,       # 使用 DistributedSampler 来自动分配数据
        num_workers=2,               # 数据加载的工作进程数
        shuffle=False,
        pin_memory=True
    )

    # train_loader = DataLoader(db, batch_size=1, shuffle=True, num_workers=8)

    net = VONet().cuda(rank)
    # net = DDP(net, device_ids=[rank], output_device=rank)
    net = DDP(net, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    # net = VONet()
    net.train()
    # net.cuda()

    if args.ckpt is not None:
        
        # state_dict = torch.load(args.ckpt)
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     new_state_dict[k.replace('module.', '')] = v
        # net.load_state_dict(new_state_dict, strict=False)
        
        #改后的多卡checkpoint导入
        print(f"Loading checkpoint from {args.ckpt} ...")
        state_dict = torch.load(args.ckpt, map_location='cuda')
        net.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully.")
        
        
        # state_dict = torch.load(args.ckpt)
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     new_state_dict[k.replace('module.', '')] = v
        # missing_keys, unexpected_keys = net.load_state_dict(new_state_dict, strict=False)

        # print(f"Missing keys: {missing_keys}")
        # print(f"Unexpected keys: {unexpected_keys}")

        # if not missing_keys:
        #     print("Checkpoint loaded successfully! Training will resume from checkpoint.")
        # else:
        #     print("Checkpoint loaded, but some keys are missing. Verify the model architecture.")

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-6)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
        args.lr, args.steps, pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    if rank == 0:
        logger = Logger(args.name, scheduler)

    total_steps = 0

    # Add a progress bar for the entire training loop
    for epoch in range(args.steps):  # You might want to track epochs too, not just steps
        # Use tqdm to track progress
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{args.steps}') as pbar:
            for data_blob in train_loader:
                images, poses, disps, intrinsics = [x.cuda().float() for x in data_blob]
                optimizer.zero_grad()

                # fix poses to gt for first 1k steps
                so = total_steps < 1000 and args.ckpt is None

                poses = SE3(poses).inv()
                traj = net(images, poses, disps, intrinsics, M=1024, STEPS=18, structure_only=so)

                loss = 0.0
                for i, (v, x, y, P1, P2, kl) in enumerate(traj):
                    e = (x - y).norm(dim=-1)
                    e = e.reshape(-1, net.module.P**2)[(v > 0.5).reshape(-1)].min(dim=-1).values

                    N = P1.shape[1]
                    ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
                    ii = ii.reshape(-1).cuda()
                    jj = jj.reshape(-1).cuda()

                    k = ii != jj
                    ii = ii[k]
                    jj = jj[k]

                    P1 = P1.inv()
                    P2 = P2.inv()

                    t1 = P1.matrix()[...,:3,3]
                    t2 = P2.matrix()[...,:3,3]

                    s = kabsch_umeyama(t2[0], t1[0]).detach().clamp(max=10.0)
                    P1 = P1.scale(s.view(1, 1))

                    dP = P1[:,ii].inv() * P1[:,jj]
                    dG = P2[:,ii].inv() * P2[:,jj]

                    e1 = (dP * dG.inv()).log()
                    tr = e1[...,0:3].norm(dim=-1)
                    ro = e1[...,3:6].norm(dim=-1)

                    loss += args.flow_weight * e.mean()
                    if not so and i >= 2:
                        loss += args.pose_weight * ( tr.mean() + ro.mean() )

                # kl is 0 (not longer used)
                loss += kl
                loss.backward()

                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
                optimizer.step()
                scheduler.step()

                total_steps += 1

                metrics = {
                    "loss": loss.item(),
                    "kl": kl.item(),
                    "px1": (e < .25).float().mean().item(),
                    "ro": ro.float().mean().item(),
                    "tr": tr.float().mean().item(),
                    "r1": (ro < .001).float().mean().item(),
                    "r2": (ro < .01).float().mean().item(),
                    "t1": (tr < .001).float().mean().item(),
                    "t2": (tr < .01).float().mean().item(),
                }

                if rank == 0:
                    logger.push(metrics)

                log_epoch = 10000 // args.world_size
                if total_steps % log_epoch == 0:
                # if total_steps % 100 == 0: ####10000
                # if total_steps == 1: ####10000
                    torch.cuda.empty_cache()

                    if rank == 0:

                        PATH = 'checkpoints/%s_%06d.pth' % (args.name, total_steps)
                        torch.save(net.state_dict(), PATH)

                    # validation_results = validate(None, net)
                    # if rank == 0:
                    #     logger.write_dict(validation_results)

                    torch.cuda.empty_cache()
                    net.train()

                # Update the progress bar
                pbar.update(1)
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bla', help='name your experiment')
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--steps', type=int, default=240000)
    parser.add_argument('--lr', type=float, default=0.00008)
    parser.add_argument('--clip', type=float, default=10.0)
    parser.add_argument('--n_frames', type=int, default=15)
    parser.add_argument('--pose_weight', type=float, default=10.0)
    parser.add_argument('--flow_weight', type=float, default=0.1)
    ####multigpu
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(), help='number of gpus')
    args = parser.parse_args()
    
    ####multigpu
    world_size = args.world_size
    rank = int(os.environ["LOCAL_RANK"])
    setup(rank, world_size)

    train(args)

# export NCCL_P2P_LEVEL=NVL
# CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nproc_per_node=2 train_multi_gpu.py --steps=270000 --lr=0.00008 --name=dy
# CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --standalone --nproc_per_node=4 train_multi_gpu.py --steps=240000 --lr=0.00008 --name=0311
#CUDA_VISIBLE_DEVICES=6,7,8,9 torchrun --standalone --nproc_per_node=4 train_multi_gpu.py --steps=240000 --lr=0.00008 --name=forth
#CUDA_VISIBLE_DEVICES=8,9 torchrun --standalone --nproc_per_node=1 train_multi_gpu.py --ckpt --steps=240000 --lr=0.00008 --name=forth 
# 0.00000004
# CUDA_VISIBLE_DEVICES=9 torchrun --standalone --nproc_per_node=1 train_multi_gpu.py --steps=240000 --lr=0.00008 --name=forth
#CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --standalone --nproc_per_node=4 train_multi_gpu.py --ckpt '/home/ghb/DPVO/checkpoints/0311_065000.pth' --steps=2000 --lr=0.00004 --name=0405
# CUDA_VISIBLE_DEVICES=6,9 torchrun --standalone --nproc_per_node=2 train_multi_gpu.py --ckpt '/home/ghb/DPVO/checkpoints/third_075000.pth' --steps=70000 --lr=0.00004 --name=