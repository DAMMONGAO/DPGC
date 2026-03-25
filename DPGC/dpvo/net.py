import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import torch_scatter
from torch_scatter import scatter_sum

from . import fastba
from . import altcorr
from . import lietorch
from .lietorch import SE3

from .extractor import BasicEncoder, BasicEncoder4
from .blocks import GradientClip, GatedResidual, SoftAgg
from .newadd import SoftAgg_moe, transformer_block, l_and_g_attention, gating_multi_conv, g_to_p_module, g_en_module

from .utils import *
from .ba import BA
from . import projective_ops as pops


autocast = torch.cuda.amp.autocast
import matplotlib.pyplot as plt

import logging

DIM = 384

###p all的话 仅打开penhance和patch_choose_new
### lgmvo 打开updater和moe

##新增patch选择方法
# 输入： b, n, c, h, w的特征图
# 输出：x：n, num_patches y: n, num_patches
import torch
import torch.nn as nn

class patch_vote(nn.Module):
    def __init__(self, 
                 channel=384,
                 select_k=20,
                 num_patches=96,):
        super(patch_vote, self).__init__()
        self.c = channel
        self.select_k = select_k
        self.num_patches = num_patches
        self.dwconv = nn.Conv2d(self.c, self.c, kernel_size=3, padding=1, groups=self.c)
        self.pointwise_conv = nn.Conv2d(self.c, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature):
        # 确定设备
        device = feature.device
        b, n, c, h, w = feature.shape
        # 计算每个点的得分
        x_tmp = feature.reshape(b*n, c, h, w)
        x_tmp = self.dwconv(x_tmp)
        x_tmp = self.pointwise_conv(x_tmp)
        score = self.sigmoid(x_tmp)  # b*n, 1, h, w
        score = score.reshape(b*n, -1)  # b*n, h*w

        # 根据得分值投票
        vote_values, vote_indices = torch.topk(score, self.select_k, dim=1)  # b*n, k

        # 确保 vote_indices 在正确的设备上
        vote_indices = vote_indices.to(device)

        # 生成每个样本的所有索引
        all_indices = torch.arange(h * w).unsqueeze(0).expand(b * n, -1).to(device)

        # 标记出已经选中的索引
        selected_mask = torch.zeros((b * n, h * w), dtype=torch.bool, device=device)
        selected_mask.scatter_(1, vote_indices, True)

        # 获取未选中的索引
        unselected_indices = all_indices[~selected_mask].reshape(b * n, -1)

        # 从剩余索引中随机选取 96 - k 个索引
        num_remaining = self.num_patches - self.select_k
        # 生成随机排列的索引
        random_perm = torch.randperm(unselected_indices.size(1)).unsqueeze(0).expand(b * n, -1).to(device)
        # 截取前 num_remaining 个索引
        random_remaining_indices = torch.gather(unselected_indices, 1, random_perm[:, :num_remaining])

        # 根据索引计算 x 和 y 坐标
        x_coords = vote_indices % w  # b*n, k
        y_coords = vote_indices // w  # b*n, k

        x_select = x_coords[:n, :]
        y_select = y_coords[:n, :]

        x_c = random_remaining_indices % w  # b*n, 96-k
        y_c = random_remaining_indices // w  # b*n, 96-k

        x_remain = x_c[:n, :]
        y_remain = y_c[:n, :]
        
        # 裁剪坐标，确保在 1 到 h-1 和 1 到 w-1 范围内
        x_select = torch.clamp(x_select, min=1, max=w - 1)
        y_select = torch.clamp(y_select, min=1, max=h - 1)
        x_remain = torch.clamp(x_remain, min=1, max=w - 1)
        y_remain = torch.clamp(y_remain, min=1, max=h - 1)

        # 将 x_select 和 x_remain 拼接在一起
        x = torch.cat((x_select, x_remain), dim=1)
        y = torch.cat((y_select, y_remain), dim=1)

        return x, y
    
class Update(nn.Module):
    def __init__(self, p):
        super(Update, self).__init__()
        
        ##新加参数
        temp = False  ##用于计算相邻边的注意力
        loc_k = False  ###对所有相邻的边，取topk进行局部整合
        attn = False  ##在gate基础上添加linear attention实现全局增强
        
        self.DIM = 384
        self.c1 = nn.Sequential(
            nn.Linear(self.DIM, self.DIM),
            nn.ReLU(inplace=True),
            nn.Linear(self.DIM, self.DIM))

        self.c2 = nn.Sequential(
            nn.Linear(self.DIM, self.DIM),
            nn.ReLU(inplace=True),
            nn.Linear(self.DIM, self.DIM))

        if temp:
            self.c1_score = nn.Sequential(
            nn.Linear(2*self.DIM, self.DIM),
            nn.Sigmoid())
            self.c2_score = nn.Sequential(
            nn.Linear(2*self.DIM, self.DIM),
            nn.Sigmoid())
        
        self.norm = nn.LayerNorm(self.DIM, eps=1e-3)

        ##old 
        if loc_k:
            self.agg_kk = SoftAgg_moe(self.DIM) #软聚合 加权求和
            self.agg_ij = SoftAgg_moe(self.DIM)
        else:
            self.agg_kk = SoftAgg(self.DIM) #软聚合 加权求和
            self.agg_ij = SoftAgg(self.DIM)

        if attn:
            self.gru = nn.Sequential(
                nn.LayerNorm(self.DIM, eps=1e-3),
                transformer_block(self.DIM, 
                                  num_heads=4, 
                                  mlp_ratio=2., 
                                  drop=0.1, 
                                  drop_path=0., 
                                  act_layer=nn.GELU, 
                                  norm_layer=nn.LayerNorm, 
                                  fr_ratio = 1, 
                                  linear=False, 
                                  se_ratio=1),
                nn.LayerNorm(self.DIM, eps=1e-3),
                GatedResidual(self.DIM),
            )           
        else:
            self.gru = nn.Sequential(
                nn.LayerNorm(self.DIM, eps=1e-3),
                GatedResidual(self.DIM),#x+f(x)sigma(x)
                nn.LayerNorm(self.DIM, eps=1e-3),
                GatedResidual(self.DIM),
            )
        
        ##add
        
        self.corr = nn.Sequential(
            nn.Linear(2*49*p*p, self.DIM),
            nn.ReLU(inplace=True),
            nn.Linear(self.DIM, self.DIM),
            nn.LayerNorm(self.DIM, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(self.DIM, self.DIM),
        )

        self.d = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(self.DIM, 2),
            GradientClip()) #梯度裁剪

        self.w = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(self.DIM, 2),
            GradientClip(),
            nn.Sigmoid())

    def forward(self, net, inp, corr, flow, ii, jj, kk):
        """ update operator """

        net = net + inp + self.corr(corr)
        net = self.norm(net)

        ix, jx = fastba.neighbors(kk, jj)

        mask_ix = (ix >= 0).float().reshape(1, -1, 1)
        mask_jx = (jx >= 0).float().reshape(1, -1, 1)
        
        result_tmp_1 = self.c1(mask_ix * net[:,ix])
        result_tmp_2 = self.c2(mask_jx * net[:,jx])
        
        
        temp = False
        if temp:
            # print('jiaquan')
            result_tmp = torch.cat([result_tmp_1,result_tmp_2],dim=-1)
            score_c1 = self.c1_score(result_tmp)
            score_c2 = self.c2_score(result_tmp)
            net = net + score_c1 * result_tmp_1  
            net = net + score_c2 * result_tmp_2  

        else:
            net = net + result_tmp_1
            net = net + result_tmp_2

        net = net + self.agg_kk(net, kk) ##patch聚合
        net = net + self.agg_ij(net, ii*12345 + jj) ##帧聚合

        net = self.gru(net)

        return net, (self.d(net), self.w(net), None)

class Patchifier(nn.Module):
    def __init__(self, patch_size=3):
        super(Patchifier, self).__init__()
        
        ##新增patch选择方法
        self.patch_choose_new = True
        if self.patch_choose_new:
            self.patch_vote = patch_vote(channel=128, select_k=20, num_patches=97)  ###k=20 num_patches=97   ##96 100 ####48 97
        
        # self.ldpm = False
        self.moe = False  #2 ###改成最后一个跑 无需注释 只需要修改false和true
        self.penhance = True #1 #测的时候把if self.moe注释掉 self.xnorm注释掉
        self.xnorm = False #3 #测的时候把if self.moe注释掉 把self.penhance变成false
        
        self.DIM = 384
        self.patch_size = patch_size
        self.fnet = BasicEncoder4(output_dim=128, norm_fn='instance')
        self.inet = BasicEncoder4(output_dim=self.DIM, norm_fn='none')
        
        if self.moe:
            self.moe_fnet = gating_multi_conv(d_model=128, d_hidden_model=256)
        if self.penhance:
            self.pe_fnet = g_to_p_module(d_model=128, 
                               nhead=4)
        if self.xnorm:
            self.xn_fnet = g_en_module(d_model=128,
                 nhead=4,
                 attention='linear',
                 iscoarse=False)
        
    
    def __image_gradient(self, images):
        gray = ((images + 0.5) * (255.0 / 2)).sum(dim=2)
        dx = gray[...,:-1,1:] - gray[...,:-1,:-1]
        dy = gray[...,1:,:-1] - gray[...,:-1,:-1]
        g = torch.sqrt(dx**2 + dy**2)
        g = F.avg_pool2d(g, 4, 4)
        return g

    def forward(self, images, patches_per_image=80, disps=None, centroid_sel_strat='RANDOM', return_color=False):
        """ extract patches from input images """
        #image的维度torch.Size([1, 1, 3, 640, 352]) 

        fmap = self.fnet(images) / 4.0 #torch.Size([1, 1, 128, 160, 88])
        
        if self.moe:
            fmap = self.moe_fnet(x=fmap)
        if self.xnorm:
            # import pdb; pdb.set_trace()
            fmap = self.xn_fnet(x=fmap)

        
        #上下文特征
        imap = self.inet(images) / 4.0 #torch.Size([1, 1, 384, 160, 88])

        b, n, c, h, w = fmap.shape #检测到n为1

        P = self.patch_size

        # bias patch selection towards regions with high gradient
        if centroid_sel_strat == 'GRADIENT_BIAS':
            g = self.__image_gradient(images)
            x = torch.randint(1, w-1, size=[n, 3*patches_per_image], device="cuda")
            y = torch.randint(1, h-1, size=[n, 3*patches_per_image], device="cuda")

            coords = torch.stack([x, y], dim=-1).float()
            g = altcorr.patchify(g[0,:,None], coords, 0).view(n, 3 * patches_per_image)
            
            ix = torch.argsort(g, dim=1)
            x = torch.gather(x, 1, ix[:, -patches_per_image:])
            y = torch.gather(y, 1, ix[:, -patches_per_image:])

        elif centroid_sel_strat == 'RANDOM':
            if self.patch_choose_new:
                x, y = self.patch_vote(fmap) #torch.Size([1, 96]) torch.Size([1, 96])
                x = x[:,:patches_per_image]
                y = y[:,:patches_per_image]
            # print(patches_per_image)
            else:
                x = torch.randint(1, w-1, size=[n, patches_per_image], device="cuda")
                y = torch.randint(1, h-1, size=[n, patches_per_image], device="cuda")

        ##新增
        # elif centroid_sel_strat == 'BLOCKRANDOM':
        #     # 每个区域内的patch数量
        #     patches_per_region = patches_per_image // 4  # 假设patches_per_image是总共需要的patch数量
            
        #     # 分区范围 (假设特征图的大小是 h x w)
        #     # 区域格式： (y_start, y_end, x_start, x_end)
        #     regions = [
        #         (1, h//2, 1, w//2),  # 上左区，避免边界
        #         (1, h//2, w//2, w - 1),  # 上右区，避免边界
        #         (h//2, h - 1, 1, w//2),  # 下左区，避免边界
        #         (h//2, h - 1, w//2, w - 1)   # 下右区，避免边界
        #     ]
            
        #     # 在每个区域内采样的x和y坐标
        #     x_coords_list = []
        #     y_coords_list = []
            
        #     # 在每个区域内随机采样
        #     for (y_start, y_end, x_start, x_end) in regions:
        #         # 在当前区域内随机生成patch坐标
        #         x_coords = torch.randint(x_start, x_end, size=[n, patches_per_region], device="cuda")
        #         y_coords = torch.randint(y_start, y_end, size=[n, patches_per_region], device="cuda")
                
        #         # 将当前区域的采样结果添加到列表中
        #         x_coords_list.append(x_coords)
        #         y_coords_list.append(y_coords)
            
        #     # 将每个区域的坐标合并成一个大的batch
        #     x = torch.cat(x_coords_list, dim=1)  # 在列方向上拼接（合并所有区域的x坐标）
        #     y = torch.cat(y_coords_list, dim=1)  # 在列方向上拼接（合并所有区域的y坐标）
        
        else:
            raise NotImplementedError(f"Patch centroid selection not implemented: {centroid_sel_strat}")

        coords = torch.stack([x, y], dim=-1).float()
        # print(coords.shape)
        # exit()
        #上下文特征提取patch 利用随机方法
        imap = altcorr.patchify(imap[0], coords, 0).view(b, -1, self.DIM, 1, 1) #torch.Size([1, 96, 384, 1, 1])
        #匹配特征提取patch 利用随机方法
        gmap = altcorr.patchify(fmap[0], coords, P//2).view(b, -1, 128, P, P) #torch.Size([1, 96, 128, 3, 3])

        if self.penhance:
            gmap = self.pe_fnet(x=gmap, source=fmap)


        if return_color:
            clr = altcorr.patchify(images[0], 4*(coords + 0.5), 0).view(b, -1, 3)

        if disps is None:
            disps = torch.ones(b, n, h, w, device="cuda")

        grid, _ = coords_grid_with_index(disps, device=fmap.device) ##torch.Size([1, 1, 3, 160, 88]) x y d=1

        patches = altcorr.patchify(grid[0], coords, P//2).view(b, -1, 3, P, P) #构造出来的patch

        index = torch.arange(n, device="cuda").view(n, 1)
        index = index.repeat(1, patches_per_image).reshape(-1) #0,0,0,1,1,1,...,96,96,96

        if return_color:
            return fmap, gmap, imap, patches, index, clr

        return fmap, gmap, imap, patches, index
    
#使用卷积层和resnet层从图像提取匹配特征和上下文特征
# fmap 维度为#torch.Size([1, 1, 128, 160, 88])的匹配特征
# gmap 维度为#torch.Size([1, 96, 128, 3, 3])的匹配特征提取的patch patch大小为3 每张图像共96个
# imap #torch.Size([1, 96, 384, 1, 1])的上下文特征提取的patch
# patches #通过加入d=1构建的包含x y d 在内的提取的patch
# index 构建的包含帧数、patch数量在内的索引

class CorrBlock:
    def __init__(self, fmap, gmap, radius=3, dropout=0.2, levels=[1,4]):
        self.dropout = dropout
        self.radius = radius
        self.levels = levels

        self.gmap = gmap
        self.pyramid = pyramidify(fmap, lvls=levels)

    def __call__(self, ii, jj, coords):
        corrs = []
        for i in range(len(self.levels)):
            corrs += [ altcorr.corr(self.gmap, self.pyramid[i], coords / self.levels[i], ii, jj, self.radius, self.dropout) ]
        return torch.stack(corrs, -1).view(1, len(ii), -1)


class VONet(nn.Module):
    def __init__(self, use_viewer=False):
        super(VONet, self).__init__()
        self.P = 3
        self.patchify = Patchifier(self.P)
        self.update = Update(self.P)

        self.DIM = 384
        self.RES = 4


    @autocast(enabled=False)
    def forward(self, images, poses, disps, intrinsics, M=1024, STEPS=12, P=1, structure_only=False, rescale=False):
        """ Estimates SE3 or Sim3 between pair of frames """

        images = 2 * (images / 255.0) - 0.5
        intrinsics = intrinsics / 4.0
        # print('sss')
        # print(disps.shape)
        # exit()
        disps = disps[:, :, 1::4, 1::4].float()

        fmap, gmap, imap, patches, ix = self.patchify(images, disps=disps)

        corr_fn = CorrBlock(fmap, gmap)

        b, N, c, h, w = fmap.shape
        p = self.P

        patches_gt = patches.clone()
        Ps = poses

        d = patches[..., 2, p//2, p//2]
        patches = set_depth(patches, torch.rand_like(d))

        kk, jj = flatmeshgrid(torch.where(ix < 8)[0], torch.arange(0,8, device="cuda"), indexing='ij')
        ii = ix[kk]

        imap = imap.view(b, -1, self.DIM)
        net = torch.zeros(b, len(kk), self.DIM, device="cuda", dtype=torch.float)
        
        Gs = SE3.IdentityLike(poses)

        if structure_only:
            Gs.data[:] = poses.data[:]

        traj = []
        bounds = [-64, -64, w + 64, h + 64]
        
        while len(traj) < STEPS:
            Gs = Gs.detach()
            patches = patches.detach()

            n = ii.max() + 1
            if len(traj) >= 8 and n < images.shape[1]:
                if not structure_only: Gs.data[:,n] = Gs.data[:,n-1]
                kk1, jj1 = flatmeshgrid(torch.where(ix  < n)[0], torch.arange(n, n+1, device="cuda"), indexing='ij')
                kk2, jj2 = flatmeshgrid(torch.where(ix == n)[0], torch.arange(0, n+1, device="cuda"), indexing='ij')

                ii = torch.cat([ix[kk1], ix[kk2], ii])
                jj = torch.cat([jj1, jj2, jj])
                kk = torch.cat([kk1, kk2, kk])

                net1 = torch.zeros(b, len(kk1) + len(kk2), self.DIM, device="cuda")
                net = torch.cat([net1, net], dim=1)

                if np.random.rand() < 0.1:
                    k = (ii != (n - 4)) & (jj != (n - 4))
                    ii = ii[k]
                    jj = jj[k]
                    kk = kk[k]
                    net = net[:,k]

                patches[:,ix==n,2] = torch.median(patches[:,(ix == n-1) | (ix == n-2),2])
                n = ii.max() + 1

            coords = pops.transform(Gs, patches, intrinsics, ii, jj, kk)
            coords1 = coords.permute(0, 1, 4, 2, 3).contiguous()

            corr = corr_fn(kk, jj, coords1)
            net, (delta, weight, _) = self.update(net, imap[:,kk], corr, None, ii, jj, kk)

            lmbda = 1e-4
            target = coords[...,p//2,p//2,:] + delta

            ep = 10
            for itr in range(2):
                Gs, patches = BA(Gs, patches, intrinsics, target, weight, lmbda, ii, jj, kk, 
                    bounds, ep=ep, fixedp=1, structure_only=structure_only)

            kl = torch.as_tensor(0)
            dij = (ii - jj).abs()
            k = (dij > 0) & (dij <= 2)

            coords = pops.transform(Gs, patches, intrinsics, ii[k], jj[k], kk[k])
            coords_gt, valid, _ = pops.transform(Ps, patches_gt, intrinsics, ii[k], jj[k], kk[k], jacobian=True)

            traj.append((valid, coords, coords_gt, Gs[:,:n], Ps[:,:n], kl))

        return traj

