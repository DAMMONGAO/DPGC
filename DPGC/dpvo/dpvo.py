import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from . import altcorr, fastba, lietorch
from . import projective_ops as pops
from .lietorch import SE3
from .net import VONet
from .patchgraph import PatchGraph
from .utils import *
import logger
import logging

mp.set_start_method('spawn', True)

autocast = torch.cuda.amp.autocast
Id = SE3.Identity(1, device="cuda")

class DPVO:

    def __init__(self, cfg, network, ht=480, wd=640, viz=False):
        self.cfg = cfg
        self.load_weights(network) #加载模型权重 vonet相应的初始化和模型调用
        self.is_initialized = False
        self.enable_timing = False
        torch.set_num_threads(2)

        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE

        self.ht = ht    # image height
        self.wd = wd    # image width

        DIM = self.DIM #384
        RES = self.RES #4

        ### state attributes ###
        self.tlist = []
        self.counter = 0

        # keep track of global-BA calls
        self.ran_global_ba = np.zeros(100000, dtype=bool)

        ht = ht // RES
        wd = wd // RES

        # dummy image for visualization
        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        ### network attributes ###
        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}

        ### frame memory size ###
        self.pmem = self.mem = 36 # 32 was too small given default settings
        if self.cfg.LOOP_CLOSURE:
            self.last_global_ba = -1000 # keep track of time since last global opt
            self.pmem = self.cfg.MAX_EDGE_AGE # patch memory

        self.imap_ = torch.zeros(self.pmem, self.M, DIM, **kwargs)
        self.gmap_ = torch.zeros(self.pmem, self.M, 128, self.P, self.P, **kwargs)

        self.pg = PatchGraph(self.cfg, self.P, self.DIM, self.pmem, **kwargs) #初始化patch图

        # classic backend
        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.load_long_term_loop_closure()

        self.fmap1_ = torch.zeros(1, self.mem, 128, ht // 1, wd // 1, **kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, 128, ht // 4, wd // 4, **kwargs)

        # feature pyramid 特征金字塔
        self.pyramid = (self.fmap1_, self.fmap2_)

        self.viewer = None
        if viz:
            self.start_viewer()

    def load_long_term_loop_closure(self):
        try:
            from .loop_closure.long_term import LongTermLoopClosure
            self.long_term_lc = LongTermLoopClosure(self.cfg, self.pg)
        except ModuleNotFoundError as e:
            self.cfg.CLASSIC_LOOP_CLOSURE = False
            print(f"WARNING: {e}")

    # ##原始
    # def load_weights(self, network):
    #     # load network from checkpoint file
    #     if isinstance(network, str):
    #         from collections import OrderedDict
    #         state_dict = torch.load(network)
    #         new_state_dict = OrderedDict()
    #         for k, v in state_dict.items():
    #             if "update.lmbda" not in k:
    #                 new_state_dict[k.replace('module.', '')] = v
            
    #         self.network = VONet()
    #         self.network.load_state_dict(new_state_dict)

    #     else:
    #         self.network = network

    #     # steal network attributes
        
    #     if hasattr(self.network, 'module'):
    #         self.DIM = self.network.module.DIM
    #         self.RES = self.network.module.RES
    #         self.P = self.network.module.P
    #     else:
    #         self.DIM = self.network.DIM
    #         self.RES = self.network.RES
    #         self.P = self.network.P
    #     # self.DIM = self.network.DIM
    #     # self.RES = self.network.RES
    #     # self.P = self.network.P

    #     self.network.cuda()
    #     self.network.eval()

    ##改1
    # def load_weights(self, network):
    #     # load network from checkpoint file
    #     if isinstance(network, str):
    #         from collections import OrderedDict
    #         state_dict = torch.load(network)
    #         new_state_dict = OrderedDict()
    #         for k, v in state_dict.items():
    #             if "update.lmbda" not in k:
    #                 if 'patchify.xn_fnet' not in k: 
    #                     new_state_dict[k.replace('module.', '')] = v
    #                 else:
    #                     if k.startswith('module.'):
    #                         new_key = k[len('module.'):]  # 移除开头的 'module.'
    #                     else:
    #                         new_key = k
    #                     new_state_dict[new_key] = v
            
    #         # import pdb; pdb.set_trace()
    #         self.network = VONet()
    #         self.network.load_state_dict(new_state_dict)

    #     else:
    #         self.network = network

    #     # steal network attributes
        
    #     if hasattr(self.network, 'module'):
    #         self.DIM = self.network.module.DIM
    #         self.RES = self.network.module.RES
    #         self.P = self.network.module.P
    #     else:
    #         self.DIM = self.network.DIM
    #         self.RES = self.network.RES
    #         self.P = self.network.P
    #     # self.DIM = self.network.DIM
    #     # self.RES = self.network.RES
    #     # self.P = self.network.P

    #     self.network.cuda()
    #     self.network.eval()
    
    ##for levo
    def load_weights(self, network):
        """加载模型权重并设置网络属性"""
        try:
            # 从检查点文件加载网络
            if isinstance(network, str):
                # 加载状态字典
                state_dict = torch.load(network, map_location='cpu')
                
                # 处理DataParallel包装的模型权重
                new_state_dict = {}
                for k, v in state_dict.items():
                    # 排除不需要的参数
                    if "update.lmbda" in k:
                        continue
                        
                    # 处理patchify.xn_fnet特殊情况
                    if 'patchify.xn_fnet' in k:
                        k = k.replace('module.', '')
                        
                    # 移除DataParallel添加的前缀
                    elif k.startswith('module.'):
                        k = k[7:]  # 移除'module.'
                    
                    # 特殊处理: 修复线性模块命名不匹配
                    if 'patchify.patch_vote.g_en_func.linear_' in k:
                        k = k.replace('patchify.patch_vote.g_en_func.linear_', 
                                    'patchify.patch_vote.g_en_func.linear_module.')
                    
                    new_state_dict[k] = v
                
                # 初始化网络并加载权重
                self.network = VONet()
                # 使用非严格模式加载，允许部分参数不匹配
                self.network.load_state_dict(new_state_dict, strict=False)
                
            else:
                # 直接使用提供的网络实例
                self.network = network

            # 获取网络属性
            target_module = self.network.module if hasattr(self.network, 'module') else self.network
            self.DIM = getattr(target_module, 'DIM', None)
            self.RES = getattr(target_module, 'RES', None)
            self.P = getattr(target_module, 'P', None)
            
            # 检查关键属性是否成功获取
            if any(attr is None for attr in [self.DIM, self.RES, self.P]):
                logger.warning("网络缺少部分关键属性: DIM={}, RES={}, P={}".format(
                    self.DIM, self.RES, self.P))

            # 设置网络为评估模式并移至GPU
            self.network.cuda()
            self.network.eval()
            print("模型权重加载成功并已设置为评估模式")
            
        except Exception as e:
            print(f"加载模型权重时发生错误: {str(e)}")
            raise

    def start_viewer(self):
        from dpviewer import Viewer

        intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")

        self.viewer = Viewer(
            self.image_,
            self.pg.poses_,
            self.pg.points_,
            self.pg.colors_,
            intrinsics_)

    @property
    def poses(self):
        return self.pg.poses_.view(1, self.N, 7)

    @property
    def patches(self): #self.patches
        # print(self.N)
        return self.pg.patches_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.pg.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.pg.index_.view(-1)#展平

    @property
    def imap(self):
        return self.imap_.view(1, self.pmem * self.M, self.DIM)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.pmem * self.M, 128, 3, 3)

    @property
    def n(self):
        return self.pg.n

    @n.setter
    def n(self, val):
        self.pg.n = val

    @property
    def m(self):
        return self.pg.m

    @m.setter
    def m(self, val):
        self.pg.m = val

    def get_pose(self, t):
        #import pdb; pdb.set_trace()
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.pg.delta[t]
        return dP * self.get_pose(t0)

    #独立函数
    def terminate(self):
        #import pdb; pdb.set_trace()

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc.terminate(self.n)

        if self.cfg.LOOP_CLOSURE:
            self.append_factors(*self.pg.edges_loop())

        for _ in range(12):
            self.ran_global_ba[self.n] = False
            self.update()

        """ interpolate missing poses """
        self.traj = {}
        for i in range(self.n):
            #import pdb; pdb.set_trace()
            self.traj[self.pg.tstamps_[i]] = self.pg.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=np.float64)
        if self.viewer is not None:
            self.viewer.join()

        # Poses: x y z qx qy qz qw
        return poses, tstamps

    #用于motion_probe函数 计算特征相关性
    def corr(self, coords, indicies=None):
        """ local correlation volume """
        ii, jj = indicies if indicies is not None else (self.pg.kk, self.pg.jj)
        
        # print('ii', ii.shape)
        # print('jj', jj.shape)
        # print('-----------')
        
        ii1 = ii % (self.M * self.pmem)
        jj1 = jj % (self.mem)
        
        # print('ii1', ii1)
        # print('jj1', jj1)
        
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        # print('corr1',corr1.shape) #corr1 torch.Size([1, 34080, 7, 7, 3, 3])
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        # print('corr2',corr2.shape) #corr2 torch.Size([1, 34080, 7, 7, 3, 3])
        # print(torch.stack([corr1, corr2], -1).shape)
        # print('-----------')
        
        #torch.Size([1, 34080, 7, 7, 3, 3, 2]) 
        
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    #用于motion_probe函数和update函数
    def reproject(self, indicies=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies if indicies is not None else (self.pg.ii, self.pg.jj, self.pg.kk)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        # print(coords.shape) torch.Size([1, 36864, 3, 3, 2])
        # print('------------')

        return coords.permute(0, 1, 4, 2, 3).contiguous()

    #用于call函数 用于添加图网络的边 更新patch图
    def append_factors(self, ii, jj):
        self.pg.jj = torch.cat([self.pg.jj, jj]) #
        self.pg.kk = torch.cat([self.pg.kk, ii]) #patch的索引
        self.pg.ii = torch.cat([self.pg.ii, self.ix[ii]]) #第n帧
        
        # print('jj:',jj.shape)
        # print('pg.jj:',self.pg.jj.shape)
        # print('ii:',ii.shape)
        # print('pg.kk:',self.pg.kk.shape)
        # # print('len',len(ii))
        # print('-------------------------')

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        self.pg.net = torch.cat([self.pg.net, net], dim=1)
        # print('self.pg.net.shape:', self.pg.net.shape)
        
    #用于keyframe函数
    def remove_factors(self, m, store: bool):
        assert self.pg.ii.numel() == self.pg.weight.shape[1]
        if store:
            self.pg.ii_inac = torch.cat((self.pg.ii_inac, self.pg.ii[m]))
            self.pg.jj_inac = torch.cat((self.pg.jj_inac, self.pg.jj[m]))
            self.pg.kk_inac = torch.cat((self.pg.kk_inac, self.pg.kk[m]))
            self.pg.weight_inac = torch.cat((self.pg.weight_inac, self.pg.weight[:,m]), dim=1)
            self.pg.target_inac = torch.cat((self.pg.target_inac, self.pg.target[:,m]), dim=1)
        self.pg.weight = self.pg.weight[:,~m]
        self.pg.target = self.pg.target[:,~m]

        self.pg.ii = self.pg.ii[~m]
        self.pg.jj = self.pg.jj[~m]
        self.pg.kk = self.pg.kk[~m]
        self.pg.net = self.pg.net[:,~m]
        assert self.pg.ii.numel() == self.pg.weight.shape[1]

    #用于call函数 探测运动是否充足
    def motion_probe(self):
        """ kinda hacky way to ensure enough motion for initialization """
        
        kk = torch.arange(self.m-self.M, self.m, device="cuda")
        jj = self.n * torch.ones_like(kk)
        ii = self.ix[kk]
        
        #ii i jj i+1 kk i*96_i*96+95
        # 0 1 0-95 
        # 1 2 96-191
        
        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs) #1 96 384
        coords = self.reproject(indicies=(ii, jj, kk)) #torch.Size([1, 96, 2, 3, 3]) x y kj上的投影

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            corr = self.corr(coords, indicies=(kk, jj)) #
            # print(corr.shape) #torch.Size([1, 96, 882]) 7 7 3 3 2 882
            
            ctx = self.imap[:,kk % (self.M * self.pmem)]
            # print(ctx.shape) #torch.Size([1, 96, 384])
            # print('-------------')
            if hasattr(self.network, 'module'):
                net, (delta, weight, _) = \
                self.network.module.update(net, ctx, corr, None, ii, jj, kk)
            else:
                net, (delta, weight, _) = \
                self.network.update(net, ctx, corr, None, ii, jj, kk)
            # print(delta.shape) torch.Size([1, 96, 2])

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    #用于keyframe函数
    def motionmag(self, i, j):
        k = (self.pg.ii == i) & (self.pg.jj == j)
        ii = self.pg.ii[k]
        jj = self.pg.jj[k]
        kk = self.pg.kk[k]

        flow, _ = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    #用于call函数
    def keyframe(self):

        i = self.n - self.cfg.KEYFRAME_INDEX - 1
        j = self.n - self.cfg.KEYFRAME_INDEX + 1
        m = self.motionmag(i, j) + self.motionmag(j, i)
 
        if m / 2 < self.cfg.KEYFRAME_THRESH:
            k = self.n - self.cfg.KEYFRAME_INDEX
            t0 = self.pg.tstamps_[k-1]
            t1 = self.pg.tstamps_[k]

            dP = SE3(self.pg.poses_[k]) * SE3(self.pg.poses_[k-1]).inv()
            self.pg.delta[t1] = (t0, dP)

            to_remove = (self.pg.ii == k) | (self.pg.jj == k)
            self.remove_factors(to_remove, store=False)

            self.pg.kk[self.pg.ii > k] -= self.M
            self.pg.ii[self.pg.ii > k] -= 1
            self.pg.jj[self.pg.jj > k] -= 1

            for i in range(k, self.n-1):
                self.pg.tstamps_[i] = self.pg.tstamps_[i+1]
                self.pg.colors_[i] = self.pg.colors_[i+1]
                self.pg.poses_[i] = self.pg.poses_[i+1]
                self.pg.patches_[i] = self.pg.patches_[i+1]
                self.pg.intrinsics_[i] = self.pg.intrinsics_[i+1]

                self.imap_[i % self.pmem] = self.imap_[(i+1) % self.pmem]
                self.gmap_[i % self.pmem] = self.gmap_[(i+1) % self.pmem]
                self.fmap1_[0,i%self.mem] = self.fmap1_[0,(i+1)%self.mem]
                self.fmap2_[0,i%self.mem] = self.fmap2_[0,(i+1)%self.mem]

            self.n -= 1
            self.m-= self.M

            if self.cfg.CLASSIC_LOOP_CLOSURE:
                self.long_term_lc.keyframe(k)

        to_remove = self.ix[self.pg.kk] < self.n - self.cfg.REMOVAL_WINDOW # Remove edges falling outside the optimization window
        if self.cfg.LOOP_CLOSURE:
            # ...unless they are being used for loop closure
            lc_edges = ((self.pg.jj - self.pg.ii) > 30) & (self.pg.jj > (self.n - self.cfg.OPTIMIZATION_WINDOW))
            to_remove = to_remove & ~lc_edges
        self.remove_factors(to_remove, store=True)

    #用于update函数
    def __run_global_BA(self):
        """ Global bundle adjustment
         Includes both active and inactive edges """
        full_target = torch.cat((self.pg.target_inac, self.pg.target), dim=1)
        full_weight = torch.cat((self.pg.weight_inac, self.pg.weight), dim=1)
        full_ii = torch.cat((self.pg.ii_inac, self.pg.ii))
        full_jj = torch.cat((self.pg.jj_inac, self.pg.jj))
        full_kk = torch.cat((self.pg.kk_inac, self.pg.kk))

        self.pg.normalize()
        lmbda = torch.as_tensor([1e-4], device="cuda")
        t0 = self.pg.ii.min().item()
        fastba.BA(self.poses, self.patches, self.intrinsics,
            full_target, full_weight, lmbda, full_ii, full_jj, full_kk, t0, self.n, M=self.M, iterations=2, eff_impl=True)
        self.ran_global_ba[self.n] = True

    #用于call函数
    def update(self):
        with Timer("other", enabled=self.enable_timing):#创建一个计时器对象 便于获取代码运行时间
            coords = self.reproject()
            # print('coords',coords.shape) #coords torch.Size([1, 34080, 2, 3, 3])

            with autocast(enabled=True):
                corr = self.corr(coords)
                # print('corr',corr.shape) #torch.Size([1, 34080, 882])
                ctx = self.imap[:, self.pg.kk % (self.M * self.pmem)]
                # print('ctx',ctx.shape) #torch.Size([1, 34080, 384])
                if hasattr(self.network, 'module'):
                    self.pg.net, (delta, weight, _) = \
                    self.network.module.update(self.pg.net, ctx, corr, None, self.pg.ii, self.pg.jj, self.pg.kk)
                else:
                    self.pg.net, (delta, weight, _) = \
                    self.network.update(self.pg.net, ctx, corr, None, self.pg.ii, self.pg.jj, self.pg.kk)
            # print('delta',delta.shape) #delta torch.Size([1, 34080, 2])
            lmbda = torch.as_tensor([1e-4], device="cuda")
            weight = weight.float()
            target = coords[...,self.P//2,self.P//2] + delta.float()

        self.pg.target = target
        self.pg.weight = weight

        with Timer("BA", enabled=self.enable_timing):
            try:
                # run global bundle adjustment if there exist long-range edges
                if (self.pg.ii < self.n - self.cfg.REMOVAL_WINDOW - 1).any() and not self.ran_global_ba[self.n]:
                    self.__run_global_BA()
                else:
                    t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
                    t0 = max(t0, 1)
                    fastba.BA(self.poses, self.patches, self.intrinsics, 
                        target, weight, lmbda, self.pg.ii, self.pg.jj, self.pg.kk, t0, self.n, M=self.M, iterations=2, eff_impl=False)
            except:
                print("Warning BA failed...")

            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            self.pg.points_[:len(points)] = points[:]

    #用于call函数 生成网格索引 96 n-r~n-1,1(n-1) 这些索引帧下的patch才能被保存或记住
    #过去存储的多帧n-r到n-2patch对n-1一帧的索引
    def __edges_forw(self):
        r=self.cfg.PATCH_LIFETIME
        # print('r',r)
        # print('ffffff')
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n-1, self.n, device="cuda"), indexing='ij')

    #用于call函数 生成网格索引
    #过去n-1一帧的patch对存储的n-r到n-1多个帧的索引
    def __edges_back(self):
        r=self.cfg.PATCH_LIFETIME
        # print('bbbbbb')
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n-r, 0), self.n, device="cuda"), indexing='ij')

    #测试直接调用该函数
    def __call__(self, tstamp, image, intrinsics):
        #import pdb; pdb.set_trace()
        
        """ track new frame """
        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc(image, self.n)

        if (self.n+1) >= self.N:
            raise Exception(f'The buffer size is too small. You can increase it using "--opts BUFFER_SIZE={self.N*2}"')

        if self.viewer is not None:
            self.viewer.update_image(image.contiguous())

        image = 2 * (image[None,None] / 255.0) - 0.5 #转化为灰度
        
        with autocast(enabled=self.cfg.MIXED_PRECISION):
            if hasattr(self.network, 'module'):
                fmap, gmap, imap, patches, _, clr = \
                self.network.module.patchify(image,
                    patches_per_image=self.cfg.PATCHES_PER_FRAME, 
                    centroid_sel_strat=self.cfg.CENTROID_SEL_STRAT, 
                    return_color=True)
            else:
                fmap, gmap, imap, patches, _, clr = \
                self.network.patchify(image,
                    patches_per_image=self.cfg.PATCHES_PER_FRAME, 
                    centroid_sel_strat=self.cfg.CENTROID_SEL_STRAT, 
                    return_color=True)
        #使用卷积层和resnet层从图像提取匹配特征和上下文特征
        # fmap 维度为#torch.Size([1, 1, 128, 160, 88])的匹配特征
        # gmap 维度为#torch.Size([1, 96, 128, 3, 3])的匹配特征提取的patch patch大小为3 每张图像共96个
        # imap #torch.Size([1, 96, 384, 1, 1])的上下文特征提取的patch
        # patches #通过加入d=1构建的包含x y d 在内的提取的patch ([1, 96, 3, 3, 3])
        # index 构建的包含帧数、patch数量在内的索引
        
        ### update state attributes ### 更新状态属性 时间戳 内参等
        self.tlist.append(tstamp)
        self.pg.tstamps_[self.n] = self.counter
        self.pg.intrinsics_[self.n] = intrinsics / self.RES

        # color info for visualization
        clr = (clr[0,:,[2,1,0]] + 0.5) * (255.0 / 2)
        self.pg.colors_[self.n] = clr.to(torch.uint8)

        #self.n 帧数量
        #self.m patch数量
        #self.M 每个帧中最多的patch数量

        self.pg.index_[self.n + 1] = self.n + 1
        self.pg.index_map_[self.n + 1] = self.m + self.M
        
        # print(self.pg.index_)
        # print(self.pg.index_map_)
        # exit()

        #利用相机模型 过去两帧的位姿预测当前帧的位姿态 存储到self.pg.poses_中
        if self.n > 1:
            # print('--------------')
            # print('start', self.n)
            if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                P1 = SE3(self.pg.poses_[self.n-1])
                P2 = SE3(self.pg.poses_[self.n-2])

                # To deal with varying camera hz
                *_, a,b,c = [1]*3 + self.tlist
                fac = (c-b) / (b-a)

                xi = self.cfg.MOTION_DAMPING * fac * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.pg.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.poses[self.n-1]
                self.pg.poses_[self.n] = tvec_qvec
        # else:
        #     print('n',self.n)

        # TODO better depth initialization 深度估计
        # print(patches[:,:,2,0,0,None,None].shape)
        patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None]) #深度值随机初始化
        # print(patches.shape)
        # exit()
        if self.is_initialized:
            s = torch.median(self.pg.patches_[self.n-3:self.n,:,2]) #多帧中位数更新
            patches[:,:,2] = s

        self.pg.patches_[self.n] = patches

        ### update network attributes ### 只能存储pmem个patch和特征金字塔
        self.imap_[self.n % self.pmem] = imap.squeeze() #移除这些张量中所有维度为 1 的维度
        self.gmap_[self.n % self.pmem] = gmap.squeeze()
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        self.counter += 1
        #n++++++        
        if self.n > 0 and not self.is_initialized:
            #如果推算出的网络的delta过小的话就不更新delta列表，仍旧保存上一帧的
            if self.motion_probe() < 2.0:
            # if self.motion_probe() < 0.002:
                self.pg.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return

        self.n += 1
        self.m += self.M

        if self.cfg.LOOP_CLOSURE:
            if self.n - self.last_global_ba >= self.cfg.GLOBAL_OPT_FREQ:
                """ Add loop closure factors """
                lii, ljj = self.pg.edges_loop()
                if lii.numel() > 0:
                    self.last_global_ba = self.n
                    self.append_factors(lii, ljj)

        # Add forward and backward factors #添加边edge的一些索引
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())

        if self.n == 8 and not self.is_initialized:
            self.is_initialized = True

            for itr in range(12):
                self.update()

        elif self.is_initialized:
            self.update()
            self.keyframe()

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc.attempt_loop_closure(self.n)
            self.long_term_lc.lc_callback()
