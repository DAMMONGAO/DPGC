from yacs.config import CfgNode as CN

_C = CN()

# max number of keyframes BUFFER_SIZE 表示关键帧的最大数量，限制系统在内存中保存的关键帧数量。
_C.BUFFER_SIZE = 4096

# bias patch selection towards high gradient regions? 关键帧的选取策略。此参数设置为 'RANDOM'，意味着关键帧选择是随机的，但在高梯度区域中可能存在一种偏向性。
_C.CENTROID_SEL_STRAT = 'RANDOM'
# _C.CENTROID_SEL_STRAT = 'BLOCKRANDOM'


# VO config (increase for better accuracy) 
_C.PATCHES_PER_FRAME = 80 #每帧图像要提取的“补丁”或特征点数量，数量越多精度越高，但计算开销也会增大。
_C.REMOVAL_WINDOW = 20 #移除补丁的窗口大小，即一段时间后被认为不再需要的特征点会被删除。
_C.OPTIMIZATION_WINDOW = 12 #优化窗口的大小，控制系统在多大时间范围内执行特征点位置的优化。
_C.PATCH_LIFETIME = 12 #补丁的生命周期，表示某一特征点在几帧后将被移除。

# threshold for keyframe removal 关键帧管理
_C.KEYFRAME_INDEX = 4 #关键帧的索引阈值
_C.KEYFRAME_THRESH = 12.5 #移除关键帧的阈值，数值越高意味着关键帧会保持更长时间。

# camera motion model 相机运动模型
_C.MOTION_MODEL = 'DAMPED_LINEAR' #相机的运动模型，用 'DAMPED_LINEAR' 表示一个带阻尼的线性模型。
_C.MOTION_DAMPING = 0.5 #运动阻尼系数，控制阻尼效果的强度。

_C.MIXED_PRECISION = True #是否启用混合精度运算，True 表示启用，用于提升性能，尤其在 GPU 上的推理效率。

# Loop closure 闭环检测
_C.LOOP_CLOSURE = False #是否启用闭环检测（通常在 SLAM 系统中用于检测是否重新访问了之前的位置）。
_C.BACKEND_THRESH = 64.0 #后台检测的阈值，控制检测闭环的频率
_C.MAX_EDGE_AGE = 1000 #最大边缘年龄（在图优化中可能表示边的使用次数上限）。
_C.GLOBAL_OPT_FREQ = 15 #全局优化频率，表示每隔多少次循环执行一次全局优化。

# Classic loop closure 经典闭环检测
_C.CLASSIC_LOOP_CLOSURE = True #是否使用经典的闭环检测方法。
_C.LOOP_CLOSE_WINDOW_SIZE = 3 #闭环检测窗口大小。
_C.LOOP_RETR_THRESH = 0.04 #闭环重检阈值，用于控制闭环检测的精度。

cfg = _C
