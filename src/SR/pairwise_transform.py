from __future__ import division  # 确保除法运算符的行为与 Python 3.x 一致（Python 2.x 的兼容性代码）
import random
import numpy as np
from PIL import Image
from lib import data_augmentation
from lib import iproc

# 定义 _noise 函数，用于对图像添加 JPEG 噪声
def _noise(src, p, level):
    # 初始化 YUV 采样因子为 444 格式（全采样）
    sampling_factor = '1x1,1x1,1x1'
    # 以概率 p 改变 YUV 采样因子为 420 格式（降低色度采样率）
    if np.random.uniform() < p:
        sampling_factor = '2x2,1x1,1x1'
    # 根据噪声级别 level 添加不同程度的 JPEG 噪声
    if level == 0:
        # 轻微噪声：随机 JPEG 质量因子在 85 到 100 之间
        dst = iproc.jpeg(src, sampling_factor, random.randint(90, 100))
        return dst
    elif level == 1:
        # 中等噪声：随机 JPEG 质量因子在 65 到 90 之间
        dst = iproc.jpeg(src, sampling_factor, random.randint(65, 90))
        return dst
    elif level == 2 or level == 3:
        # 较高噪声：随机多次压缩，质量因子更低
        rand = np.random.uniform()
        if rand < 0.6:
            dst = iproc.jpeg(src, sampling_factor, random.randint(25, 70))
        elif rand < 0.9:
            dst = iproc.jpeg(src, sampling_factor, random.randint(35, 70))
            dst = iproc.jpeg(dst, sampling_factor, random.randint(25, 65))
        else:
            dst = iproc.jpeg(src, sampling_factor, random.randint(50, 70))
            dst = iproc.jpeg(dst, sampling_factor, random.randint(35, 65))
            dst = iproc.jpeg(dst, sampling_factor, random.randint(25, 55))
        return dst
    else:
        # 如果噪声级别不在预定义范围内，抛出错误
        raise ValueError('Unknown noise level: {}'.format(level))

# 定义 noise 函数，用于对图像以一定概率添加噪声
def noise(src, p, p_chroma, level):
    # 以概率 p 决定是否对图像添加噪声
    if np.random.uniform() < p:
        # 使用 iproc 的 array_to_wand 将 NumPy 数组转换为 Wand 图像对象
        with iproc.array_to_wand(src) as tmp:
            # 调用 _noise 函数添加噪声
            tmp = _noise(tmp, p_chroma, level)
            # 将 Wand 图像对象转换回 NumPy 数组
            dst = iproc.wand_to_array(tmp)
        return dst
    else:
        # 如果未添加噪声，直接返回原始图像
        return src

# 定义 scale 函数，用于对图像进行缩放和模糊处理
def scale(src, filters, bmin, bmax, scale):
    # 获取图像的高度和宽度
    h, w = src.shape[:2]
    # 随机生成模糊程度
    blur = np.random.uniform(bmin, bmax)
    # 随机选择一个滤波器
    rand = random.randint(0, len(filters) - 1)
    # 将 NumPy 数组转换为 Wand 图像对象
    with iproc.array_to_wand(src) as tmp:
        # 对图像进行下采样（缩小尺寸）
        tmp.resize(w // 2, h // 2, filters[rand], blur)
        # 如果 scale 参数为 True，则将图像上采样回原始尺寸
        if scale:
            tmp.resize(w, h, 'box')
        # 将 Wand 图像对象转换回 NumPy 数组
        dst = iproc.wand_to_array(tmp)
    return dst

# 定义 noise_scale 函数，结合噪声和缩放处理
def noise_scale(src, filters, bmin, bmax, scale, p, p_chroma, level):
    # 获取图像的高度和宽度
    h, w = src.shape[:2]
    # 随机生成模糊程度
    blur = np.random.uniform(bmin, bmax)
    # 随机选择一个滤波器
    rand = random.randint(0, len(filters) - 1)
    # 将 NumPy 数组转换为 Wand 图像对象
    with iproc.array_to_wand(src) as tmp:
        # 对图像进行下采样（缩小尺寸）
        tmp.resize(w // 2, h // 2, filters[rand], blur)
        # 以概率 p 决定是否对图像添加噪声
        if np.random.uniform() < p:
            tmp = _noise(tmp, p_chroma, level)
        # 如果 scale 参数为 True，则将图像上采样回原始尺寸
        if scale:
            tmp.resize(w, h, 'box')
        # 将 Wand 图像对象转换回 NumPy 数组
        dst = iproc.wand_to_array(tmp)
    return dst

# 定义 crop_if_large 函数，用于裁剪过大的图像
def crop_if_large(src, max_size):
    # 如果 max_size 大于 0，且图像的宽度和高度都大于 max_size，则进行裁剪
    if max_size > 0 and src.shape[1] > max_size and src.shape[0] > max_size:
        # 随机选择裁剪的起始点
        point_x = random.randint(0, src.shape[1] - max_size)
        point_y = random.randint(0, src.shape[0] - max_size)
        # 裁剪图像
        dst = src[point_y:point_y + max_size, point_x:point_x + max_size, :]
        return dst
    # 如果不需要裁剪，直接返回原图
    return src

# 定义 preprocess 函数，对图像进行预处理，包括数据增强
def preprocess(src, cfg):
    # 使用 data_augmentation 的 half 函数对图像进行随机裁剪或缩放
    dst = data_augmentation.half(src, cfg.random_half_rate)
    # 如果图像尺寸过大，则裁剪到最大尺寸
    dst = crop_if_large(dst, cfg.max_size)
    # 对图像进行随机翻转
    dst = data_augmentation.flip(dst)
    # 添加随机颜色噪声
    dst = data_augmentation.color_noise(dst, cfg.random_color_noise_rate)
    # 应用随机锐化滤波
    dst = data_augmentation.unsharp_mask(dst, cfg.random_unsharp_mask_rate)
    # 对图像进行 1 像素偏移
    dst = data_augmentation.shift_1px(dst)
    return dst

# 定义 active_cropping 函数，用于裁剪图像对（x 和 y），并根据条件选择最佳裁剪区域
def active_cropping(x, y, ly, size, scale, p, tries):
    # 检查裁剪尺寸是否满足缩放要求
    if size % scale != 0:
        raise ValueError('crop_size % scale must be 0')
    # 检查 x 和 y 的尺寸是否满足缩放关系
    if x.shape[0] * scale != y.shape[0] or x.shape[1] * scale != y.shape[1]:
        raise ValueError('Scaled shape must be equal ({}, {})'.format(
            x.shape[:1], y.shape[:1]))
    # 计算裁剪尺寸
    size_x = size // scale
    # 以概率 p 决定是否进行主动裁剪
    if np.random.uniform() < p:
        best_mse = 0  # 初始化最佳均方误差
        for i in range(tries):
            # 随机选择裁剪位置
            pw = random.randint(0, x.shape[1] - size_x) * scale
            ph = random.randint(0, x.shape[0] - size_x) * scale
            # 裁剪 x 和 ly
            crop_x = x[ph // scale:ph // scale + size_x,
                       pw // scale:pw // scale + size_x, :]
            crop_ly = ly[ph // scale:ph // scale + size_x,
                         pw // scale:pw // scale + size_x, :]
            # 计算均方误差
            mse = np.mean(np.square(crop_ly - crop_x))
            # 如果当前裁剪区域的均方误差更大，则更新最佳裁剪区域
            if mse >= best_mse:
                best_mse = mse
                best_cx = crop_x
                best_cy = y[ph:ph + size, pw:pw + size, :]
        return best_cx, best_cy
    else:
        # 如果不进行主动裁剪，则随机选择裁剪区域
        pw = random.randint(0, x.shape[1] - size_x) * scale
        ph = random.randint(0, x.shape[0] - size_x) * scale
        crop_x = x[ph // scale:ph // scale + size_x,
                   pw // scale:pw // scale + size_x, :]
        crop_y = y[ph:ph + size, pw:pw + size, :]
        return crop_x, crop_y

# 定义 pairwise_transform 函数，用于对图像对进行变换和裁剪
def pairwise_transform(src, cfg):
    # 定义不稳定区域的偏移量
    unstable_region_offset_x = 8
    unstable_region_offset_y = unstable_region_offset_x * cfg.inner_scale
    # 计算裁剪范围
    top = cfg.offset
    bottom = cfg.crop_size - top
    # 对原始图像进行预处理
    y = preprocess(src, cfg)

    # 根据配置选择不同的处理方法
    if cfg.method == 'scale':
        # 使用 scale 函数对图像进行缩放处理
        x = scale(
            y, cfg.downsampling_filters,
            cfg.resize_blur_min, cfg.resize_blur_max, cfg.inner_scale == 1)
    elif cfg.method == 'noise':
        # 使用 noise 函数对图像添加噪声
        if cfg.inner_scale != 1:
            raise ValueError('inner_scale must be 1')
        x = noise(y, cfg.nr_rate, cfg.chroma_subsampling_rate, cfg.noise_level)
    elif cfg.method == 'noise_scale':
        # 结合噪声和缩放处理
        if cfg.inner_scale == 1:
            raise ValueError('inner_scale must be > 1')
        x = noise_scale(
            y, cfg.downsampling_filters,
            cfg.resize_blur_min, cfg.resize_blur_max, False,
            cfg.nr_rate, cfg.chroma_subsampling_rate, cfg.noise_level)

    # 裁剪不稳定区域
    y = y[unstable_region_offset_y:y.shape[0] - unstable_region_offset_y,
          unstable_region_offset_y:y.shape[1] - unstable_region_offset_y]
    x = x[unstable_region_offset_x:x.shape[0] - unstable_region_offset_x,
          unstable_region_offset_x:x.shape[1] - unstable_region_offset_x]
    # 生成低分辨率版本的 y
    lowres_y = y.copy()
    if cfg.crop_size != cfg.in_size:
        lowres_y = iproc.nn_scaling(y, 1 / cfg.inner_scale)

    # 初始化存储裁剪块的数组
    patch_x = np.zeros(
        (cfg.patches, cfg.ch, cfg.in_size, cfg.in_size), dtype=np.uint8)
    patch_y = np.zeros(
        (cfg.patches, cfg.ch, cfg.out_size, cfg.out_size), dtype=np.uint8)

    # 对每个裁剪块进行处理
    for i in range(cfg.patches):
        # 使用 active_cropping 函数裁剪图像对
        crop_x, crop_y = active_cropping(
            x, y, lowres_y, cfg.crop_size, cfg.inner_scale,
            cfg.active_cropping_rate, cfg.active_cropping_tries)
        # 根据通道数处理裁剪块
        if cfg.ch == 1:
            # 转换为 YCbCr 格式并提取 Y 分量
            ycbcr_x = Image.fromarray(crop_x).convert('YCbCr')
            ycbcr_y = Image.fromarray(crop_y).convert('YCbCr')
            crop_x = np.array(ycbcr_x)[:, :, 0]
            crop_y = np.array(ycbcr_y)[top:bottom, top:bottom, 0]
            patch_x[i] = crop_x[np.newaxis, :, :]  # 添加通道维度
            patch_y[i] = crop_y[np.newaxis, :, :]
        elif cfg.ch == 3:
            # 裁剪 RGB 分量
            crop_y = crop_y[top:bottom, top:bottom, :]
            patch_x[i] = crop_x.transpose(2, 0, 1)  # 调整维度顺序
            patch_y[i] = crop_y.transpose(2, 0, 1)
    # 返回裁剪后的图像块
    return patch_x, patch_y