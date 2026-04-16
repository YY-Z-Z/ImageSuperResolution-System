import os
import time
import argparse
import chainer
import six
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim  # 引入 SSIM 评估指标
from skimage.metrics import mean_squared_error as mse  # 引入 MSE 评估指标
import sys
sys.path.append('..')
from flask_app.lib import iproc, srcnn, pairwise_transform, utils, reconstruct
from flask_app.SR.pairwise_transform import noise, scale  # 导入 noise 和 scale 函数

# 定义去噪图像的函数
def denoise_image(cfg, src, model):
    dst = src.copy()  # 复制源图像
    six.print_('Level {} denoising...'.format(cfg.noise_level),  # 打印去噪级别
               end=' ', flush=True)
    if cfg.tta:  # 如果使用测试时增强（TTA）
        dst = reconstruct.image_tta(  # 使用TTA进行图像重建
            dst, model, cfg.tta_level, cfg.block_size, cfg.batch_size)
    else:
        dst = reconstruct.image(  # 不使用TTA进行图像重建
            dst, model, cfg.block_size, cfg.batch_size)
    if model.inner_scale != 1:  # 如果模型的内部缩放比例不为1
        dst = dst.resize((src.size[0], src.size[1]), Image.LANCZOS)  # 调整图像大小
    six.print_('OK')  # 打印完成信息
    return dst  # 返回去噪后的图像

# 定义放大图像的函数
def upscale_image(cfg, src, model):
    dst = src.copy()  # 复制源图像
    six.print_('2.0x scaling...', end=' ', flush=True)  # 打印放大信息
    if model.inner_scale == 1:  # 如果模型的内部缩放比例为1
        dst = iproc.nn_scaling(dst, 2)  # 使用最近邻算法进行2倍放大
    if cfg.tta:  # 如果使用测试时增强（TTA）
        dst = reconstruct.image_tta(  # 使用TTA进行图像重建
            dst, model, cfg.tta_level, cfg.block_size, cfg.batch_size)
    else:
        dst = reconstruct.image(  # 不使用TTA进行图像重建
            dst, model, cfg.block_size, cfg.batch_size)
    six.print_('OK')  # 打印完成信息
    return dst  # 返回放大后的图像

def load_image(image_path):
    """加载图像并转换为 NumPy 数组"""
    img = Image.open(image_path).convert('RGB')
    return np.array(img)

def save_image(image_array, output_path):
    """将 NumPy 数组保存为图像文件"""
    img = Image.fromarray(image_array)
    img.save(output_path)

# 定义加载模型的函数
def load_models(cfg):
    ch = 3 if cfg.color == 'rgb' else 1  # 根据颜色模式确定通道数
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件的绝对路径
    model_dir = os.path.join(current_dir, '..', 'SR', 'models', cfg.arch.lower())  # 模型目录路径

    models = {}  # 初始化模型字典
    if cfg.method == 'noise_scale':  # 如果方法为去噪和放大
        model_name = 'anime_style_noise{}_scale_{}.npz'.format(  # 模型文件名
            cfg.noise_level, cfg.color)
        model_path = os.path.join(model_dir, model_name)  # 模型文件路径
        if os.path.exists(model_path):  # 如果模型文件存在
            models['noise_scale'] = srcnn.archs[cfg.arch](ch)  # 加载模型
            chainer.serializers.load_npz(model_path, models['noise_scale'])
        else:  # 如果模型文件不存在
            model_name = 'anime_style_noise{}_{}.npz'.format(  # 去噪模型文件名
                cfg.noise_level, cfg.color)
            model_path = os.path.join(model_dir, model_name)  # 去噪模型文件路径
            models['noise'] = srcnn.archs[cfg.arch](ch)  # 加载去噪模型
            chainer.serializers.load_npz(model_path, models['noise'])
            model_name = 'anime_style_scale_{}.npz'.format(cfg.color)  # 放大模型文件名
            model_path = os.path.join(model_dir, model_name)  # 放大模型文件路径
            models['scale'] = srcnn.archs[cfg.arch](ch)  # 加载放大模型
            chainer.serializers.load_npz(model_path, models['scale'])
    if cfg.method == 'scale':  # 如果方法为放大
        model_name = 'anime_style_scale_{}.npz'.format(cfg.color)  # 模型文件名
        model_path = os.path.join(model_dir, model_name)  # 模型文件路径
        models['scale'] = srcnn.archs[cfg.arch](ch)  # 加载模型
        chainer.serializers.load_npz(model_path, models['scale'])

    if cfg.gpu >= 0:  # 如果使用GPU
        chainer.backends.cuda.check_cuda_available()  # 检查CUDA是否可用
        chainer.backends.cuda.get_device(cfg.gpu).use()  # 使用指定的GPU设备
        for _, model in models.items():  # 将模型转移到GPU
            model.to_gpu()
    return models  # 返回加载的模型

# 定义基准测试的函数
def benchmark(cfg, models, images, sampling_factor, quality):
    scores = []  # 初始化分数列表
    for src in images:  # 遍历图像列表
        # 使用 pairwise_transform 中的 noise 函数对图像加噪
        noisy_image = noise(np.array(src), p=1.0, p_chroma=0.5, level=cfg.noise_level)
        noisy_image = Image.fromarray(noisy_image)

        # 对加噪后的图像进行处理
        if 'noise_scale' in models:  # 如果存在去噪和放大模型
            dst = upscale_image(cfg, noisy_image, models['noise_scale'])  # 放大图像
        else:
            if 'noise' in models:  # 如果存在去噪模型
                dst = denoise_image(cfg, noisy_image, models['noise'])  # 去噪图像
            if 'scale' in models:  # 如果存在放大模型
                dst = upscale_image(cfg, noisy_image, models['scale'])  # 放大图像

        # 将处理后的图像与原始图像进行比较
        src = src.resize(dst.size, Image.LANCZOS)
        src_array = np.array(src)
        dst_array = np.array(dst)

        # 使用多个评估指标
        win_size = min(7, src_array.shape[0], src_array.shape[1])  # 确保 win_size 不超过图像尺寸
        ssim_score = ssim(src_array, dst_array, multichannel=True, data_range=255,win_size=win_size, channel_axis=2)
        mse_score = mse(src_array, dst_array)
        psnr_score = iproc.clipped_psnr(src_array, dst_array, a_max=255)

        # 综合评分（可以根据需要调整权重）
        combined_score = 0.1 * ssim_score + 0.7 * (1 - mse_score / 10000) + 0.2 * (psnr_score / 50)
        # combined_score=psnr_score
        scores.append(combined_score)  # 将分数添加到列表中
    return np.mean(scores), np.std(scores) / np.sqrt(len(scores))  # 返回平均分数和标准误差


current_progress = {
    'total_steps': 0,
    'completed_steps': 0,
    'current_model': None,
    'current_noise_level': None
}
def auto_select(image_path):
    # 进度追踪
    global current_progress
    # 初始化进度
    models_to_test = ['VGG7', 'ResNet10', 'UpConv7', 'UpResNet10']
    noise_levels_to_test = [0, 1, 2, 3]
    current_progress['total_steps'] = len(models_to_test) * len(noise_levels_to_test)
    current_progress['completed_steps'] = 0

    best_score = -1
    best_model = None
    best_noise_level = None

    # 加载图像
    img = Image.open(image_path).convert('RGB')
    w, h = img.size[:2]
    img = img.crop((0, 0, w - (w % 2), h - (h % 2)))  # 裁剪图像以确保尺寸为偶数
    images = [img]


    # 存储每个模型和噪声级别的综合分数
    results = []

    # 遍历模型和降噪程度
    for model_name in models_to_test:
        current_progress['current_model'] = model_name # 进度追踪
        six.print_('Model {} ---------\n'.format(model_name),end=' ', flush=True)
        for noise_level in noise_levels_to_test:
            current_progress['current_noise_level'] = noise_level
            # 设置参数
            args = argparse.Namespace(
                gpu=-1,  # 不使用 GPU
                input=image_path,
                arch=model_name,
                tta=False,
                batch_size=16,
                block_size=128,
                chroma_subsampling=False,
                downsampling_filter='box',
                method='noise_scale',
                noise_level=noise_level,
                color='rgb',
                tta_level=8
            )

            # 加载模型
            models = load_models(args)

            # 进行基准测试
            score, _ = benchmark(args, models, images, '1x1,1x1,1x1', 100)  # 使用最高质量 JPEG

            # 更新最佳模型和降噪程度
            if score > best_score:
                best_score = score
                best_model = model_name
                best_noise_level = noise_level

            # 记录结果
            results.append({
                'model': model_name,
                'noise_level': noise_level,
                'score': score
            })
            current_progress['completed_steps'] += 1

    current_progress['completed_steps'] = current_progress['total_steps']

    # 生成图表
    plt.figure(figsize=(10, 6))
    for model_name in models_to_test:
        scores = [result['score'] for result in results if result['model'] == model_name]
        plt.plot(noise_levels_to_test, scores, label=model_name)

    plt.xlabel('Noise Level')
    plt.ylabel('Combined Score')
    plt.title('Combined Scores for Different Models and Noise Levels')
    plt.legend()
    plt.grid(True)

    # 确保图表目录存在
    chart_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads', 'chart')
    os.makedirs(chart_dir, exist_ok=True)

    # 生成图表文件名（添加时间戳）
    timestamp = int(time.time())
    chart_filename = f'./static/uploads/chart/chart_{timestamp}.png'

    # 保存图表
    plt.savefig(chart_filename)
    plt.close()

    # 返回最佳模型、降噪级别和图表文件名
    return best_model, best_noise_level, best_score, chart_filename


if __name__ == '__main__':
    # 调用 auto_select 函数
    best_model, best_noise_level, best_score, chart_filename = auto_select('../SR/zready/dim/1_noisy1_small.jpg')
    print(f"Best Model: {best_model}, Best Noise Level: {best_noise_level}, Best Score: {best_score}")
    print(f"Chart saved to: {chart_filename}")



