import os
import time
import argparse
import chainer
import six
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import sys
sys.path.append('..')
from flask_app.lib import iproc, srcnn, pairwise_transform, utils, reconstruct
from flask_app.SR.pairwise_transform import noise, scale

def denoise_image(cfg, src, model):
    dst = src.copy()
    six.print_('Level {} denoising...'.format(cfg.noise_level),
               end=' ', flush=True)
    if cfg.tta:
        dst = reconstruct.image_tta(
            dst, model, cfg.tta_level, cfg.block_size, cfg.batch_size)
    else:
        dst = reconstruct.image(
            dst, model, cfg.block_size, cfg.batch_size)
    if model.inner_scale != 1:
        dst = dst.resize((src.size[0], src.size[1]), Image.LANCZOS)
    six.print_('OK')
    return dst

def upscale_image(cfg, src, model):
    dst = src.copy()
    six.print_('2.0x scaling...', end=' ', flush=True)
    if model.inner_scale == 1:
        dst = iproc.nn_scaling(dst, 2)
    if cfg.tta:
        dst = reconstruct.image_tta(
            dst, model, cfg.tta_level, cfg.block_size, cfg.batch_size)
    else:
        dst = reconstruct.image(
            dst, model, cfg.block_size, cfg.batch_size)
    six.print_('OK')
    return dst

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    return img

def load_models(cfg):
    ch = 3 if cfg.color == 'rgb' else 1
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, '..', 'SR', 'models', cfg.arch.lower())

    models = {}
    if cfg.method == 'noise_scale':
        model_name = 'anime_style_noise{}_scale_{}.npz'.format(
            cfg.noise_level, cfg.color)
        model_path = os.path.join(model_dir, model_name)
        if os.path.exists(model_path):
            models['noise_scale'] = srcnn.archs[cfg.arch](ch)
            chainer.serializers.load_npz(model_path, models['noise_scale'])
        else:
            model_name = 'anime_style_noise{}_{}.npz'.format(
                cfg.noise_level, cfg.color)
            model_path = os.path.join(model_dir, model_name)
            models['noise'] = srcnn.archs[cfg.arch](ch)
            chainer.serializers.load_npz(model_path, models['noise'])
            model_name = 'anime_style_scale_{}.npz'.format(cfg.color)
            model_path = os.path.join(model_dir, model_name)
            models['scale'] = srcnn.archs[cfg.arch](ch)
            chainer.serializers.load_npz(model_path, models['scale'])
    if cfg.method == 'scale':
        model_name = 'anime_style_scale_{}.npz'.format(cfg.color)
        model_path = os.path.join(model_dir, model_name)
        models['scale'] = srcnn.archs[cfg.arch](ch)
        chainer.serializers.load_npz(model_path, models['scale'])

    if cfg.gpu >= 0:
        chainer.backends.cuda.check_cuda_available()
        chainer.backends.cuda.get_device(cfg.gpu).use()
        for _, model in models.items():
            model.to_gpu()
    return models

def benchmark_image_processing(cfg, model, image):
    # 加噪
    noisy_image = noise(np.array(image), p=1.0, p_chroma=0.5, level=cfg.noise_level)
    noisy_image = Image.fromarray(noisy_image)

    # 去噪和放大
    start_time = time.time()
    if 'noise_scale' in model:
        dst = upscale_image(cfg, noisy_image, model['noise_scale'])
    else:
        if 'noise' in model:
            dst = denoise_image(cfg, noisy_image, model['noise'])
        if 'scale' in model:
            dst = upscale_image(cfg, dst, model['scale'])
    end_time = time.time()

    # 计算指标
    src = image.resize(dst.size, Image.LANCZOS)
    src_array = np.array(src)
    dst_array = np.array(dst)

    win_size = min(7, src_array.shape[0], src_array.shape[1])
    ssim_score = ssim(src_array, dst_array, multichannel=True, data_range=255, win_size=win_size, channel_axis=2)
    psnr_score = iproc.clipped_psnr(src_array, dst_array, a_max=255)

    return ssim_score, psnr_score, end_time - start_time

def test_models():
    # 模型名称列表
    model_names = ['VGG7', 'ResNet10', 'UpConv7', 'UpResNet10']
    noise_level = 2  # 只测试 noise=2 的情况

    # 初始化结果存储
    results = {model: {'ssim': [], 'psnr': [], 'time': []} for model in model_names}

    # 加载测试集
    test_dir = r'D:\MyWorkSpace\BiShe\Mycode\flask_app\SR\anime-data\anime-data\anime-faces\anime-faces'
    images = []
    for filename in os.listdir(test_dir):
        if len(images) >= 1000:  # 限制为前1000张图像
            break
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(test_dir, filename)
            img = load_image(img_path)
            images.append(img)

    # 遍历模型进行测试
    for model_name in model_names:
        print(model_name+" star---------------------------")
        # 设置参数
        args = argparse.Namespace(
            gpu=-1,
            input=None,
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

        # 测试每张图像
        total_ssim = 0
        total_psnr = 0
        total_time = 0
        for img in images:
            ssim_score, psnr_score, process_time = benchmark_image_processing(args, models, img)
            total_ssim += ssim_score
            total_psnr += psnr_score
            total_time += process_time

        # 计算平均值
        avg_ssim = total_ssim / len(images)
        avg_psnr = total_psnr / len(images)
        avg_time = total_time / len(images)

        # 保存结果
        results[model_name]['ssim'] = avg_ssim
        results[model_name]['psnr'] = avg_psnr
        results[model_name]['time'] = avg_time

        print(f"Model: {model_name}, Noise Level: {noise_level}")
        print(f"Average SSIM: {avg_ssim:.4f}, Average PSNR: {avg_psnr:.2f} dB, Average Time: {avg_time:.4f}s")

    # 绘制图表
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # SSIM 图表
    models = list(results.keys())
    ssim_values = [results[model]['ssim'] for model in models]
    axs[0].bar(models, ssim_values)
    axs[0].set_title('SSIM Scores')
    axs[0].set_ylabel('SSIM')
    axs[0].grid(True)

    # PSNR 图表
    psnr_values = [results[model]['psnr'] for model in models]
    axs[1].bar(models, psnr_values)
    axs[1].set_title('PSNR Scores')
    axs[1].set_ylabel('PSNR (dB)')
    axs[1].grid(True)

    # 运行时间图表
    time_values = [results[model]['time'] for model in models]
    axs[2].bar(models, time_values)
    axs[2].set_title('Average Processing Time')
    axs[2].set_ylabel('Time (seconds)')
    axs[2].grid(True)

    # 保存图表
    chart_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(chart_dir, exist_ok=True)
    chart_filename = os.path.join(chart_dir, 'model_performance_noise2.png')
    plt.tight_layout()
    plt.savefig(chart_filename)
    plt.close()

    print(f"Charts saved to: {chart_filename}")

if __name__ == '__main__':
    test_models()