# 导入__future__模块中的division，确保除法运算结果为浮点数
from __future__ import division
# 导入argparse模块，用于解析命令行参数
import argparse
# 导入os模块，用于操作系统级的文件和目录操作
import os
# 导入time模块，用于获取当前时间等
import time

# 导入Chainer框架，用于构建和训练神经网络模型
import chainer
# 导入numpy模块，用于进行高效的数值计算
import numpy as np
# 导入PIL库中的Image模块，用于图像处理
from PIL import Image
# 导入six模块，用于Python 2和3之间的兼容性
import six

# 导入自定义的模块，包含图像处理相关的函数和类
from lib import iproc
from lib import reconstruct
from lib import srcnn
from lib import utils

'''
功能：对图像进行去噪处理。
参数：
cfg：配置对象，包含去噪相关的参数。
src：输入图像。
model：去噪模型。
过程：
分离图像的 alpha 通道。
使用模型进行去噪处理。
如果模型的缩放比例不为 1，则对图像进行缩放。
如果存在 alpha 通道，则将其合并回图像中。
'''
# 定义denoise_image函数，用于对图像进行去噪处理
def denoise_image(cfg, src, model):
    # 调用split_alpha函数分离图像的RGB和alpha通道
    dst, alpha = split_alpha(src, model)
    # 打印去噪级别信息
    six.print_('Level {} denoising...'.format(cfg.noise_level),
               end=' ', flush=True)
    # 如果配置中启用了tta（Test Time Augmentation）
    if cfg.tta:
        # 使用tta方式进行去噪处理
        dst = reconstruct.image_tta(
            dst, model, cfg.tta_level, cfg.block_size, cfg.batch_size)
    else:
        # 使用普通方式进行去噪处理
        dst = reconstruct.image(dst, model, cfg.block_size, cfg.batch_size)
    # 如果模型的缩放比例不为1，则对图像进行缩放
    if model.inner_scale != 1:
        dst = dst.resize((src.size[0], src.size[1]), Image.LANCZOS)
    # 打印去噪完成信息
    six.print_('OK')
    # 如果存在alpha通道，则将其合并回图像中
    if alpha is not None:
        dst.putalpha(alpha)
    # 返回去噪后的图像
    return dst


'''
功能：对图像进行放大处理。
参数：
cfg：配置对象，包含放大相关的参数。
src：输入图像。
scale_model：放大模型。
alpha_model：用于alpha通道的放大模型（可选）。
过程：
分离图像的alpha通道。
进行多次2倍放大处理，直到达到目标缩放比例。
如果目标尺寸与放大后的尺寸不匹配，则进行最终的缩放。
如果存在alpha通道，则将其合并回图像中。
'''
# 定义upscale_image函数，用于对图像进行放大处理
def upscale_image(cfg, src, scale_model, alpha_model=None):
    # 调用split_alpha函数分离图像的RGB和alpha通道
    dst, alpha = split_alpha(src, scale_model)
    # 计算需要进行多少次2倍放大处理
    for i in range(int(np.ceil(np.log2(cfg.scale_ratio)))):
        # 打印2倍放大信息
        six.print_('2.0x scaling...', end=' ', flush=True)
        # 根据当前放大次数和是否提供alpha模型选择使用哪个模型
        model = scale_model if i == 0 or alpha_model is None else alpha_model
        # 如果模型的缩放比例为1，则使用最近邻插值进行2倍放大
        if model.inner_scale == 1:
            dst = iproc.nn_scaling(dst, 2)  # Nearest neighbor 2x scaling
            alpha = iproc.nn_scaling(alpha, 2)  # Nearest neighbor 2x scaling
        # 如果配置中启用了tta（Test Time Augmentation）
        if cfg.tta:
            # 使用tta方式进行放大处理
            dst = reconstruct.image_tta(
                dst, model, cfg.tta_level, cfg.block_size, cfg.batch_size)
        else:
            # 使用普通方式进行放大处理
            dst = reconstruct.image(dst, model, cfg.block_size, cfg.batch_size)
        # 如果没有提供alpha模型，则使用scale_model对alpha通道进行放大
        if alpha_model is None:
            alpha = reconstruct.image(
                alpha, scale_model, cfg.block_size, cfg.batch_size)
        else:
            # 否则使用alpha_model对alpha通道进行放大
            alpha = reconstruct.image(
                alpha, alpha_model, cfg.block_size, cfg.batch_size)
        # 打印放大完成信息
        six.print_('OK')
    # 计算目标尺寸
    dst_w = int(np.round(src.size[0] * cfg.scale_ratio))
    dst_h = int(np.round(src.size[1] * cfg.scale_ratio))
    # 如果目标尺寸与放大后的尺寸不匹配，则进行最终的缩放
    if dst_w != dst.size[0] or dst_h != dst.size[1]:
        six.print_('Resizing...', end=' ', flush=True)
        dst = dst.resize((dst_w, dst_h), Image.LANCZOS)
        six.print_('OK')
    # 如果存在alpha通道，则将其合并回图像中
    if alpha is not None:
        if alpha.size[0] != dst_w or alpha.size[1] != dst_h:
            alpha = alpha.resize((dst_w, dst_h), Image.LANCZOS)
        dst.putalpha(alpha)
    # 返回放大后的图像
    return dst


'''
功能：分离图像的alpha通道。
参数：
src：输入图像。
model：模型对象。
返回值：
rgb：RGB图像。
alpha：alpha通道图像（如果存在）。
'''
# 定义split_alpha函数，用于分离图像的alpha通道
def split_alpha(src, model):
    # 初始化alpha通道为None
    alpha = None
    # 如果输入图像的模式为L、RGB或P，并且包含透明度信息，则转换为RGBA模式
    if src.mode in ('L', 'RGB', 'P'):
        if isinstance(src.info.get('transparency'), bytes):
            src = src.convert('RGBA')
    # 将图像转换为RGB模式
    rgb = src.convert('RGB')
    # 如果输入图像的模式为LA或RGBA，则分离alpha通道
    if src.mode in ('LA', 'RGBA'):
        six.print_('Splitting alpha channel...', end=' ', flush=True)
        alpha = src.split()[-1]
        # 调用iproc.alpha_make_border函数为RGB图像添加alpha通道边界
        rgb = iproc.alpha_make_border(rgb, alpha, model)
        six.print_('OK')
    # 返回RGB图像和alpha通道图像
    return rgb, alpha


'''
功能：加载所需的模型。
参数：
cfg：配置对象，包含模型相关的参数。
返回值：
models：包含加载的模型的字典。
过程：
根据配置加载去噪模型、放大模型和alpha通道模型。
如果配置中指定了GPU，则将模型移动到GPU上。
'''
# 定义load_models函数，用于加载所需的模型
def load_models(cfg):
    # 根据配置确定通道数（3表示RGB，1表示灰度）
    ch = 3 if cfg.color == 'rgb' else 1
    # 根据配置确定模型目录
    if cfg.model_dir is None:
        model_dir = os.path.join('models', cfg.arch.lower())  # 使用 os.path.join 拼接路径
    else:
        model_dir = cfg.model_dir

    # 初始化模型字典
    models = {}
    # 初始化标志变量
    flag = False
    # 如果配置的方法为noise_scale
    if cfg.method == 'noise_scale':
        # 构建去噪放大模型的文件名
        model_name = 'anime_style_noise{}_scale_{}.npz'.format(
            cfg.noise_level, cfg.color)
        model_path = os.path.join(model_dir, model_name)  # 使用 os.path.join 拼接路径
        # 如果模型文件存在，则加载模型
        if os.path.exists(model_path):
            models['noise_scale'] = srcnn.archs[cfg.arch](ch)
            chainer.serializers.load_npz(model_path, models['noise_scale'])
            # 构建alpha通道放大模型的文件名
            alpha_model_name = 'anime_style_scale_{}.npz'.format(cfg.color)
            alpha_model_path = os.path.join(model_dir, alpha_model_name)  # 使用 os.path.join 拼接路径
            # 加载alpha通道放大模型
            models['alpha'] = srcnn.archs[cfg.arch](ch)
            chainer.serializers.load_npz(alpha_model_path, models['alpha'])
        else:
            # 如果模型文件不存在，则设置标志变量为True
            flag = True
    # 如果配置的方法为scale或标志变量为True
    if cfg.method == 'scale' or flag:
        # 构建放大模型的文件名
        model_name = 'anime_style_scale_{}.npz'.format(cfg.color)
        model_path = os.path.join(model_dir, model_name)  # 使用 os.path.join 拼接路径
        # 加载放大模型
        models['scale'] = srcnn.archs[cfg.arch](ch)
        chainer.serializers.load_npz(model_path, models['scale'])
    # 如果配置的方法为noise或标志变量为True
    if cfg.method == 'noise' or flag:
        # 构建去噪模型的文件名
        model_name = 'anime_style_noise{}_{}.npz'.format(
            cfg.noise_level, cfg.color)
        model_path = os.path.join(model_dir, model_name)  # 使用 os.path.join 拼接路径
        # 如果模型文件不存在，则尝试加载去噪放大模型
        if not os.path.exists(model_path):
            model_name = 'anime_style_noise{}_scale_{}.npz'.format(
                cfg.noise_level, cfg.color)
            model_path = os.path.join(model_dir, model_name)  # 使用 os.path.join 拼接路径
        # 加载去噪模型
        models['noise'] = srcnn.archs[cfg.arch](ch)
        chainer.serializers.load_npz(model_path, models['noise'])

    # 如果配置中指定了GPU，则将模型移动到GPU上
    if cfg.gpu >= 0:
        chainer.backends.cuda.check_cuda_available()
        chainer.backends.cuda.get_device(cfg.gpu).use()
        for _, model in models.items():
            model.to_gpu()
    # 返回加载的模型字典
    return models


'''
功能：处理命令行参数并执行图像处理。
过程：
解析命令行参数。
加载所需的模型。
处理输入文件列表。
对每个输入文件进行去噪和放大处理。
保存处理后的图像。
'''
# 定义处理命令行参数并执行图像处理的函数
def main():
    # 创建argparse解析器对象
    p = argparse.ArgumentParser(description='Chainer implementation of waifu2x')
    # 添加命令行参数
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--input', '-i', default='images/small.png')
    p.add_argument('--output', '-o', default='./')
    p.add_argument('--quality', '-q', type=int, default=None)
    p.add_argument('--model_dir', '-d', default='SR/models/vgg7')
    p.add_argument('--scale_ratio', '-s', type=float, default=2.0)  # 图像缩放比例
    p.add_argument('--tta', '-t', action='store_true')
    p.add_argument('--batch_size', '-b', type=int, default=16)
    p.add_argument('--block_size', '-l', type=int, default=128)
    p.add_argument('--extension', '-e', default='png',
                   choices=['png', 'webp'])  # 输出文件扩展名
    p.add_argument('--arch', '-a', default='VGG7',
                   choices=['VGG7', '0', 'UpConv7', '1', 'ResNet10', '2', 'UpResNet10', '3'])  # 使用的模型架构
    p.add_argument('--method', '-m', default='scale',
                   choices=['noise', 'scale', 'noise_scale'])  # 处理方法
    p.add_argument('--noise_level', '-n', type=int, default=1,
                   choices=[0, 1, 2, 3])
    p.add_argument('--color', '-c', default='rgb',
                   choices=['y', 'rgb'])
    p.add_argument('--tta_level', '-T', type=int, default=8,
                   choices=[2, 4, 8])
    g = p.add_mutually_exclusive_group()
    g.add_argument('--width', '-W', type=int, default=0)
    g.add_argument('--height', '-H', type=int, default=0)
    g.add_argument('--shorter_side', '-S', type=int, default=0)
    g.add_argument('--longer_side', '-L', type=int, default=0)

    # 解析命令行参数
    args = p.parse_args()
    # 如果配置的架构在srcnn.table中，则进行转换
    if args.arch in srcnn.table:
        args.arch = srcnn.table[args.arch]

    # 加载所需的模型
    models = load_models(args)

    # 定义输入和输出文件的扩展名列表
    input_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp']
    output_exts = ['.png', '.webp']
    # 获取输出文件的扩展名
    outext = '.' + args.extension

    # 初始化输出文件名和目录
    outname = None
    outdir = args.output
    # 如果输入为目录，则加载目录中的文件列表
    if os.path.isdir(args.input):
        filelist = utils.load_filelist(args.input)
    else:
        # 如果输入为文件，则处理文件名和目录
        tmpname, tmpext = os.path.splitext(os.path.basename(args.output))
        if tmpext in output_exts:
            outext = tmpext
            outname = tmpname
            outdir = os.path.dirname(args.output)
            outdir = './' if outdir == '' else outdir
        elif tmpext != '':
            raise ValueError('Format {} is not supported'.format(tmpext))
        filelist = [args.input]

    # 如果输出目录不存在，则创建目录
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # 对每个输入文件进行处理
    for path in filelist:
        # 获取文件名和扩展名
        tmpname, tmpext = os.path.splitext(os.path.basename(path))
        if outname is None or len(filelist) > 1:
            outname = tmpname
        # 构建输出文件的路径
        outpath = os.path.join(outdir, '{}{}'.format(outname, outext))
        # 如果文件扩展名在输入扩展名列表中
        if tmpext.lower() in input_exts:
            # 打开输入图像
            src = Image.open(path)
            # 获取图像的宽度和高度
            w, h = src.size[:2]
            # 根据配置计算缩放比例
            if args.width != 0:
                args.scale_ratio = args.width / w
            elif args.height != 0:
                args.scale_ratio = args.height / h
            elif args.shorter_side != 0:
                if w < h:
                    args.scale_ratio = args.shorter_side / w
                else:
                    args.scale_ratio = args.shorter_side / h
            elif args.longer_side != 0:
                if w > h:
                    args.scale_ratio = args.longer_side / w
                else:
                    args.scale_ratio = args.longer_side / h

            # 复制输入图像
            dst = src.copy()
            # 记录开始时间
            start = time.time()
            # 构建输出文件名
            outname += '_(tta{})'.format(args.tta_level) if args.tta else '_'
            # 根据配置进行去噪和放大处理
            if 'noise_scale' in models:
                outname += '(noise{}_scale{:.1f}x)'.format(
                    args.noise_level, args.scale_ratio)
                dst = upscale_image(
                    args, dst, models['noise_scale'], models['alpha'])
            else:
                if 'noise' in models:
                    outname += '(noise{})'.format(args.noise_level)
                    dst = denoise_image(args, dst, models['noise'])
                if 'scale' in models:
                    outname += '(scale{:.1f}x)'.format(args.scale_ratio)
                    dst = upscale_image(args, dst, models['scale'])
            # 打印处理时间
            print('Elapsed time: {:.6f} sec'.format(time.time() - start))

            # 构建最终的输出文件名
            outname += '({}_{}){}'.format(args.arch, args.color, outext)
            # 如果输出文件已存在，则更新输出文件路径
            if os.path.exists(outpath):
                outpath = os.path.join(outdir, outname)

            # 获取是否无损保存的标志和质量参数
            lossless = args.quality is None
            quality = 100 if lossless else args.quality
            # 获取ICC配置文件
            icc_profile = src.info.get('icc_profile')
            icc_profile = "" if icc_profile is None else icc_profile
            # 保存处理后的图像
            dst.convert(src.mode).save(
                outpath, quality=quality, lossless=lossless,
                icc_profile=icc_profile)
            # 打印保存信息
            print(f"Output file saved to: {outpath}")



if __name__ == '__main__':
    main()