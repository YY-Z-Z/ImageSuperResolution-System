import os
import numpy as np
from PIL import Image
from pairwise_transform import _noise, scale, iproc


def load_image(image_path):
    """加载图像并转换为NumPy数组"""
    img = Image.open(image_path)
    img = img.convert('RGB')  # 确保图像是RGB格式
    return np.array(img)


def save_image(image_array, output_path):
    """将NumPy数组保存为图像文件"""
    img = Image.fromarray(image_array)
    img.save(output_path)


def generate_noisy_images(input_image_path, output_dir, noise_levels, filters, bmin, bmax, target_size=(150, 150)):
    """生成不同噪声级别的图像并保存到输出目录"""
    # 加载输入图像
    src = load_image(input_image_path)

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取原图像名（不带扩展名）
    image_name = os.path.splitext(os.path.basename(input_image_path))[0]

    # 将图像缩小到目标尺寸
    img = Image.fromarray(src)
    img = img.resize(target_size, Image.LANCZOS)  # 使用Lanczos插值进行缩小
    scaled_image_array = np.array(img)

    # 保存缩小后的图像
    scaled_output_path = os.path.join(output_dir, f"{image_name}_scaled.jpg")
    save_image(scaled_image_array, scaled_output_path)
    print(f"Saved scaled image to {scaled_output_path}")

    # 对缩小后的图像进行模糊处理
    for level in noise_levels:
        # 将缩小后的图像转换为Wand图像对象
        with iproc.array_to_wand(scaled_image_array) as tmp:
            # 添加噪声（这里可以替换为模糊处理）
            noisy_image = _noise(tmp, p=1.0, level=level)
            # 将Wand图像对象转换回NumPy数组
            noisy_image_array = iproc.wand_to_array(noisy_image)

        # 保存图像
        output_path = os.path.join(output_dir, f"{image_name}_noisy{level}_small.jpg")
        save_image(noisy_image_array, output_path)
        print(f"Saved noisy scaled image (level {level}) to {output_path}")


if __name__ == "__main__":
    input_image_path = "zready/orige/17.jpg"  # 输入图像路径
    output_dir = "zready/dim"  # 输出目录
    noise_levels = [0, 1, 2, 3]  # 不同的噪声级别

    # 模糊处理参数
    filters = ['box', 'lanczos', 'hamming']  # 缩小图像时使用的滤波器
    bmin = 0.5  # 模糊程度的最小值
    bmax = 1.5  # 模糊程度的最大值

    # 目标尺寸
    target_size = (150, 150)

    generate_noisy_images(input_image_path, output_dir, noise_levels, filters, bmin, bmax, target_size)