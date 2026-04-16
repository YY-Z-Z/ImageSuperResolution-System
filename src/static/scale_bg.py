from PIL import Image, ImageFilter
import cv2
import numpy as np


def basic_enhancement(input_path, output_path):
    """基础版：双三次插值缩放+锐化处理"""
    # 打开图像
    img = Image.open(input_path)

    # 使用双三次插值法调整尺寸
    resized = img.resize((2880, 1536), Image.Resampling.BICUBIC)

    # 应用锐化滤镜（可调节强度）
    enhanced = resized.filter(ImageFilter.UnsharpMask(
        radius=2,  # 控制锐化范围（2-5）
        percent=150,  # 锐化强度（建议100-200）
        threshold=3  # 锐化阈值（3-10）
    ))

    # 保存结果
    enhanced.save(output_path)
    print("基础增强处理完成")



# 使用示例
if __name__ == "__main__":
    input_image = "./images/bg5.jpg"
    output_basic = "./images/bg5.jpg"

    # 基础版处理
    basic_enhancement(input_image, output_basic)

