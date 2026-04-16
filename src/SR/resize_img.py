from PIL import Image
import os


def resize_image(input_path, output_path):
    """
    将图像缩小到150x150像素。

    参数:
        input_path (str): 输入图像的路径。
        output_path (str): 输出图像的路径。
    """
    try:
        # 打开图像
        with Image.open(input_path) as img:
            # 获取原始图像的尺寸
            width, height = img.size

            # 计算新的尺寸（保持宽高比）
            new_width = 200
            new_height = 200

            # 缩放图像
            img_resized = img.resize((new_width, new_height), Image.LANCZOS)

            # 保存缩放后的图像
            img_resized.save(output_path)
            print(f"图像已成功缩放到 {new_width}x{new_height} 并保存到 {output_path}")
    except Exception as e:
        print(f"处理图像时出错: {e}")

def large_img(input_path, output_path):
    # 计算新的尺寸（保持宽高比）
    new_width = 300
    new_height = 300
    img=Image.open(input_path)
    # 缩放图像
    img_resized = img.resize((new_width, new_height), Image.LANCZOS)
    # 保存缩放后的图像
    img_resized.save(output_path)

def resize_all(orige_folder, small_folder):
    """
    批量处理orige文件夹中的所有图像，将它们缩小到200x200，并保存到small文件夹中。

    参数:
        orige_folder (str): 原始图像文件夹路径。
        small_folder (str): 缩小后图像文件夹路径。
    """
    # 确保输出文件夹存在
    if not os.path.exists(small_folder):
        os.makedirs(small_folder)

    # 遍历orige文件夹中的所有文件
    for filename in os.listdir(orige_folder):
        # 构造完整的文件路径
        input_path = os.path.join(orige_folder, filename)
        output_path = os.path.join(small_folder, filename)

        # 检查是否是文件
        if os.path.isfile(input_path):
            # 尝试处理图像
            try:
                resize_image(input_path, output_path)
            except Exception as e:
                print(f"处理文件 {input_path} 时出错: {e}")


# 示例用法
if __name__ == "__main__":
    input= "zready/orige/2.jpg"  # 替换为你的图像路径
    output1= "zready/small/2.jpg"  # 替换为你想保存的路径
    output2="zready/small/8_large.jpg"


    inputpath="zready/orige"
    outputpath="zready/small"
    resize_image(input, output1)
    # large_img(output1,output2)
    # resize_all(inputpath,outputpath)