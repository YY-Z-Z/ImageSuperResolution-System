import matplotlib.pyplot as plt
import os
import matplotlib

# 设置 Matplotlib 后端以确保图形可以正确显示和保存
matplotlib.use('TkAgg')

def plot_performance_charts():
    # 创建一个包含模型名称和对应指标值的字典
    results = {
        "VGG7": {"ssim": 0.923, "psnr": 31.2, "time": 0.42},
        "Upconv7": {"ssim": 0.885, "psnr": 29.1, "time": 0.20},
        "ResNet10": {"ssim": 0.912, "psnr": 30.5, "time": 0.29},
        "UpResNet10": {"ssim": 0.884, "psnr": 28.7, "time": 0.14}
    }

    models = list(results.keys())
    ssim_values = [results[model]['ssim'] for model in models]
    psnr_values = [results[model]['psnr'] for model in models]
    time_values = [results[model]['time'] for model in models]

    # 绘制图表
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # SSIM 图表
    axs[0].bar(models, ssim_values, color='skyblue')
    axs[0].set_title('SSIM Scores')
    axs[0].set_ylabel('SSIM')
    axs[0].grid(True, axis='y')

    # PSNR 图表
    axs[1].bar(models, psnr_values, color='lightgreen')
    axs[1].set_title('PSNR Scores')
    axs[1].set_ylabel('PSNR (dB)')
    axs[1].grid(True, axis='y')

    # 运行时间图表
    axs[2].bar(models, time_values, color='salmon')
    axs[2].set_title('Average Processing Time')
    axs[2].set_ylabel('Time (seconds)')
    axs[2].grid(True, axis='y')

    # 保存图表
    chart_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(chart_dir, exist_ok=True)
    chart_filename = os.path.join(chart_dir, 'model_performance.png')
    plt.tight_layout()
    plt.savefig(chart_filename)

    # 显示图表
    plt.show()

    # 关闭图形
    plt.close()

    print(f"Charts saved to: {chart_filename}")

if __name__ == '__main__':
    plot_performance_charts()