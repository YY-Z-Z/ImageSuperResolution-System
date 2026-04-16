import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import tensorflow as tf

def classify_and_plot_probability_distribution(image_path, temperature=1.0):
    # 加载ResNet50模型
    model = ResNet50(weights='imagenet')

    # 加载图像
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # 预测
    predictions = model.predict(img_array)
    predictions = tf.nn.softmax(predictions / temperature).numpy()  # 应用温度参数平滑概率分布

    decoded_predictions = decode_predictions(predictions, top=5)[0]

    # 提取标签和概率
    labels = [pred[1] for pred in decoded_predictions]
    probabilities = [pred[2] for pred in decoded_predictions]

    # 绘制概率分布图
    plt.figure(figsize=(10, 6))
    plt.barh(labels, probabilities, color='skyblue')
    plt.xlabel('Probability')
    plt.title('Probability Distribution')
    plt.gca().invert_yaxis()  # 反转y轴，使最高概率在顶部
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    image_path = "img.jpg"
    temperature = 1.5  # 温度参数，可以调整这个值来改变概率分布的平滑程度
    classify_and_plot_probability_distribution(image_path, temperature)

# import numpy as np
# import matplotlib.pyplot as plt
#
# # 定义标签和概率
# labels = ['volcano', 'fountain', 'jellyfish', 'promontory', 'cliff']
# probabilities = [0.6, 0.2, 0.1, 0.06, 0.04]
#
# # 将标签和概率按概率降序排列
# sorted_items = sorted(zip(labels, probabilities), key=lambda x: x[1], reverse=True)
# sorted_labels = [item[0] for item in sorted_items]
# sorted_probabilities = [item[1] for item in sorted_items]
#
# # 设置横坐标的刻度
# x_ticks = np.arange(0, 0.71, 0.1)
#
# # 绘制概率分布图
# plt.figure(figsize=(10, 6))
# bars = plt.barh(sorted_labels, sorted_probabilities, color='skyblue', height=0.6)
#
# # 设置横坐标的刻度
# plt.xticks(x_ticks)
#
# # 设置标题和标签
# plt.xlabel('Probability', fontsize=12, fontfamily='sans-serif')
# plt.gca().invert_yaxis()
# plt.title('Probability Distribution', fontsize=14, fontfamily='sans-serif')
#
# # 添加网格线
# plt.grid(axis='x', linestyle='--', alpha=0.7)
#
# # 添加数据标签
# for bar in bars:
#     width = bar.get_width()
#     plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
#              ha='left', va='center', fontsize=10, fontfamily='sans-serif')
#
# # 调整布局
# plt.tight_layout()
#
# # 显示图表
# plt.show()