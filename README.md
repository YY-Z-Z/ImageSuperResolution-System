# ImageSuperResolution-System
> 低清照片超高分辨率重建系统 —— 智能图像库管理与超分辨率重建一体化平台

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![Chainer](https://img.shields.io/badge/Chainer-7.0+-orange.svg)](https://chainer.org/)
[![MySQL](https://img.shields.io/badge/MySQL-8.0+-lightblue.svg)](https://www.mysql.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📌 项目简介

**ImageSuperResolution-System** 是一个全栈式图像处理平台，集成了智能图像库管理与前沿的超分辨率重建技术。用户可高效组织、分类和增强图像，系统提供自动标签分类、多种超分算法及参数智能推荐等功能。

### 核心功能

| 模块 | 功能描述 |
|------|----------|
| **图像库管理** | 创建/管理多个图像库，支持上传、删除、编辑、回收站，提供网格视图与标签视图两种展示模式 |
| **自动分类标签** | 基于 ResNet50 的图像分类算法，上传时自动检测并生成分类标签，用户亦可自定义编辑 |
| **超分辨率重建** | 集成 VGG7、UpConv7、ResNet10、UpResNet10 四种深度学习模型，支持自定义降噪程度、放大倍数、输出格式 |
| **最优参数推荐** | 基于多项评价指标，自动检测并推荐最适合上传图像的超分模型与参数组合 |
| **交互体验** | 按钮交互动画、进度条、过场动画等精心设计的视觉与交互效果 |

---
## 🏗️ 系统架构
![image.png](https://raw.gitcode.com/user-images/assets/9695586/841e02b6-f0da-4b5c-8311-cdc5172e062f/image.png 'image.png')

### 技术栈

- **前端**：HTML5 + CSS3 + JavaScript
- **后端**：Python + Flask 框架
- **数据库**：MySQL 
- **算法引擎**：Python + Chainer 深度学习框架
- **通信方式**：Fetch API 实现前后端数据交互

---

## 📂 项目目录结构
```text
ImageSuperResolution-System/
├── README.md             
│
├── paper/                   
│   └── 低清照片超高分辨率重建系统设计与开发.pdf
│
├── src/ # 源代码目录
│   ├── app.py  # Flask 应用主入口
│   ├── instance/  # 实例配置文件
│   ├── lib/   # 核心库与工具函数
│   ├── session_cache/  # 会话缓存
│   ├── SR/  # 超分模型
│   ├── static/  # 静态资源
│   └── templates/   # HTML
```

## 🧠 算法核心
### 图像分类算法（ResNet50）
上传图像时自动识别内容并生成分类标签，支持用户自定义编辑标签，便于图像的归类与检索。

### 超分辨率重建模型
| 模型 | 特点 |
| ---- | ---- |
| VGG7 | 轻量级结构，推理速度快，适用于通用图像 |
| UpConv7 | 基于转置卷积的上采样优化，适合平滑纹理 |
| ResNet10 | 残差学习框架，擅长恢复高频细节信息 |
| UpResNet10 | 残差块与上采样层的混合架构，综合性能最佳 |
### 最优参数推荐算法（Benchmark）
系统通过多项图像质量评价指标（PSNR、SSIM 等），自动分析上传图像特征，为用户推荐最优的超分模型和参数组合（降噪程度、放大倍数、输出格式），降低使用门槛。

## 🎨使用说明
1. 部署 MySQL 数据库并完成配置
2. 安装 Python 依赖环境
3. 运行 `src/app.py` 启动系统
4. 访问前端页面即可使用图像管理与超分重建功能



