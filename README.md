# Bare Soil Detector (裸土检测器)

This project uses the Ultralytics YOLO framework to train a model for detecting exposed or bare soil in images, often found on construction sites.

这个项目使用 Ultralytics YOLO 框架来训练一个模型，用于检测图像中（通常是工地上）的裸露土壤。

## Features 功能

- Trains a YOLO model to identify 'soil' and 'cover' classes.
  训练一个 YOLO 模型来识别 'soil'（裸土）和 'cover'（覆盖物）类别。
- Provides scripts for training (`train.py`), testing (`test.py`), and a class for easy integration (`SoilDetector` in `soil_detector.py`).
  提供用于训练 (`train.py`)、测试 (`test.py`) 的脚本，以及一个易于集成的类 (`SoilDetector` in `soil_detector.py`)。
- Saves detection results with bounding boxes overlaid on the original images.
  将带有边界框的检测结果保存在原始图片上。

## Requirements 依赖

- Python 3.x
- Ultralytics YOLO: `pip install ultralytics`
- (Optional) A CUDA-enabled GPU or Apple Silicon for faster training/inference.
  （可选）支持 CUDA 的 GPU 或 Apple Silicon 以加速训练/推理。

## Setup 设置

1.  **Clone the repository:**
    **克隆仓库:**

    ```bash
    git clone git@github.com:bochili/bare-soil-detector.git
    cd bare-soil-detector
    ```

2.  **Install dependencies:**
    **安装依赖:**

    ```bash
    pip install ultralytics
    ```

3.  **(Optional) Prepare your dataset:**
    **(可选) 准备你的数据集:**
    - Organize your images and labels in the YOLO format.
      按照 YOLO 格式组织你的图片和标签。
    - Update the `train:` and `val:` paths in `soil.yaml` to point to your dataset's image directories. **Important:** Use relative paths or ensure absolute paths are correct for your environment.
      更新 `soil.yaml` 中的 `train:` 和 `val:` 路径，使其指向你的数据集图片目录。**重要提示：** 使用相对路径或确保绝对路径在你的环境中是正确的。
    - Verify the class names in `soil.yaml` match your dataset.
      确认 `soil.yaml` 中的类别名称与你的数据集一致。

## Usage 使用

### Training 训练

- Modify `soil.yaml` to point to your dataset (if not using the example paths).
  修改 `soil.yaml` 指向你的数据集（如果不使用示例路径）。
- Run the training script. Adjust `epochs` and `device` as needed.
  运行训练脚本。根据需要调整 `epochs` 和 `device`。

  ```bash
  # For Apple Silicon (M1/M2/M3...)
  # 适用于 Apple Silicon (M1/M2/M3...)
  python train.py

  # Or for CPU/CUDA (modify train.py to uncomment the relevant line)
  # 或者适用于 CPU/CUDA (修改 train.py 取消相关行的注释)
  # python train.py
  ```

- The trained model (`best.pt`) and results will be saved in the `runs/detect/train*` directory (or similar, depending on YOLO version) by default, or potentially in `train_result/` if training was run previously with that output structure.
  训练好的模型 (`best.pt`) 和结果默认会保存在 `runs/detect/train*` 目录（或类似目录，取决于 YOLO 版本），或者如果之前训练时是那样的输出结构，则可能在 `train_result/` 目录下。

### Detection 检测

**Method 1: Using `test.py` (Detects all images in a folder)**
**方法一：使用 `test.py`（检测文件夹中所有图片）**

- Place your images in the `test_imgs/` folder (or modify the path in `test.py`).
  将你的图片放入 `test_imgs/` 文件夹（或修改 `test.py` 中的路径）。
- Ensure the model path in `test.py` (`train_result/weights/best.pt`) is correct.
  确保 `test.py` 中的模型路径 (`train_result/weights/best.pt`) 正确。
- Run the script. Results will be saved in `runs/detect/predict*`.
  运行脚本。结果将保存在 `runs/detect/predict*` 目录。

  ```bash
  python test.py
  ```

**Method 2: Using `SoilDetector` class**
**方法二：使用 `SoilDetector` 类**

- See the example usage at the bottom of `soil_detector.py`.
  参考 `soil_detector.py` 文件底部的示例用法。
- You can integrate the `SoilDetector` class into your own applications.
  你可以将 `SoilDetector` 类集成到你自己的应用程序中。

  ```python
  from soil_detector import SoilDetector

  # Initialize detector (adjust model path and output dir if needed)
  # 初始化检测器（如果需要，调整模型路径和输出目录）
  detector = SoilDetector(model_path="train_result/weights/best.pt", conf_threshold=0.3, output_dir="output")

  # List of image paths to detect
  # 需要检测的图片路径列表
  image_paths = ["test_imgs/1.jpg", "test_imgs/another_image.png"]

  # Perform detection
  # 执行检测
  detection_results = detector.detect(image_paths)

  # Process results
  # 处理结果
  for result in detection_results:
      print(f"Image: {result['image_path']}")
      if result["has_soil"]:
          print(f"  Detected bare soil with confidence: {result['soil_confidence']:.2f}")
      else:
          print("  No bare soil detected.")
      print(f"  Result saved to: {result['result_path']}")
      print("-" * 20)

  ```

- Results (images and detection info) will be saved in the specified `output_dir` (default: `output/`).
  结果（图片和检测信息）将保存在指定的 `output_dir`（默认：`output/`）中。

## Dataset 数据集

The dataset configuration is defined in `soil.yaml`. It expects images in the specified `train` and `val` directories, with corresponding YOLO format label files (`.txt`).

数据集配置在 `soil.yaml` 中定义。它期望在指定的 `train` 和 `val` 目录中有图片，并附带相应的 YOLO 格式标签文件 (`.txt`)。

Classes / 类别:

- `cover`: Covered areas (e.g., with nets, tarps) / 覆盖区域（例如，有网、篷布）
- `soil`: Bare soil / 裸露土壤

## Example Results 示例结果

The `train_result/` directory contains example outputs from a previous training run, including:
`train_result/` 目录包含一次先前训练运行的示例输出，包括：

- `weights/best.pt`: The best trained model weights. / 最佳训练模型权重。
- Various performance plots (`*.png`). / 各种性能图表 (`*.png`)。
- Validation batch predictions (`val_batch*.jpg`). / 验证批次的预测结果 (`val_batch*.jpg`)。
- `results.csv`: Training metrics per epoch. / 每个周期的训练指标。
