# LogART: Learnable Logarithmic Adaptive Rounding Techniques for Post-Training Quantization

This repository contains the official PyTorch implementation for the paper *"LogART: Learnable Logarithmic Adaptive Rounding Techniques for Post-Training Quantization"*.

## Get Started

1. **Clone this repository**:

   ```bash
   git clone https://github.com/logart-lab/logart.git
   ```

   Then navigate to the desired directory:

   ```bash
   cd LogART/LogART-CNN
   ```

   Or:

   ```bash
   cd LogART/LogART-ViT
   ```

2. **Install PyTorch**:

   Ensure you have PyTorch installed. You can install PyTorch 1.10.0 with the following command:

   ```bash
   pip install torch==1.10.0 torchvision --index-url https://download.pytorch.org/whl/cu113
   ```

## Pretrained Models

### LogART-CNN
The pretrained models for LogART-CNN are sourced from [BRECQ](https://github.com/yhhhli/BRECQ) and can be accessed via `torch.hub`. For example, to load a pretrained ResNet-18 model:

```python
res18 = torch.hub.load('yhhhli/BRECQ', model='resnet18', pretrained=True)
```

If you encounter a `URLError` during download (likely due to network issues), you can manually download the model using `wget` and place it in the appropriate directory:

```bash
wget https://github.com/yhhhli/BRECQ/releases/download/v1.0/resnet50_imagenet.pth.tar
mv resnet50_imagenet.pth.tar ~/.cache/torch/checkpoints
```

The `load_state_dict_from_url` function will check the `~/.cache/torch/checkpoints` directory before attempting to download.

### LogART-ViT
Pretrained models for LogART-ViT can be obtained using the `timm` library (version 0.9.2). Ensure `timm` is installed:

```bash
pip install timm==0.9.2
```

Alternatively, you can directly download checkpoints provided by [AdaLog](https://github.com/GoatWu/AdaLog). For example:

```bash
wget https://github.com/GoatWu/AdaLog/releases/download/v1.0/deit_tiny_patch16_224.bin
mkdir -p ./checkpoint/vit_raw/
mv deit_tiny_patch16_224.bin ./checkpoint/vit_raw/
```

## Usage

### LogART-CNN
The code for LogART-CNN was modified based on [BRECQ](https://github.com/yhhhli/BRECQ). To quantize and evaluate a single CNN model, use the following command:

```bash
python main.py --data_path DATA_DIR --arch resnet18 --n_bits_w 4 [--channel_wise] --scale_method dynamic --search_method tensor_wise --search_samples 32 [--test_before_calibration] --iters_w 20000 --weight 10000 [--hardware_approx] 
    
```

#### Command-Line Arguments
- `--data_path`: Path to ImageNet data.
- `--arch`: Model architecture to use. Choices: `resnet18`, `resnet50`, `mobilenetv2`.
- `--n_bits_w`: Bitwidth for weight quantization.
- `--channel_wise`: Enable channel-wise quantization for weights.
- `--scale_method`: Linear quantization method or logarithmic base. Choices: `linear_mse`, `linear_minmax`, `log_2`, `log_sqrt2`, `dynamic`.
- `--search_method`: Dynamic base search method of the hyperparameter searching process for dynamic logarithmic quantization.
- `--search_samples`: Size of the hyperparameter searching dataset.
- `--test_before_calibration`: Test the quantization accuracy after hyperparameter searching and before reconstruction.
- `--iters_w`: Number of iterations in learnable rounding reconstruction.
- `--rrweight`: Weight of rounding cost vs the reconstruction loss.
- `--hardware_approx`: Apply hardware-level log \sqrt{2} approximation to `1+1/2`.

### LogART-ViT
The code for LogART-ViT was modified based on [APHQ](https://github.com/GoatWu/APHQ-ViT). To quantize and evaluate a single ViT model, use the following command:

```bash
python main.py --dataset DATA_DIR --model vit_base --config ./configs/4bit/best.py --iters_w 20000 --scale-method dynamic --rrweight 10000 [--hardware_approx][--optimize]
    
```

#### Command-Line Arguments
- `--dataset`: Path to the dataset.
- `--model`: Model architecture. Choices: `vit_tiny`, `vit_small`, `vit_base`, `vit_large`, `deit_tiny`, `deit_small`, `deit_base`.
- `--config`: File path to import the Config class.
- `--optimize`: Perform learnable rounding reconstruction to the model.
- `--iters_w`: Number of iterations in learnable rounding reconstruction.
- `--scale_method`: Quantization method. Choices: `linear_mse`, `linear_minmax`, `log_2`, `log_sqrt2`, `dynamic`.
- `--rrweight`: Weight of rounding cost vs the reconstruction loss.
- `--hardware_approx`: Apply `1+1/2` hardware approximation method to \sqrt{2}.

## Results
Results will be stored in `./results.csv`. The ablation results of LogART's key components on CNNs with 4-bit channel-wise weight quantization are shown in the table below:

| Base  | DBS | LLR | DBS Calib. Data | LLR Calib. Data | ResNet18 Acc(%) | ResNet50 Acc(%) | MobileNetV2 Acc(%) |
|-------|-----|-----|-----------------|-----------------|-----------------|-----------------|--------------------|
| 2     | -   | ×   | -               | -               | 31.53           | 42.76           | 1.22               |
| √2    | -   | ×   | -               | -               | 3.48            | 22.34           | 0.94               |
| DLog  | TW  | ×   | 32              | -               | 57.10           | 61.53           | 15.84              |
| DLog  | LW  | ×   | 32              | -               | 67.07           | 74.12           | 66.05              |
| DLog  | BW  | ×   | 32              | -               | 64.41           | 73.68           | 64.92              |
| 2     | -   | ✓   | -               | 1024            | 69.86           | 75.90           | 69.03              |
| √2    | -   | ✓   | -               | 1024            | 63.27           | 72.00           | 70.20              |
| DLog  | TW  | ✓   | 32              | 1024            | 69.24           | 76.44           | 71.55              |
| DLog  | LW  | ✓   | 32              | 1024            | 70.33           | 76.16           | 71.03              |
| DLog  | BW  | ✓   | 32              | 1024            | 70.29           | 76.17           | 70.56              |