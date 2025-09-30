# LOGART: PUSHING THE LIMIT OF EFFICIENT LOGARITHMIC POST-TRAINING QUANTIZATION

This repository contains the official PyTorch implementation for the paper *"LogART: Pushing The Limit of Efficient Logarithmic Post-training Quantization"*.

## Get Started

1. **Clone this repository**:

   ```bash
   git clone https://github.com/logart-lab/logart.git
   ```

   Then navigate to the desired directory:

   ```bash
   cd LogART/LogART-LLM
   ```

   Or:

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

### LogART-LLM
Pretrained models for LogART-LLM can be obtained using the transformers library (version 4.43.2). Ensure transformers is installed:

```bash
pip install transformers==4.43.2
```

For dataset loading and preprocessing, this project uses the datasets library, tested on version 2.20.0. Install it with:

```bash
pip install datasets==2.20.0
```

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

### LogART-LLM
The code for LogART-LLM was modified based on [aespa](https://github.com/SamsungLabs/aespa). To quantize and evaluate the LLM, use the following command:

```bash
python main.py --model_path facebook/opt-125m --calib_data c4 --nsamples 32 --seqlen 2048 --seed 0 --w_bits 3 --num_iters 500 --lr 0.05 --w_method log_dynamic -scale_search --rrweight 1 [--hardware_approx] --learn_rounding [--test_with_hardware_approx]
    
```

#### Command-Line Arguments
- `--model_path`: Path to the LLM model. Choices: `facebook/opt-125m`, `facebook/opt-1.3b`, `facebook/opt-6.7b`, `meta-llama/Llama-2-7b`, `meta-llama/Llama-3.1-8B`.
- `--calib_data`: Name of the calibration dataset used for quantization.
- `--nsamples`: Number of samples selected from the calibration dataset for quantization.
- `--seqlen`: Sequence length of the input text processed by the model.
- `--seed`: Random seed for ensuring reproducibility of experiments.
- `--w_bits`: Bitwidth for weight quantization.
- `--num_iters`: Number of iterations in learnable rounding reconstruction.
- `--lr`: Adaptive rounding learning rate.
- `--w_method`: Quantization method. Choices: `log_2`, `log_sqrt2`, `log_dynamic`.
- `--scale_search`: Whether to search for the best scaling factor.
- `--rrweight`: Weight of rounding cost vs the reconstruction loss.
- `--hardware_approx`: Apply `1+1/2` hardware approximation method to $\sqrt{2}$.
- `--learn_rounding`: Whether to learn weight-rounding policy based on AdaRound.
- `--test_with_hardware_approx`: Whether to test the quantized model with hardware approximation. 

### LogART-CNN
The code for LogART-CNN was modified based on [BRECQ](https://github.com/yhhhli/BRECQ). To quantize and evaluate a single CNN model, use the following command:

```bash
python main.py --data_path DATA_DIR --arch resnet18 -seed 0 --n_bits_w 4 --channel_wise --scale_method log_dynamic --search_method layer_wise --search_samples 32 [--test_before_calibration] --iters_w 2000 --calib_lr 0.05 --weight 1 [--hardware_approx] [--test_with_hardware_approx]
 
```

#### Command-Line Arguments
- `--data_path`: Path to ImageNet data.
- `--arch`: Model architecture to use. Choices: `resnet18`, `resnet50`, `mobilenetv2`.
- `--seed`: Random seed for ensuring reproducibility of experiments.
- `--n_bits_w`: Bitwidth for weight quantization.
- `--channel_wise`: Enable channel-wise quantization for weights.
- `--scale_method`: Linear quantization method or logarithmic base. Choices: `log_2`, `log_sqrt2`, `log_dynamic`.
- `--search_method`: Dynamic base search method of the hyperparameter searching process for dynamic logarithmic quantization. Choices: 'tensor_wise', 'layer_wise'.
- `--search_samples`: Size of the hyperparameter searching dataset.
- `--test_before_calibration`: Test the quantization accuracy after hyperparameter searching and before reconstruction.
- `--iters_w`: Number of iterations in learnable rounding reconstruction.
- `--calib_lr`: Adaptive rounding learning rate.
- `--weight`: Weight of rounding cost vs the reconstruction loss.
- `--hardware_approx`: Apply hardware-level log $\sqrt{2}$ approximation to `1+1/2`.
- `--test_with_hardware_approx`: Whether to test the quantized model with hardware approximation. 

### LogART-ViT
The code for LogART-ViT was modified based on [APHQ](https://github.com/GoatWu/APHQ-ViT). To quantize and evaluate a single ViT model, use the following command:

```bash
python main.py --dataset DATA_DIR --model vit_base -seed 0 --config ./configs/4bit/best.py --iters_w 2000 --lr 0.05 --scale-method log_dynamic -search_method block_wise --rrweight 1 [--hardware_approx] [--optimize] [--test_with_hardware_approx]

```

#### Command-Line Arguments
- `--dataset`: Path to the dataset.
- `--model`: Model architecture. Choices: `vit_tiny`, `vit_small`, `vit_base`, `vit_large`, `deit_tiny`, `deit_small`, `deit_base`.
- `--seed`: Random seed for ensuring reproducibility of experiments.
- `--config`: File path to import the Config class.
- `--optimize`: Perform learnable rounding reconstruction to the model.
- `--iters_w`: Number of iterations in learnable rounding reconstruction.
- `--lr`: Adaptive rounding learning rate.
- `--scale-method`: Quantization method. Choices: `log_2`, `log_sqrt2`, `log_dynamic`.
- `--search_method`: Dynamic base search method of the hyperparameter searching process for dynamic logarithmic quantization. Choices: 'tensor_wise', 'block_wise'.
- `--rrweight`: Weight of rounding cost vs the reconstruction loss.
- `--hardware_approx`: Apply `1+1/2` hardware approximation method to $\sqrt{2}$.
- `--test_with_hardware_approx`: Whether to test the quantized model with hardware approximation. 

## Results
Results will be stored in `./results.csv`. The ablation results of LogART's key components on LLMs with 3-bit channel-wise weight quantization are shown in the table below:

<table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse; width: 100%; text-align: center; font-family: Arial, sans-serif;">
    <!-- 表头第1行：全局配置项 + 模型名称（跨3列，对应3个指标） -->
    <tr style="background-color: #f0f0f0; font-weight: bold;">
        <td rowspan="2">DBS</td>
        <td rowspan="2">SFS</td>
        <td rowspan="2">ABS</td>
        <td rowspan="2">LLR</td>
        <td rowspan="2">Calib. Data</td>
        <td colspan="3">OPT-125M</td>
        <td colspan="3">LLaMA2-7B</td>
    </tr>
    <!-- 表头第2行：模型对应的评估指标（PPL/Time/Memory） -->
    <tr style="background-color: #f0f0f0; font-weight: bold;">
        <td>PPL</td>
        <td>Time</td>
        <td>Memory</td>
        <td>PPL</td>
        <td>Time</td>
        <td>Memory</td>
    </tr>
    <!-- 数据行1 -->
    <tr>
        <td>×</td>
        <td>×</td>
        <td>×</td>
        <td>×</td>
        <td>-</td>
        <td>170.64</td>
        <td>0.7 s</td>
        <td>0.40 GB</td>
        <td>60.16</td>
        <td>13.0 s</td>
        <td>9.8 GB</td>
    </tr>
    <!-- 数据行2 -->
    <tr>
        <td>×</td>
        <td>×</td>
        <td>×</td>
        <td>✓</td>
        <td>32</td>
        <td>38.55</td>
        <td>61.3 s</td>
        <td>0.75 GB</td>
        <td>9.74</td>
        <td>58.6 min</td>
        <td>20.9 GB</td>
    </tr>
    <!-- 数据行3 -->
    <tr>
        <td>×</td>
        <td>×</td>
        <td>✓</td>
        <td>×</td>
        <td>-</td>
        <td>79.70</td>
        <td>0.7 s</td>
        <td>0.40 GB</td>
        <td>8.28</td>
        <td>13.2 s</td>
        <td>9.8 GB</td>
    </tr>
    <!-- 数据行4 -->
    <tr>
        <td>×</td>
        <td>×</td>
        <td>✓</td>
        <td>✓</td>
        <td>32</td>
        <td>36.39</td>
        <td>61.3 s</td>
        <td>0.75 GB</td>
        <td>9.16</td>
        <td>58.2 min</td>
        <td>20.9 GB</td>
    </tr>
    <!-- 数据行5 -->
    <tr>
        <td>×</td>
        <td>✓</td>
        <td>×</td>
        <td>×</td>
        <td>32</td>
        <td>38.41</td>
        <td>12.1 s</td>
        <td>0.75 GB</td>
        <td>6.66</td>
        <td>6.6 min</td>
        <td>20.9 GB</td>
    </tr>
    <!-- 数据行6 -->
    <tr>
        <td>×</td>
        <td>✓</td>
        <td>×</td>
        <td>✓</td>
        <td>32</td>
        <td>33.21</td>
        <td>64.6 s</td>
        <td>0.75 GB</td>
        <td>6.24</td>
        <td>63.1 min</td>
        <td>20.9 GB</td>
    </tr>
    <!-- 数据行7 -->
    <tr>
        <td>×</td>
        <td>✓</td>
        <td>✓</td>
        <td>×</td>
        <td>32</td>
        <td>35.15</td>
        <td>12.2 s</td>
        <td>0.75 GB</td>
        <td>6.55</td>
        <td>6.6 min</td>
        <td>20.9 GB</td>
    </tr>
    <!-- 数据行8 -->
    <tr>
        <td>×</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>32</td>
        <td>32.55</td>
        <td>64.6 s</td>
        <td>0.75 GB</td>
        <td>6.23</td>
        <td>63.5 min</td>
        <td>20.9 GB</td>
    </tr>
    <!-- 数据行9 -->
    <tr>
        <td>✓</td>
        <td>×</td>
        <td>×</td>
        <td>×</td>
        <td>32</td>
        <td>66.63</td>
        <td>3.8 s</td>
        <td>0.75 GB</td>
        <td>18.49</td>
        <td>83.2 s</td>
        <td>20.9 GB</td>
    </tr>
    <!-- 数据行10 -->
    <tr>
        <td>✓</td>
        <td>×</td>
        <td>×</td>
        <td>✓</td>
        <td>32</td>
        <td>35.46</td>
        <td>62.7 s</td>
        <td>0.75 GB</td>
        <td>9.26</td>
        <td>59.0 min</td>
        <td>20.9 GB</td>
    </tr>
    <!-- 数据行11 -->
    <tr>
        <td>✓</td>
        <td>×</td>
        <td>✓</td>
        <td>×</td>
        <td>32</td>
        <td>47.92</td>
        <td>3.8 s</td>
        <td>0.75 GB</td>
        <td>7.82</td>
        <td>82.2 s</td>
        <td>20.9 GB</td>
    </tr>
    <!-- 数据行12 -->
    <tr>
        <td>✓</td>
        <td>×</td>
        <td>✓</td>
        <td>✓</td>
        <td>32</td>
        <td>33.68</td>
        <td>62.9 s</td>
        <td>0.75 GB</td>
        <td>9.10</td>
        <td>59.1 min</td>
        <td>20.9 GB</td>
    </tr>
    <!-- 数据行13 -->
    <tr>
        <td>✓</td>
        <td>✓</td>
        <td>×</td>
        <td>×</td>
        <td>32</td>
        <td>36.10</td>
        <td>16.8 s</td>
        <td>0.75 GB</td>
        <td>6.56</td>
        <td>17.9 min</td>
        <td>20.9 GB</td>
    </tr>
    <!-- 数据行14 -->
    <tr>
        <td>✓</td>
        <td>✓</td>
        <td>×</td>
        <td>✓</td>
        <td>32</td>
        <td>32.37</td>
        <td>75.0 s</td>
        <td>0.75 GB</td>
        <td>6.19</td>
        <td>73.7 min</td>
        <td>20.9 GB</td>
    </tr>
    <!-- 数据行15 -->
    <tr>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>×</td>
        <td>32</td>
        <td>34.29</td>
        <td>17.0 s</td>
        <td>0.75 GB</td>
        <td>6.45</td>
        <td>17.9 min</td>
        <td>20.9 GB</td>
    </tr>
    <!-- 数据行16 -->
    <tr>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>32</td>
        <td>31.15</td>
        <td>75.1 s</td>
        <td>0.75 GB</td>
        <td>6.14</td>
        <td>74.2 min</td>
        <td>20.9 GB</td>
    </tr>
</table>
