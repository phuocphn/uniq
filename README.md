# UniQ 

This repo contains the code and data of the following paper:

**Training Multi-bit Quantized and Binarized Networks with A Learnable Symmetric Quantizer**


## Prerequisites
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the library dependencies


```bash
pip install -r requirements.txt
```

## Training

```bash
export CUDA_VISIBLE_DEVICES=[GPU_IDs] && \
python main.py --train_id [training_id] \
--lr [learning_rate_value] --wd [weight_decay_value] --batch-size [batch_size] \
--dataset [dataset_name] --arch [architecture_name] \
--bit [bit-width] --epoch [training_epochs] \
--data_root [path_to_dataset] \
--init_from [path_to_pretrained_model] \
--train_scheme uniq --quant_mode [quantization_mode] \
--num_calibration_batches [number_of_batches_for_initialization] 
```



## Testing

```bash
export CUDA_VISIBLE_DEVICES=[GPU_IDs] && \
python main.py --train_id [training_id] \
--batch-size [batch_size] \
--dataset [dataset_name] --arch [architecture_name] \
--bit [bit-width] 
--data_root [path_to_dataset] \
--init_from [path_to_trained_model] \
--train_scheme uniq --quant_mode [quantization_mode] \
-e
```


| Arguments  | Description |
| ------------- | ------------- |
| `--train_id`  | ID for experiment management (arbitrary).   |
| `--lr`  | Learning rate   |
| `--wd`  | Weight decay  |
| `--batch_size`  | Batch size  |
| `--dataset`  | Dataset name <br/> Possible values: `cifar100`, `imagenet`   |
| `--data_root`  | Path to the dataset directory  |
| `--arch`  | Architecture name <br/> Possible values: `presnet18`, `presnet32`, `glouncv-presnet34`, `glouncv-mobilenetv2_w1`   |
| `--bit`  | Bit-width (W/A)  |
| `--epoch`  | Number of training epochs  |
| `--init_from`  | Path to the pretrained model.  |
| `--train_scheme`  | Training scheme <br/> Possible values: `fp32` (normal training), `uniq` (low-bit quantization training)  |
| `--quant_mode`  | Quantization mode <br/> Possible values: `layer_wise` (layer-wise quantization), `kernel-wise` (kernel-wise quantization)  |
| `--num_calibration_batches`  | Number of batches used for initialization |


For each experiment details and hyperparameter setting, we refer the readers to the paper and `main.py` file.

##Citation
If you find RBNN useful in your research, please consider citing:
```
@ARTICLE{9383003,
  author={P. {Pham} and J. A. {Abraham} and J. {Chung}},
  journal={IEEE Access}, 
  title={Training Multi-Bit Quantized and Binarized Networks with a Learnable Symmetric Quantizer}, 
  year={2021},
  volume={9},
  number={},
  pages={47194-47203},
  doi={10.1109/ACCESS.2021.3067889}}
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
