# ImageNet-100 (IN100) PyTorch Implementation

PyTorch Implementation: Training ResNets on ImageNet-100 data

## Prepare Datasets (ImageNet)
ImageNet-1K data could be accessed with [ILSVRC 2012](http://www.image-net.org/challenges/LSVRC/2012/). If ImageNet-1K data is available already, jump to the Quick Start section below to generate ImageNet-100.

```
root
├── data
│   ├── imagenet
│   │   ├── train
│   │   ├── val
│   ├── imagenet-100 (would be generated, no need to specify)
│   │   ├── train
│   │   ├── val

```


## Quick Start

Generate ImageNet-100 dataset based on [selected class file](https://arxiv.org/pdf/1906.05849.pdf) randomly sampled from ImageNet-1K dataset. Simply run the generate_IN100.py could generate folder of ImageNet-100.

For example, run the following command to generate ImageNet-100 from ImageNet-1K data.

arguments:
  - `--source_folder`: specify the ImageNet-1K data folder (e.g., `/root/data/imagenet/train`)
  - `--target_folder`: specify the ImageNet-100 data folder (e.g., `/root/data/imagenet-100/train`)
  - `--target_class`: specify the ImageNet-100 txt file with list of classes [default: 'IN100.txt']

```
python generate_IN100.py \
  --source_folder /path/to/ImageNet-1K data
  --target_folder /path/to/ImageNet-100 data
```

Note: Replace `train` with `val` to generate ImageNet-100 val data as well

Corruption Benchmarks:

```
python generate_IN100.py \
  --source_folder /path/to/ImageNet-1K-c-(bar) data
  --target_folder /path/to/ImageNet-100-c-(bar) data
```

This lets you create the same class IN-100-c and IN-100-c-bar dataset. Those are the corrupted version of the testset, with 19/10 different corruptions and 5 severity levels each, that have to be looped over. Obtain information towards the corruption benchmarks from:
ImagenNet-c: https://github.com/hendrycks/robustness
ImageNet-c-bar: https://github.com/facebookresearch/augmentation-corruption

## Training ResNets on ImageNet-100

The implementation of training and validation code can be used in main_IN100.py, and run it for the usage.

```
python main_IN100.py --model resnet18 \
  --data_folder /path/to/ImageNet-100 main folder \
  --batch_size 256 \
  --epochs 200 \
  --learning_rate 0.2 \
  --cosine \
```
Note: Please set up the augment: `--data_folder` as main path (e.g., `/root/data/imagenet-100`) to ImageNet-100.

Cosine annealing schedule is applied as defult in the implementation (remove `--cosine` to switch to step learning schedule).

## Results
Experiments on ImageNet-100:
| Arch | Batch Size | Epoch | Loss | kNN Accuracy(%) |
|:----:|:---:|:---:|:---:|:---:|
| ResNet18 | 256 | 200 | Cross Entropy |  -  |
| ResNet50 | 256 | 200 | Cross Entropy |  -  |

## Citation

If you use this toolbox in your work, please cite this project.

```bibteX
@misc{imagenet100pytorch,
    title={{IN100pytorch}: PyTorch Implementation: Training ResNets on ImageNet-100},
    author={Chun-Hsiao Yeh, Yubei Chen},
    howpublished={\url{https://github.com/danielchyeh/ImageNet-100-Pytorch}},
    year={2022}
}
```

## Acknowledgements

Part of this code is based on [HobbitLong/SupContrast](https://github.com/HobbitLong/SupContrast).
