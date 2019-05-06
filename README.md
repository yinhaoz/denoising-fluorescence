# [Fluorescence Microscopy Denoising (FMD) dataset](https://drive.google.com/drive/folders/1aygMzSDdoq63IqSk-ly8cMq0_owup8UM) 

Code for CVPR 2019 paper "A Poisson-Gaussian Denoising Dataset with Real Fluorescence Microscopy
Images", [arXiv.1812.10366](https://arxiv.org/abs/1812.10366).

```latex
@inproceedings{zhang2018poisson,
    title={A Poisson-Gaussian Denoising Dataset with Real Fluorescence Microscopy Images},
    author={Yide Zhang and Yinhao Zhu and Evan Nichols and Qingfei Wang and Siyuan Zhang and Cody Smith and Scott Howard},
    booktitle={CVPR},
    year={2019}
}
```
```
git clone https://github.com/bmmi/denoising-fluorescence.git
cd denoising-fluorescence/denoising
```
### Dependency
- Python 3
- PyTorch 1.0
- skimage
- MATLAB

## FMD dataset

![](denoising/images/fmdd_teaser.png?raw=true)

To download the whole dataset at once
```
bash download_dataset.sh
```
To download dataset separately according to the microscope used
```
bash download_dataset.sh confocal
```
Change `confocal` to `twophoton` or `widefield` to download other categories.

## Benchmark


### Deep learning models
Download the FMD dataset into the default directory `denoising/dataset/`.
#### Train Noise2Noise model
```
python train_n2n.py
```
If your dataset is not within the default directory, you need to set `--data-root path_to_dataset`, where your downloaded dataset is under `path_to_dataset`.
Try Noise2Noise model with BatchNorm using additional argument `--net unetv2`. It is more stable across different learning rate, but no denoising performance improvement if learning rate is well tuned. Experiment results are saved in `./experiments/`.

#### Train DnCNN model
```
python train_dncnn.py
```
Try DnCNN with non-residual learning using additional argument `--net dncnn_nrl`. It is worse than the residual learning.
#### Benchmark with pretrained models
Download the pre-trained models in the dataset folder on google drive.
```
bash download_pretrained.sh
```
The pretrained model are saved in `./experiments/pretrained/`.

Benchmark with the pretrained Noise2Noise model
```
python benchmark.py --model n2n
```
Use `--model dncnn` to benchmark with pretrained DnCNN model. GPU is used by default if it is available. Results are saved in `./experiments/pretrained/n2n/benchmark_gpu/`. To run on CPU, use `--no-cuda`.

Reproduce test example in Fig 6 & 7 in the paper
```
python test_example.py
```

### Traditional denosing methods
Download the FMD dataset into the default directory `denoising/dataset/`.
```
cd matlab
```
In the benchmark files (e.g., `benchmark_VST_NLM.m`), assign different folder names (e.g., `Confocal_BPAE_B`) to the variable `data_name` to benchmark different data groups. Execute the benchmark files to start benchmarking.

For more details regarding the traditional denoising methods, please refer to the following references.

#### Noise Estimation
- A. Foi, M. Trimeche, V. Katkovnik, and K. Egiazarian. "Practical Poissonian-Gaussian noise modeling and fitting for single-image raw-data". IEEE Transactions on Image Processing, 17(10):1737–1754, 2008.

#### VST
-  M. Makitalo and A. Foi. "A closed-form approximation of the exact unbiased inverse of the Anscombe variance-stabilizing transformation". IEEE Transactions on Image Processing, 20(9):2697–2698, 2011.
- M. Makitalo and A.Foi. "Optimal inversion of the generalized Anscombe transformation for Poisson-Gaussian noise". IEEE Transactions on Image Processing, 22(1):91–103, 2013.

#### VST+NLM
- A. Buades, B. Coll, and J.-M. Morel. "A non-local algorithm for image denoising". In CVPR, 2005.
- B. K. Shreyamsha Kumar, “Image Denoising based on Non Local-means Filter and its Method Noise Thresholding”. Signal, Image and Video Processing, 7(6):1211-1227, 2013.


#### VST+BM3D
- K. Dabov, A. Foi, V. Katkovnik, and K. Egiazarian. "Image denoising by sparse 3-D transform-domain collaborative filtering". IEEE Transactions on Image Processing, 16(8):2080–2095, 2007.

#### VST+KSVD
-  M. Aharon, M. Elad, and A. Bruckstein. "K-SVD: An algorithm for designing overcomplete dictionaries for sparse representation". IEEE Transactions on Signal Processing, 54(11):4311–4322, 2006.

#### VST+EPLL
-  D. Zoran and Y. Weiss. "From learning models of natural image patches to whole image restoration". In ICCV, 2011.

#### VST+WNNM
- S. Gu, L. Zhang, W. Zuo, and X. Feng. "Weighted nuclear norm minimization with application to image denoising". In CVPR, 2014.

#### PURE-LET
-  F. Luisier, T. Blu, and M. Unser. "Image denoising in mixed PoissonGaussian noise". IEEE Transactions on Image Processing, 20(3):696–708, 2011.
