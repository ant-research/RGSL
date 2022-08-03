# RGSL (IJCAI22 paper)
Hongyuan Yu, Ting Li, Weichen Yu, "Regularized Graph Structure Learning with Explicit and Implicit Knowledge for Multi-variates Time-Series Forecasting" in Proc. 31st International Joint Conference on Artificial Intelligence (IJCAI-22) Main Track, July 23-29, 2022

This folder concludes the code and data of our RGSL model:

In this paper, we propose Regularized Graph Structure Learning (RGSL) model to incorporate both explicit prior structure and implicit structure together, and learn the forecasting deep networks along with the graph structure. RGSL consists of two innovative modules. First, we derive an implicit dense similarity matrix through node embedding, and learn the sparse graph structure using the Regularized Graph Generation (RGG) based on the Gumbel Softmax trick. Second, we propose a Laplacian Matrix Mixed-up Module (LM$^3$) to fuse the explicit graph and implicit graph together. We conduct experiments on three real-word datasets. Results show that the proposed RGSL model outperforms existing graph forecasting algorithms with a notable margin, while learning meaningful graph structure simultaneously.

<div align="center">
  <img src="demo/framework.png" width="800px" />
  <!-- <p>cell.</p> -->
</div>

## Structure:

* data: including PEMSD4 and PEMSD8 dataset used in our experiments.

* lib: contains self-defined modules for our work, such as data loading, data pre-process, normalization, and evaluate metrics.

* model: implementation of our RGSL model


## Get Started

1. Install Python 3.6, PyTorch 1.9.0.
2. Download data. You can obtain all the six benchmarks from [[Autoformer](https://github.com/thuml/Autoformer)] or [[Informer](https://github.com/zhouhaoyi/Informer2020)].
3. Train the model `python model/Run.py`.

## Requirements

Python 3.6.5, Pytorch 1.1.0, Numpy 1.16.3, argparse and configparser

To replicate the results in PEMSD4 and PEMSD8 datasets, you can run the the codes in the "model" folder directly. If you want to use the model for your own datasets, please load your dataset by revising "load_dataset" in the "lib" folder and remember tuning the learning rate (gradient norm can be used to facilitate the training).


## Citation

If you find this repo useful, please cite our paper. 

```
@inproceedings{ijcai2022-328,
  title     = {Regularized Graph Structure Learning with Semantic Knowledge for Multi-variates Time-Series Forecasting},
  author    = {Yu, Hongyuan and Li, Ting and Yu, Weichen and Li, Jianguo and Huang, Yan and Wang, Liang and Liu, Alex},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Lud De Raedt},
  pages     = {2362--2368},
  year      = {2022},
  month     = {7},
  note      = {Main Track}
  doi       = {10.24963/ijcai.2022/328},
  url       = {https://doi.org/10.24963/ijcai.2022/328},
}
```

## Results
<div align="center">
  <img src="demo/results.png" width="800px" />
  <!-- <p>cell.</p> -->
</div>


## Contact

If you have any question or want to use the code, please contact liting6259@gmail.com.

## Acknowledgement

We appreciate the following agcrn github repo for their valuable code base:

https://github.com/LeiBAI/AGCRN.git