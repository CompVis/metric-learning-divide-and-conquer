# Divide and Conquer the Embedding Space for Metric Learning

## About

This repository contains the code for reproducing the results for [Divide and Conquer the Embedding Space for Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sanakoyeu_Divide_and_Conquer_the_Embedding_Space_for_Metric_Learning_CVPR_2019_paper.pdf) (CVPR 2019) with the datasets [In-Shop Clothes](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html), [Stanford Online Products](http://cvgl.stanford.edu/projects/lifted_struct/) and [PKU VehicleID](https://www.pkuml.org/resources/pku-vehicleid.html).

**Paper**: [pdf](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sanakoyeu_Divide_and_Conquer_the_Embedding_Space_for_Metric_Learning_CVPR_2019_paper.pdf)  
**Supplementary**: [pdf](http://openaccess.thecvf.com/content_CVPR_2019/supplemental/Sanakoyeu_Divide_and_Conquer_CVPR_2019_supplemental.pdf)

We also applied our method to the [Humpback Whale Identification Challenge](https://www.kaggle.com/c/humpback-whale-identification/overview) at Kaggle and finished at 10th place out of 2131.  
**Slides**: [link](https://slides.com/asanakoy/metric-learning-kaggle-whales)

<img src="https://asanakoy.github.io/images/teaser_cvpr19_dml.jpg" width="480" alt="method pipeline">

## Requirements

- Python version 3.6.6 or higher
- SciPy and scikit-learn packages
- PyTorch ([pytorch.org](http://pytorch.org))
- Faiss with GPU support ([Faiss](https://github.com/facebookresearch/faiss))
- download and extract the datasets for [In-Shop Clothes](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html), [Stanford Online Products](http://cvgl.stanford.edu/projects/lifted_struct/) and [PKU VehicleID](https://www.pkuml.org/resources/pku-vehicleid.html)

## Usage

The following command will train the model with Margin loss on the In-Shop Clothes dataset for 200 epochs and a batch size of 80 while splitting the embedding layer with 8 clusters and finetuning the model from epoch 190 on. You can use this command to reproduce the results of the paper for the three datasets by changing simply `--dataset=inshop` to `--dataset=sop` (Stanford Online Products) or `--dataset=vid` (Vehicle-ID).

```
CUDA_VISIBLE_DEVICES=0 python experiment.py --dataset=inshop \
--dir=test --exp=0 --random-seed=0 --nb-clusters=8 --nb-epochs=200 \
--sz-batch=80 --backend=faiss-gpu  --embedding-lr=1e-5 --embedding-wd=1e-4 \
--backbone-lr=1e-5 --backbone-wd=1e-4 --finetune-epoch=190
```

The model can be trained without the proposed method by setting the number of clusters to 1 with `--nb-clusters=1`.  
For faster clustering we run Faiss on GPU. If you installed Faiss without GPU support use flag `--backend=faiss`.
## Expected Results

The model checkpoints and log files are saved in the selected log-directory. You can print a summary of the results with `python browse_results <log path>`.

You will get slightly higher results than what we have reported in the paper. For SOP, In-Shop and Vehicle-ID the R@1 results should be somewhat around 76.40, 87.36 and 91.54.

## Related Repos

- Collection of baselines for metric learning from @Confusezius [[PyTorch](https://github.com/Confusezius/Deep-Metric-Learning-Baselines)]

## License

You may find out more about the license [here](LICENSE)

## Reference

If you use this code, please cite the following paper:

Artsiom Sanakoyeu, Vadim Tschernezki, Uta Büchler, Björn Ommer. "Divide and Conquer the Embedding Space for Metric Learning", CVPR 2019.

```
@InProceedings{dcesml,
  title={Divide and Conquer the Embedding Space for Metric Learning},
  author={Sanakoyeu, Artsiom and Tschernezki, Vadim and B\"uchler, Uta and Ommer, Bj\"orn},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019},
}
```
