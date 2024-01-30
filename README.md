# SSR : SAM is a Strong Regularizer for domain adaptive semantic segmentation&#x20;

## Overview&#x20;

We introduced SSR, which utilizes SAM (segment-anything) as a strong regularizer during training, to greatly enhance the robustness of the image encoder for handling various domains. Specifically, given the fact that SAM is pre-trained with a large number of images over the internet, which cover a diverse variety of domains, the feature encoding extracted by the SAM is obviously less dependent on specific domains when compared to the traditional ImageNet pre-trained image encoder. Meanwhile, the ImageNet pre-trained image encoder is still a mature choice of backbone for the semantic segmentation task, especially when the SAM is category-irrelevant. As a result, our SSR provides a simple yet highly effective design. It uses the ImageNet pre-trained image encoder as the backbone, and the intermediate feature of each stage (\ie there are 4 stages in MiT-B5) is regularized by SAM during training. After extensive experimentation on GTA5→Cityscapes, our SSR significantly improved performance over the baseline without introducing any extra inference overhead.

## Setup Environment&#x20;

1\. Please follow the instruction in [MIC](https://github.com/lhoyer/MIC/tree/master/seg) to install and use this repo.&#x20;

2\. Install Segment Anything:&#x20;

```Shell
pip install git+https://github.com/facebookresearch/segment-anything.git 
```

## Training

    python run_experiments.py --config configs/sam/daformer_sam_vit_b.py

## Acknowledgements &#x20;

SSR is based on the [MIC](https://github.com/lhoyer/MIC/tree/master/seg) project. We thank their authors for making the source code publicly available.&#x20;

## Citation&#x20;

If you use this code for your research, please cite our paper:&#x20;

    @misc{ge2024ssr,
          title={SSR: SAM is a Strong Regularizer for domain adaptive semantic segmentation}, 
          author={Yanqi Ge and Ye Huang and Wen Li and Lixin Duan},
          year={2024},
          eprint={2401.14686},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
    }

