# Creating Image Labels Robust to Adversarial Attacks

---

**Table of Contents**

  - [Project Description](#description)
  - [Repository Description](#repository)
  - [Example Commands](#commands)
  - [Results](#results)
  - [Citations](#citations)

---


## Project Description

This repository contains code and data for a project based on the Chen et al. paper, “[Beyond Categorical Label Representations for Image Classification](https://doi.org/10.48550/arXiv.2104.02226)” (2021). Beyond replicating the findings from the original paper that using high dimensional labels can improve a model’s robustness to adversarial attacks as well as its data efficiency, we explore possible explanations for these improvements. Following up on an idea from the paper, we investigate whether high entropy of certain label types may be responsible for their success, but find that this hypothesis is likely inaccurate. We also experiment with an alternative hypothesis, that the degree of overlap between the distributions of values in the high dimensional representations of label classes is responsible for improved robustness. We observe that less overlap results in more robustness and generally labels drawn from the uniform distribution perform better. The reasons behind the unintuitive influence of label representation on model robustness and efficiency remain unclear, and further research is required in order to better understand this phenomenon.

Models used: `VGG19`, `ResNet-32`, `ResNet-110`

Dataset: CIFAR-10

Code Source: [Label Representations](https://github.com/BoyuanChen/label_representations) (code in this repository is only *original code*; please see this repo for additional scripts used)

---

## Repository Description

The code in this repository is structured as follows:

+ `baseline_ResNet32.ipynb` - Recreation of the results of the Chen et al. [paper](https://doi.org/10.48550/arXiv.2104.02226) using ResNet-32. Also contains a data efficiency experiment with ResNet-110.
+ `experiments_ResNet32.ipynb` - Original experiments related to entropy and overlap using ResNet-32.
+ `PGD_ResNet32.ipynb` - PGD attacks against ResNet-32 models.
+ `baseline_vgg19.ipynb` - Recreation of the results of the Chen et al. [paper]((https://doi.org/10.48550/arXiv.2104.02226)) using VGG-19 model.
+ `experiments_vgg19.ipynb` - Contains code for performing `FGSM` (Targeted / Untargeted), `Iterative` (Targeted / Untargeted) and `PGD ` attacks on original labels, entropy experiment labels and overlap experiment labels. This notebook also contains code for the data efficiency experiments for models having the original labels from the [paper]((https://doi.org/10.48550/arXiv.2104.02226)).
+ `plots.ipynb` - Figure generation for presentation.
+ `labels` - Directory of labels used and code used to generate labels
+ `train_attack` - Directory of modified scripts from Chen et al. to run experiments. Please note we did not upload all dependencies, only scripts we modified. Please see Chen et al.'s original repository.
+ `utils` - [Directory](https://github.com/cdu890/robust_labels/tree/main/utils) of utility functions. Description for each function of the utils file is mentioned in the readme under `utils`.

---

## Example commands

Our repository is modularized into individual runnable `.py` files. Calling a command to do a specific task from a notebook is equivalent to simply invoking the `.py` file with correct parameters. Below is a list of frequently used commands:

+ **Training a model** : 

	Model training can be initiated by just calling the `train_attack/train_v2.py` file with the arguments specifying the model architecture and the label representation. Optional arguments for label and output locations can also be specified as follows:

	```sh
	~/$ python train_attack/train_v2.py --model vgg19 --dataset cifar10 --seed 7 --label normal_dim=5 --base_dir /project/outputs --label_dir /labels/label_files/entropy
	```

	For an example of how to run this command from a notebook, check `experiments_vgg19.ipynb` and `experiments_ResNet32.ipynb`

+ **Running FGSM, Iterative and PGD attacks** :

	Similar to the command above, the attacks can be performed on a saved model by invoking the `train_attack/attack_v3.py` file with the arguments specifying the output / model and label location. Example :

	```sh
	~/$ python train_attack/attack_v3.py --model vgg19 --dataset cifar10 --seed 7 --label category --base_dir /project/outputs --label_dir /labels/label_files/
	```

	This will run the attacks on the category label model. To see it's usage in a notebook, check `experiments_vgg19.ipynb` and `experiments_ResNet32.ipynb`

+ **Generating the labels used in our experiment** :

	To generate the labels used in our project - please run the following command in a machine with jupyter notebook server installed.

	```sh
	~/$ jupyter nbconvert --to notebook --inplace --execute labels/new_labels.ipynb
	```


## Results

### Baseline Accuracy

![plot](./figs/convergence.png)

+ High dimensional labels achieve a similar accuracy to categorical labels
+ High dimensional labels converge slower (especially Uniform label)


### Data Efficiency

![plot](./figs/data_eff.png)

+ Data efficiency improvements is network/depth-specific
+ VGG19: Improvement noticeable at < 4% of training data
+ ResNet-32: Little improvement at any subset of training data
+ ResNet-110: Improvement noticeable at < 10% of training data

### Model Robustness

![plot](./figs/orig_robust.png)

+ New labels are much more robust than categorical labels
+ Robustness is attack-specific and potentially network-specific
	+ Random labels: strong against untargeted attacks
	+ Composite labels: strong against untargeted attacks (VGG19)
	+ Lowdim labels: strong against most attacks especially targeted

### Investigating Entropy

![plot](./figs/entropy_robust.png)

+ Entropy is likely not the feature that confers robustness to labels

### Beyond Entropy: Overlap between Class Labels

![plot](./figs/gap_robust.png)

+ Larger gap resulted in more robustness in Iterative Targeted Attacks (ResNet-32)
+ No clear pattern for other attacks

## Citations

1. [**Beyond Categorical Label Representations for Image Classification**](https://doi.org/10.48550/arXiv.2104.02226); *International Conference on Learning Representations (ICLR 2021)*; Boyuan Chen, Yu Li, Sunand Raghupathi, Hod Lipson; 
2. [**Towards Deep Learning Models Resistant to Adversarial Attacks**](https://doi.org/10.48550/arXiv.1706.06083); *International Conference on Learning Representations (ICLR 2018)*; Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu
3. [**Adversarial Attacks and Defenses in Images, Graphs and Text: A Review**](https://doi.org/10.1007/s11633-019-1211-x); *International Journal of Automation and Computing*; Han Xu, Yao Ma, Hao-Chen Liu, Debayan Deb, Hui Liu, Ji-Liang Tang & Anil K. Jain
