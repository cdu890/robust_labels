from __future__ import absolute_import, division, print_function, unicode_literals

import architecture
import argparse
import cifar
import os
import torch
import torch.nn as nn
from pathlib import Path

# Params setup
parser = argparse.ArgumentParser(description="CIFAR High-dimensional Model.")
parser.add_argument(
    "--label", type=str, help="Label in [speech, uniform, shuffle, composite, random]"
)
parser.add_argument(
    "--model", type=str, help="Image encoder in [vgg19, resnet110, resnet32]"
)
parser.add_argument(
    "--num_classes", type=int, help="Number of target classes (10 or 100)."
)
parser.add_argument("--seed", type=int, help="Manual seed.", required=True)
parser.add_argument(
    "--data_dir",
    type=str,
    help="Directory where CIFAR datasets are stored",
    default="./data",
)
parser.add_argument(
    "--base_dir", type=str, default="./outputs", help="Directory where existing checkpoints are"
)
parser.add_argument(
    "--label_dir",
    type=str,
    help="Directory where labels are stored",
    default="./labels/label_files",
)
parser.add_argument("--dataset", type=str, help="Dataset to train on")

args = parser.parse_args()
label = args.label
model_name = args.model
num_classes = args.num_classes
seq_seed = args.seed
data_dir = args.data_dir
base_dir = args.base_dir
label_dir = args.label_dir
dataset = args.dataset
assert label in (
    "speech",
    "uniform",
    "shuffle",
    "composite",
    "random",
    "lowdim",
    "category",
    "bert",
)
assert model_name in ("vgg19", "resnet110", "resnet32")
assert dataset in ("cifar10", "cifar100")

# Dependin on label type, load different attacking routines
if "category" in label:
    from utils.attack_category_utils import (
        test_fgsm_untargeted,
        test_fgsm_targeted,
        test_iterative_untargeted,
        test_iterative_targeted,
        test_pgd_untargeted
    )
elif label in ("lowdim", "glove"):
    from utils.attack_lowdim_utils import (
        test_fgsm_untargeted,
        test_fgsm_targeted,
        test_iterative_untargeted,
        test_iterative_targeted,
        test_pgd_untargeted
    )
else:
    from utils.attack_highdim_utils import (
        test_fgsm_untargeted,
        test_fgsm_targeted,
        test_iterative_untargeted,
        test_iterative_targeted,
        test_pgd_untargeted
    )

num_classes = int(dataset.split("cifar")[-1])

print(
    "Start attacking {} {} model (kNN) with manual seed {} and model {}.".format(
        dataset, label, seq_seed, model_name
    )
)

# Directory setup
base_folder = Path(base_dir) / "{}/seed{}/{}/model_{}".format(
    dataset, seq_seed, model_name, label
)
best_model_file = "{}_seed{}_{}_best_model.pth".format(label, seq_seed, model_name)
best_model_path = os.path.join(base_folder, best_model_file)
attack_results_file = "{}_seed{}_{}_attack_results_NN.pth".format(
    label, seq_seed, model_name
)
attack_results_path = os.path.join(base_folder, attack_results_file)
print("Best model location: {}.".format(best_model_path))
print("Attack results location: {}.".format(attack_results_path))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_workers = 4

# Loads attack data
# data_dir, label, num_classes, num_workers, 100, label_dir
attackloader = cifar.get_test_loader(data_dir, label, num_classes, num_workers, 1, label_dir)

# Model setup
if "category" in label or label in ("lowdim", "glove"):
    model = architecture.CategoryModel(model_name, num_classes)
elif label == "bert":
    model = architecture.BERTHighDimensionalModel(model_name, num_classes)
else:
    model = architecture.HighDimensionalModel(model_name, num_classes)

model = nn.DataParallel(model).to(device)
model.load_state_dict(torch.load(best_model_path, map_location=torch.device(device)))
model.eval()


# Run test for each epsilon
epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

if "category" in label:
    mels = None
else:
    mels = torch.tensor(attackloader.dataset.mels, device=device)

num_steps = 5


print("Test PGD untargeted")
pgd_acc = []
pgd_examples = []
for eps in epsilons:
    acc, ex = test_pgd_untargeted(model, device, attackloader, eps, mels)
    pgd_acc.append(acc)
    pgd_examples.append(ex)

torch.save(
    {
        "pgd_acc": pgd_acc,
        "pgd_untargeted_acc": pgd_examples
    },
    attack_results_path,
)
