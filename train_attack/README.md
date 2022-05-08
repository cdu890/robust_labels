## Folder Description

+ All files in this directory override apects of the original `attack.py` file and `train.py` in the [original](https://github.com/BoyuanChen/label_representations) code.
+ The original training code does not work well with GloVe embedding labels and hence, out `train_v2.py` overrides this limitation and enables using embeddings in the network architecture.
+ The `attack_v*.py` scripts support attacks which are not supported in the original code (e.g. PGD)

---