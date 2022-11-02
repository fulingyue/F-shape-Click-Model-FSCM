# An F-shape Click Model for Information Retrieval on Multi-block Pages
## Introduction

This is the tensorflow implementation of FSCM proposed in the paper: [An F-shape Click Model for Information Retrieval on Multi-block Pages. WSDM 2023.](https://arxiv.org/abs/2206.08604) 

##Requirements
- python 3.6
- tensorflow 1.14
- numpy
- matplotlib
- seaborn
- pandas
Note that seaborn and matplotlib only be used in eye-tracking codes.

##Input Data Formats

We collect AppStore dataset from a mainstream commercial App Store with F-shape pages, from September 17, 2021 to November 14, 2021.  The first $54$ days are training set, while the last $5$ days are randomly split into valid set and test set. The user behavioral history is collected in real-time. We discard queries that have no positive interactions (clicks) with the retrieval system. The dataset consists of $394046$ unique queries and $1646$ items (Apps). More details of the dataset could be found in the supplementary material. Each item contains $7$ fields of features and each query contains $26$ fields of features. Each app belongs to one of 23 different categories, so that the item converge $\tau_v^j$ is a one-hot vector for the App Store dataset. The dataset statistics are shown as follows:

|                        | training | validating | testing |
| :--------------------- | -------- | ---------- | ------- |
| #sessions              | 637959   | 40792      | 40792   |
| avg. block per session | 3.6654   | 3.7392     | 3.7374  |
| avg. click per session | 0.6510   | 0.7248     | 0.7313  |

The  form of train/valid/test input file is .pkl, and the format is as follows:

- Vertical: [length(1), session_id(1), request_id(1), row_pos(4), item(PER_VER_LENGTH*(FEAT_NUM+2))]

- Horizontal:[row_id, item(PER_HOR_LENGTH*(FEAT_NUM+2))]
- Item:[click, feat(FEAT_NUM)]


##Quick Start

We provide quick start command in run.sh or run following code in command line.

```python
python run.py --train --data_dir data
```

##Citation

If you find the resources in this repo useful, please cite our work.



