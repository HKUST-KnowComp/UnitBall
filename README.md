# Unit Model Model for Learning Complex Hyperbolic Embeddings

Source code of the preprint [Unit Ball Model for Embedding Hierarchical Structures in the Complex Hyperbolic Space](https://arxiv.org/abs/2105.03966).

This project is built upon the [Poincare embeddings](https://github.com/facebookresearch/poincare-embeddings).

## Installation
Clone this repository via

```
git clone git@github.com:HKUST-KnowComp/UnitBall.git
cd UnitBall
conda create -n unitball python=3.6
source activate unitball
pip install -r requirements.txt
python setup.py build_ext --inplace
```

## Data
We provide synthetic data (balanced trees and compressed graphs) in `./data/synthetic` and real-world data ([ICD10](https://www.who.int/standards/classifications/classification-of-diseases), [YAGO3-wikiObjects](https://yago-knowledge.org/), [WordNet-noun](https://wordnet.princeton.edu/), [Xiphophorus](https://toytree.readthedocs.io/en/latest/7-multitrees.html)) in `./data`. Please refer to our paper for the data construction details.

## Usage
To train and evaluate the embedding model, run the embed.py file.
```bash
usage: embed.py [-checkpoint] [-trainset] [-testset] [-dim]
                [-manifold {unitball,lorentz,poincare,euclidean}]
                [-model] [-lr] [-eps] [-epochs] -[batchsize] [-negs]
                [-burnin] [-dampening] [-ndprocs] [-fresh] [-debug]
                [-gpu] [-sym] [-maxnorm -no-maxnorm] [-sparse]
                [-burnin_multiplier] [-neg_multiplier] [-quiet]
                [-lr_type {scale,constant}] [-train_threads]
                [-margin] [-eval {reconstruction,hypernymy}]

Description of the arguments:
  -checkpoint           The path to store the embeddings and the model checkpoint
  -trainset             The path of the training set, default='./data/ICD10/train_taxonomy.csv'
  -trainset             The path of the test set, default='./data/ICD10/test_taxonomy.csv'
  -dim                  The dimension of the embedding space
  -manifold             The geometric manifold to learn the embeddings                        
  -model                default='distance'
  -lr                   Learning rate
  -eps                  Eps to avoid numerical instabilities, default=1e-5
  -epochs               Number of epochs
  -batchsize            Number of training examples in one batch
  -negs                 Number of negative samples
  -burnin               Epochs of burn in
  -dampening            Sample dampening during burnin
  -ndprocs              Number of data loading processes
  -fresh                Whether to override checkpoint, action='store_true'
  -debug                Whether to print debuggin output, action='store_true'
  -gpu                  Which cuda to run on, -1 means using cpu
  -sym                  Whether to symmetrize dataset, action='store_true'
  -maxnorm -no-maxnorm  The max norm of the learned embedding vectors
  -sparse               Whether to use sparse gradients, action='store_true'
  -burnin_multiplier    The multiplier of the burnin
  -neg_multiplier       The multiplier of the negative samples
  -quiet                Whether to show the progress bar, action='store_true'
  -lr_type              The type of the learning rate
  -train_threads        Number of threads to use in training, applicable when gpu=-1
  -margin               Hinge margin
  -eval                 Which type of evaluation to perform, default='reconstruction'
```
