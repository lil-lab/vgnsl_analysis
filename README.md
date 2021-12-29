# What is Learned in Visually Grounded Neural Syntax Acquisition

This is the code repository for the paper: "What is Learned in Visually Grounded Neural Syntax Acquisition", [Noriyuki Kojima](https://kojimano.github.io/), [Hadar Averbuch-Elor](http://www.cs.cornell.edu/~hadarelor/), [Alexander Rush](http://nlp.seas.harvard.edu/rush.html) and [Yoav Artzi](https://yoavartzi.com/) (ACL 2020, Short Paper).

### About
[paper](https://arxiv.org/abs/2005.01678)| [talk](https://www.dropbox.com/s/dx1ecbvdsyvd0cl/Presentation.mov?dl=0)

Visual features are a promising signal for learning bootstrap textual models.  However, blackbox  learning  models  make  it  difficult  to  isolate the specific contribution of visual components.   In this analysis,  we consider the case study of the Visually Grounded Neural Syntax Learner [(Shi et al., 2019)](https://ttic.uchicago.edu/~freda/paper/shi2019visually.pdf),  a recent approach for learning syntax from a visual training signal. By constructing simplified versions of the model,  we  isolate  the  core  factors  that  yield the model’s strong performance.  Contrary to what the model might be capable of learning, we find significantly less expressive versions produce similar predictions and perform just as well, or even better. We also find that a simple lexical signal of noun concreteness plays the main role in the model’s predictions as opposed to more complex syntactic reasoning.

![](miscs/parser.gif)


## Codebase

### Contents
1. Requirement: software
2. Requirement: data
3. Test pre-trained models
4. Train your own models

### Requirement: software
Python Virtual Env Setup: All code is implemented in Python. We recommend using virtual environment for installing these python packages.
```
VERT_ENV=vgnsl_analysis

# With virtualenv
pip install virtualenv
virtualenv $VERT_ENV
source $VERT_ENV/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# With Anaconda virtual environment
conda update --all
conda create --name $VERT_ENV python=3.5
conda activate $VERT_ENV
pip install --upgrade pip
pip install -r requirements.txt
```

### Requirement: data
Follow the instruction in https://github.com/ExplorerFreda/VGNSL (`Data Preparation` section) to download all the `/mscoco` data under `/data/mscoco` directory.

### Test pre-trained models
Please refer `/outputs/README.md` to download pre-trained models
<!---1. Download pre-trained models. See outputs/README.md.--->
<!---2. Test models running `./shell/demo_test.sh CHECKPOINTS_FOLDER_NAME`. --->
```
cd src
# calculate F1 score
python test.py --candidate path_to_checkpoint --splits test

# calculate F1 score and output prediction to a text file
python test.py --candidate path_to_checkpoint --splits test --record_trees
```

#### Evaluation on catefory-wise recalls
Please download category annotation from [the link](https://drive.google.com/drive/folders/1OP1lqYXGcV5_CtADOgtHnyQS43lMgO25?usp=sharing) and put them under `/data/mscoco`.

```
# calculate F1 score and catefory-wise recalls
python test.py --candidate path_to_checkpoint --splits test --ctg_eval
```

### Train your own models
<!---1. Train models running `./scripts/demo_train.sh `--->
Coming soon!

### License
MIT

## Citing
If you find this code base and models useful in your research, please consider citing the following paper:
```
@InProceedings{Kojima2020:vgnsl,
    title = "What is Learned in Visually Grounded Neural Syntax Acquisition",
    author = "Noriyuki Kojima and Hadar Averbuch-Elor and Alexander Rush and Yoav Artzi",
    booktitle = "Proceedings of the Annual Meeting of the Association for Computational Linguistics",
    month = "July",
    year = "2020",
    publisher = "Association for Computational Linguistics",
}
```

## Ackowledegement
We would like to thank [Freda](https://ttic.uchicago.edu/~freda/) for making their code public and responding promptly to our inquiry on [Visually Grounded Neural Syntax Acquisition](https://ttic.uchicago.edu/~freda/project/vgnsl/) (Shi et al., ACL2019).
