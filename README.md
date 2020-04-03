## MAGNN

This repository provides a reference implementation of MAGNN as described in the paper:
> MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding.<br>
> Xinyu Fu, Jiani Zhang, Ziqiao Meng, Irwin King.<br>
> The Web Conference, 2020.

Available at [arXiv:2002.01680](https://arxiv.org/abs/2002.01680).

### Dependencies

Recent versions of the following packages for Python 3 are required:
* PyTorch 1.2.0
* DGL 0.3.1
* NetworkX 2.3
* scikit-learn 0.21.3
* NumPy 1.17.2
* SciPy 1.3.1

Dependencies for the preprocessing code are not listed here.

### Datasets

The preprocessed datasets are available at:
* IMDb - [Dropbox](https://www.dropbox.com/s/g0btk9ctr1es39x/IMDB_processed.zip?dl=0)
* DBLP - [Dropbox](https://www.dropbox.com/s/yh4grpeks87ugr2/DBLP_processed.zip?dl=0)
* Last.fm - [Dropbox](https://www.dropbox.com/s/jvlbs09pz6zwcka/LastFM_processed.zip?dl=0)

The GloVe word vectors are obtained from [GloVe](https://nlp.stanford.edu/projects/glove/). Here is [the direct link](http://nlp.stanford.edu/data/glove.6B.zip) for the version we used in DBLP preprocessing.

### Usage

1. Create `checkpoint/` and `data/preprocessed` directories
2. Extract the zip file downloaded from the section above to `data/preprocessed`
    * E.g., extract the content of `IMDB_processed.zip` to `data/preprocessed/IMDB_processed`
2. Execute one of the following three commands from the project home directory:
    * `python run_IMDB.py`
    * `python run_DBLP.py`
    * `python run_LastFM.py`

For more information about the available options of the model, you may check by executing `python run_IMDB.py --help`

### Citing

If you find MAGNN useful in your research, please cite the following paper:

	@inproceedings{fu2020magnn,
     title={MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding},
     author={Xinyu Fu and Jiani Zhang and Ziqiao Meng and Irwin King},
     booktitle = {WWW},
     year={2020}
    }
