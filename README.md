## MAGNN

This repository provides a reference implementation of MAGNN as described in the paper:
> MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding.<br>
> Xinyu Fu, Jiani Zhang, Ziqiao Meng, Irwin King.<br>
> The Web Conference, 2020.

Available at [arXiv:2002.01680](https://arxiv.org/abs/2002.01680).

### Dependencies

Recent versions of the following packages for Python 3 are required:
* PyTorch
* DGL
* NetworkX
* scikit-learn
* NumPy
* SciPy

### Datasets

The preprocessed datasets are available at:
* IMDb - [Dropbox](https://www.dropbox.com/s/g0btk9ctr1es39x/IMDB_processed.zip?dl=0) / [Baidu](https://pan.baidu.com/s/1xLBpcgDvb__HAy9PV0d6oA) (qdng)
* DBLP - [Dropbox](https://www.dropbox.com/s/yh4grpeks87ugr2/DBLP_processed.zip?dl=0) / [Baidu](https://pan.baidu.com/s/1RVPDvFuqdAxbFXpKoGGiTA) (fa6l)
* Last.fm - [Dropbox](https://www.dropbox.com/s/jvlbs09pz6zwcka/LastFM_processed.zip?dl=0) / [Baidu](https://pan.baidu.com/s/1Zd-W91x-14qO1mjz6-sDUg) (sv7b)

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
