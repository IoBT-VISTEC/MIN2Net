# MIN2Net

End-to-End Multi-Task Learning for Subject-Independent Motor Imagery EEG Classification

[<img src="https://img.shields.io/badge/DOI-10.1109%2FTBME.2021.3137184-blue">](https://ieeexplore.ieee.org/document/9658165/)

---


## Getting started

### Dependencies

> * Python==3.6.9
> * tensorflow-gpu==2.2.0
> * tensorflow-addons==0.9.1
> * scikit-learn>=0.24.1
> * wget>=3.2

1. Create `conda`  environment with dependencies
```bash
$ wget https://raw.githubusercontent.com/IoBT-VISTEC/MIN2Net/main/environment.yml
$ conda env create -f environment.yml
$ conda activate min2net
```

### Installation:

1. Using pip

  ```bash
  $ pip install min2net
  ```

2. Using the released python wheel

  ```bash
  $ wget https://github.com/IoBT-VISTEC/MIN2Net/releases/download/v1.0.0/min2net-1.0.0-py3-none-any.whl
  $ pip install min2net-1.0.0-py3-none-any.whl
  ```

### Citation

To cited [our paper](https://ieeexplore.ieee.org/document/9658165)

P. Autthasan et al., "MIN2Net: End-to-End Multi-Task Learning for Subject-Independent Motor Imagery EEG Classification," in IEEE Transactions on Biomedical Engineering, doi: 10.1109/TBME.2021.3137184.

```
@ARTICLE{9658165,
  author={Autthasan, Phairot and Chaisaen, Rattanaphon and Sudhawiyangkul, Thapanun and 
  Kiatthaveephong, Suktipol and Rangpong, Phurin and Dilokthanakul, Nat 
  and Bhakdisongkhram, Gun and Phan, Huy and Guan, Cuntai and 
  Wilaiprasitporn, Theerawit},
  journal={IEEE Transactions on Biomedical Engineering}, 
  title={MIN2Net: End-to-End Multi-Task Learning for Subject-Independent Motor Imagery 
  EEG Classification}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TBME.2021.3137184}}
```

### Source Code 

View our Code on [<img src="./assets/images/github.png" width="20" height="20">](https://github.com/IoBT-VISTEC/MIN2Net)

### License
Copyright &copy; 2021-All rights reserved by [INTERFACES (BRAIN lab @ IST, VISTEC, Thailand)](https://vistec.ist/interfaces).
Distributed by an [Apache License 2.0](https://github.com/IoBT-VISTEC/MIN2Net/tree/master/LICENSE.txt).
