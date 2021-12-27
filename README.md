[<img src="https://min2net.github.io/assets/images/min2net-logo.png" width="30%" height="30%">](https://min2net.github.io)

[![Pypi Downloads](https://img.shields.io/pypi/v/min2net?color=green&logo=pypi&logoColor=white)](https://pypi.org/project/min2net)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FTBME.2021.3137184-blue)](https://ieeexplore.ieee.org/document/9658165)

End-to-End Multi-Task Learning for Subject-Independent Motor Imagery EEG Classification
- **Documentation**: https://min2net.github.io
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

### License
Copyright &copy; 2021-All rights reserved by [INTERFACES (BRAIN lab @ IST, VISTEC, Thailand)](https://www.facebook.com/interfaces.brainvistec).
Distributed by an [Apache License 2.0](https://github.com/IoBT-VISTEC/MIN2Net/blob/main/LICENSE).
