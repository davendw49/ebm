# Evolving Bipartite Model

![EBM icon](http://www.z4a.net/images/2017/12/21/EBM.png)

**We build an Evolving Bipartite Model (EBM) to reveals the bounded weights in social networks by launching a case study in recommendation networks.**

If you use the code, please cite the following paper:

```
@article{TMC.2019-07-0462,
author = {Jiaqi Liu, Cheng Deng, Luoyi Fu, Huan Long, Xiaoying Gan, Xinbing Wang, Guihai Chen and Jun Xu},
title = {Evolving Bipartite Model Reveals the Bounded Weights in Mobile Social Networks: A Case Study in Recommendation Networks},
year = {2019}
journal = {IEEE Transactions on Mobile Computing}
}
```

This package is mainly contributed by [Cheng Deng](https://github.com/davendw49/) and [Lingkun Kong](https://github.com/Ohyoukillkenny), Guided by Jiaqi Liu, Luoyi FU and Xinbing Wang.

## Introduction

1. **validation** folder contains codes for analysis in 10 social network datasets, including 6 recommendation networks, 3 scholarly networks and 1 facebook messege social networks. The introduction links for these datasets are provided here:

   ​	a. [Amazon Datasets](http://jmcauley.ucsd.edu/data/amazon/links.html) including CDs, Movies, Electronic and Books.

   ​	b. [Audioscrobbler](http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/)

   ​	c. [BookCrossing](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)

   ​	d. [Acemap Academic Knowledge Graph](https://www.acemap.info/)

   ​	e. [Facebook-like Forum Messeage Network](https://toreopsahl.com/datasets/#online_social_network)
    
    if you want to download the dataset, you can refer to my [Google Drive](https://drive.google.com/drive/folders/1nNFJOcrxwpHkZNmjKy3cWJNQbt7XNIUL?usp=sharing).

2. **real_evol** folder contains codes for analysis amazon data's evolving properties.
3. **simulation** folder contains codes for simulation of two algorithm, proposed by us.
4. **simulation_evol** folder contains codes for analysis simulational data's evolving properties.
5. **regression** folder contains codes for datasets's alpha regression with the change of boundry calculating by the degrees.
