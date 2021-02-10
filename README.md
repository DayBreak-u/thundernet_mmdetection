# MMDetection_Lite

基于[mmdetection](https://github.com/open-mmlab/mmdetection) 实现一些轻量级检测模型，安装方式和mmdeteciton相同



## voc0712
voc 0712训练 voc2007测试
thundernet_voc_shufflenetv2_1.5

|  input   shape           |      mAP      |    
|--------------------|:-------------:|
| 320*320         | 0.71            | 
| 352*352         | 0.722           | 
| 384*384         | 0.734            | 
| 416*416         | 0.738           | 
| 448*448         | 0.744           | 
| 480*480         | 0.747           | 


## ncnn project
[ncnn](https://github.com/ouyanghuiyu/thundernet_mmdetection/tree/master/ncnn_project)


## Get Started

Please see [GETTING_STARTED.md](docs/get_started.md) for the basic usage of MMDetection.



```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```