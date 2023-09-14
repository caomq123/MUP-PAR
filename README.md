# MUP-PAR
MUP: Multi-granularity Unified Perception for Panoramic Activity Recognition
基于多粒度统一感知的全景活动识别
## 算法描述
针对全景数据场景下（例如校园食堂、校园门口等）如何更好地、更全面地理解个人以及群体的行为这类问题，提出了一个多粒度统一感知框架，采用端到端的方式来进行建模共享参数的多粒度行为，并通过堆叠三个统一运动编码块来解决拥挤场景中丰富的语义信息感知困难问题，从而保证模型具有良好的识别能力。
## 环境依赖
- python == 3.8.3
- pytorch == 1.11.0
- CUDA == 11.2
## 数据准备
数据集链接: https://pan.baidu.com/s/1K8RDNteaphYJY8YEAg5fyA Password: PHAR
## 训练
- On PAR benchmark.
```
python train_stage2.py
```
## Acknowledgements
本工作基于[PAR](https://github.com/RuizeHan/PAR), 感谢原作者的工作!
## Citation
如果您在研究中使用此存储库，请引用以下论文。
```
@article{Cao2023mup,
  title={MUP: Multi-granularity Unified Perception for Panoramic Activity Recognition},
  author={Meiqi Cao, Rui Yan, Xiangbo Shu, Jiachao Zhang, Jinpeng Wang, and Guo-Sen Xie},
  booktitle=ACMMM,
  year={2023}
}
 ```
