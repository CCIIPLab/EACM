# Emotion-Aware Chat Machine

The source code of the paper "Emotion-aware Chat Machine: Automatic Emotional Response Generation for Human-like Emotional Interaction" (Published on [CIKM2019](https://dl.acm.org/doi/10.1145/3357384.3357937)).

![image](https://github.com/CCIIPLab/EACM/blob/main/Model.png)

## Dependencies
	
* Python 3.5
* Numpy
* Tensorflow 0.12

## Quick Start

* Dataset

Due to the copyright of the STC dataset, you can ask Lifeng Shang (lifengshang@gmail.com) for the STC dataset ([Neural Responding Machine for Short-Text Conversation](https://arxiv.org/abs/1503.02364v2)), and build the ESTC dataset follow the instruction in the Data Preparation Section of our paper.

The basic format of the sample data is:

>  [[[post, primary emotion, secondary emotion], [response, primary emotion, secondary emotion]], ...]

where emotion tag1/tag2 is generated by the BERT model which is used as a multi-label classifier.  The training data of this classifier can be found on [NLPCC2014](http://tcci.ccf.org.cn/conference/2014/pages/page04_sam.html) and [NLPCC2013](http://tcci.ccf.org.cn/conference/2013/pages/page04_eva.html) website.

We provide an example of the Dev_Set in this repository, which has the same format as the Train_Set.

* Train

	``` python baseline.py --use_emb --use_autoEM --train_dir="train/EACM"```

Note that "--use_imemory", "--use_ememory" are originally designed for ECM model, and we do not need that for EACM.

* Test

	``` python baseline.py --use_emb --use_autoEM --train_dir="train/EACM" --decode	```

Need to note that the testing files placed in "./train/test.post", and the model will automatically generate responses according to the testfile.
After predicting the testfile, the model will go to interaction mode to wait for an input.

## Paper

**Please kindly cite our paper if this paper and the code are helpful.**
```
@inproceedings{10.1145/3357384.3357937,
author = {Wei, Wei and Liu, Jiayi and Mao, Xianling and Guo, Guibing and Zhu, Feida and Zhou, Pan and Hu, Yuchong},
title = {Emotion-Aware Chat Machine: Automatic Emotional Response Generation for Human-like Emotional Interaction},
year = {2019},
pages = {1401–1410},
location = {Beijing, China},
series = {CIKM '19}
}
```

## Acknowlegments

Thanks Hao Zhou for sharing the original code of [ECM model](https://arxiv.org/abs/1704.01074) , which is available [here](https://github.com/thu-coai/ecm).

## License

Apache License 2.0
