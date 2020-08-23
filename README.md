# openreview-code

## 目录结构
```
project
├─code
│  └─DataReaders
├─datasets
│  ├─Datasetv0
│  │  ├─ICLR
│  │  │  └─reviews
│  │  └─NIPS
│  │      └─reviews
│  └─sentence-data
├─model
│  ├─electra-small
│  └─sentence-electra-models
└─result
```

## 数据信息读取部分
按照目录结构放置好数据集后,运行Datasetv0reader.py可得到相应json文件

## 情感预测部分
代码位于 code/PaperSentimentAnalyzer.py
用于将review数据使用模型预测情感的正负性

