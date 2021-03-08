### Jigsaw Multilingual Toxic Comment Classification
### 链接：https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification

## 比赛介绍 ##
比赛要求利用纯英语训练数据构建多语言模型。文本数据在comment_text列中，训练数据来自于之前的两个比赛，都是英文数据，有一列是toxic，表示文本是否是有毒评论，有毒就是1，没有就是0，测试数据的comment_text列由多种非英语语言组成。非英语有六种。验证集只给了三类的语言数据，都差不多。训练集0和1的比例在10：1.也就是要在英语语言上训练模型，而在其他语言上做分类。

## 解决方案 ##
主要利用TPU训练了三个模型

### 1，jigsaw-bert模型
* model：bert-base-uncased  
* dataset：2018，2019年的训练数据集 ，但是把测试集的六种语言都翻译成了英语，然后只对英语进行推理。 https://www.kaggle.com/kashnitsky/jigsaw-multilingual-toxic-test-translated 
* epoch：35
* 使用了focal loss和自定义的学习率优化器
* ax_len:512


### 2，jigsaw-distilbert模型
* model：distilbert-base-multilingual-cased
* dataset：只使用2018年的数据集
* epoch：训练集训练5epoch，验证集训练10epoch
* max_len：192

### 3，jigsaw-xlm-RoBerta模型
* model：jigsaw-mlm-finetuned-xlm-r-large   在测试集上使用masked language modelling 进行了微调（预训练）。  https://www.kaggle.com/riblidezso/jigsaw-mlm-finetuned-xlm-r-large
* dataset：使用2018，2019数据集，并使用了翻译数据集，将他们翻译成其他六种语言，并进行抽取。  https://www.kaggle.com/miklgr500/jigsaw-train-multilingual-coments-google-api
* 训练策略：分别在各种语言上训练两轮，再在验证集训练5轮
* max_len：210
