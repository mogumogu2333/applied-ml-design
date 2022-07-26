# Introducing Natural Language Search for Podcast Episodes
- [Introducing Natural Language Search for Podcast Episodes](#introducing-natural-language-search-for-podcast-episodes)
  - [Natural Language Search](#natural-language-search)
  - [Technical Solution](#technical-solution)
    - [pretrain模型](#pretrain模型)
    - [Data and Train, 数据与训练](#data-and-train-数据与训练)
    - [offline evaluation](#offline-evaluation)
  - [production](#production)
    - [offline index and online](#offline-index-and-online)
    - [No silver bullet in retrieval](#no-silver-bullet-in-retrieval)

[source](https://engineering.atspotify.com/2022/03/introducing-natural-language-search-for-podcast-episodes/)

## Natural Language Search
广播(podcase)搜索，比如query: "electric cars climate impact"。此前，系统采用的elasticSearch+fuzzy match等方法，但这种方式对于自然语言的搜索，尤其是多个单词，而没有字面匹配的情况效果不好。


## Technical Solution
采用dense retrieval的方式，对query和podcase学习encoder，然后通过vector search的方式线上服务。

### pretrain模型
要找合适的sentence encoder模型。bert虽然出名，但CLS token作为dense vector效果不好。

本文采用CMLM模型“Universal Sentence Representation Learning with Conditional Masked Language Model”
* 可生成高质量sentence embedding
* 在多语言数据上做了pretrain

### Data and Train, 数据与训练
生成query,episode正例对，以下来源：
1. search log，成功的(query, podcase) 对
2. search log，搜索session中，可能存在q1->q2->点击播放，比如 "electric cars climate impact"没找到，用户改为 "electric cars environment impact"，找到并收听了podcase A。那么认为第一个query与podcase也是一对，并存在semantic相关性
3. synthetic query: 构造query。训练seq2seq模型，生成query。对eposode就可以构造query用于训练

* 负例为batch negatives
* 划分数据集，按eposode

与召回相同的方式，in-batch negative，对角线为正例，其他为负例。
spotify采用MSE loss，后来又采用了margin loss


### offline evaluation
1. in-batch metric: Recall@1 and Mean Reciprocal Rank (MRR) at the batch level.
2. full retrieval setting: eval数据全部eposode，计算MRR@30

$$MRR=\frac{1}{Q}\sum\frac{1}{rank_i}$$

对一个query, 正例排的位置为rank。一种衡量正确item位置是否靠前的指标。

## production
### offline index and online
线下计算eposode的dense vector，计算好用Vespa index起来。
query线上由Google Cloud Vertex AI serve

### No silver bullet in retrieval
* 多召回源
* retrieval方法在一些有字面匹配的问题上效果也没有traditional IR好
* 拿到多召回源的结果后，rerank by score
* 

