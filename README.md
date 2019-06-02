#### 比赛内容

网站：[<http://www.ikcest.org/bigdata2019/?lang=zh>](http://www.ikcest.org/bigdata2019/?lang=zh)

查看`introduction`中的文件

#### 代码思路

整体上采用一个双输入单输出的网络，图片输入数据使用迁移学习，用户到访记录数据使用MLP，图片由于是100*100的，直接使用迁移学习硬搬网络不太好，这也是代码需要进一步改进的地方。

最后准确率0.6。

#### 环境要求

python ==> 3.6

keras ==> 2.2.4

#### 代码结构

![](http://ww1.sinaimg.cn/large/e52819eagy1g3muzt8wyij20m00cy760.jpg)

#### 使用步骤
1. 自行从网站下载好数据
2. 运行`preprocess/process.py`，预处理数据
3. 运行`runs/main.py`训练、评估、预测模型
