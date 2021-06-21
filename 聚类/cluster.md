## 数据处理

#### 对于int value数据（出度，入度，关注数，粉丝数，微博数, 注册时间）：

* log(x+1)减小数据的极差

* 标准化处理（Z-score）

* 无数据的默认为0（注册时间采用的是原格式，未做特殊处理）

* 直接作为feature的一维

  

#### 领域、性别、是否认证：

* 独热编码
* 性别无数据默认为男



#### 认证原因、简介、标签、地址

* text2vec：每一项96维
* 缺失值默认为全0



将上述所有特征拼接得到user的初步特征

* 还未尝试对feature降维

  

## 聚类

* K-means



## 数据

* 2045个点（清洗掉出度+入度小于的点148个）



## 结果

K=5：result1 2 3

K=4： result4 5

K=3： result6 7

K=6    result 8 

K=7    result 9



## 评测指标

https://www.pianshen.com/article/63101693359/

https://blog.csdn.net/liuy9803/article/details/80762862

https://blog.csdn.net/qq_27825451/article/details/94436488