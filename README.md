# 太阳风暴智能预警模型
本项目从 [阿里云天池比赛项目](https://tianchi.aliyun.com/competition/entrance/531804/introduction?spm=5176.12281949.1003.4.493e2448py46sm) 出发，通过 Pytorch 深度学习框架，搭建了多层神经网络模型。
<br>
### 1、项目文件结构
sun_wind<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|------dada  数据集<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---train_para_input.txt 训练输出参数<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---train_output.txt 训练输出标签<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---test_para_input.txt 测试数据集<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|------read_data.py 读取数据<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|------train_1.py 训练模型<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|------test_1.py 测试数据<br>
<br>
### 2、具体说明
#### 2.1数据读取
可以从两个数据文件train_para_input.txt、train_output.txt中读取太阳耀斑的磁特征参数数据和标签，共计有 5000条数据。
输入参数采用了均值归一化处理。
#### 2.2模型搭建
在trian.py文件中，使用 pytorch 深度学习框架，搭建了四层神经网络模型，此外，使用了 Dropout 方法，防止模型过拟合。<br>
#### 2.3测试数据
执行test.Py 文件，可以得到测试数据集的标签（太阳耀斑分发生情况）。
