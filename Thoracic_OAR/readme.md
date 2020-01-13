# 说明

try文件夹里面放的都是一些用来测试tensorflow用法的py文件样例，这里放的是测试理解计算图、变量、ops之间关系，以及pb和ckpt两种权重保存方式下模型的再读取。

## cstro相关

cstro比赛肺部OAR以及GTV赛道的参赛模型，其实就是朴素Unet然后加上了很多预处理方法，例如对窗宽窗位、摆位误差的自适应，还有裁剪、旋转、翻转这些经典的数据增强的方法，z-score代替简单的最大最小正则，其他的就不多说了。

## 训练过程

1~24 epoch 使用了随机水平竖直翻转、15°旋转，此时测试结果会有左右肺互相混淆的情况，这是因为左右肺的灰度太过于相似，并且小器官分割效果比较差  
25 epoch 开始使用10°旋转，并禁用翻转  
55 epoch 开始使用5°旋转  
67 epoch 开始使用0°翻转
82 epoch 训练结束
最终最佳结果为77 epoch 平均训练dice值为：0.925  
测试结果：test_result文件夹下

## 后处理
腐蚀：去除微小的多余轮廓  
根据轮廓大小判断，左右肺弄混的情况  
膨胀：填充空隙  
后处理的一些结果展示:  
![png1](https://raw.githubusercontent.com/sennnnn/Unet-cstro/master/Thoracic_OAR/material/patient_1_slice_52.png)  
![png2](https://raw.githubusercontent.com/sennnnn/Unet-cstro/master/Thoracic_OAR/material/patient_1_slice_62.png)