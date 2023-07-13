## Baidu2023_CVR_Challenge

百度商业AI技术创新大赛：[商业转化行为预测]()

本赛道任务是广告转化率预估，在广告领域中，转化率预估被广泛应用于优化广告投放，提高广告效果和ROI。 转化率预估是指通过对用户行为数据的分析，预测用户是否会执行某种特定的转化行为，比如下载APP、购买商品或服务等。 本次比赛提供了百度真实的广告数据集，包含了海量的用户行为数据和广告特征。希望参赛者使用深度学习模型，建模转化率预估模型。

本方案采用FuxiCTR为开发框架，选用FINAL模型(SIGIR'23发表)，未经任何精细化设计的情况下，迅速完成了较高的基线构建 (AUC: 0.74184, pcoc: 1.033, 排名40)


### 数据准备
1. 数据解压

    ```bash
    cd ~/Baidu2023_CVR_Challenge/data/data205411
    7zr x train_data.7z
    mv 2023-cvr-contest-data/train_data ~/Baidu2023_CVR_Challenge/data/
    cd ~/Baidu2023_CVR_Challenge/data/data204194
    7zr x test_data.7z
    mv test_data ~/Baidu2023_CVR_Challenge/data/

    ```

2. 数据转换，采用csv输入格式，并合并多值序列特征

    ```bash
    cd ~/Baidu2023_CVR_Challenge/
    python data_prepare_v1.py
    ```

### Version 1

1. 训练模型，针对train集和valid集的数据划分进行超参搜索

    ```
    python run_param_tuner.py --config config/FINAL_data_v1_tuner_config_01.yaml --gpu 0
    ```

2. 对test数据进行预测，并生成提交文件

   从实验结果中选取一组最好的验证集效果[val] AUC: 0.739392 - logloss: 0.216473，从[command]列提取训练命令行，并将`run_expid.py`改为`predict.py`进行预测。

    ```
    python predict.py --config config/tuner_config_v1 --expid FINAL_data_v1_001_507152ea --gpu 0
    ```

   对生成的结果文件test-1.txt进行压缩为submission.zip后提交。


3. 提交结果，测试AUC为0.74063
<div align="left">
    <img width="100%" src="./img/submit_0.74063.png">
</div>


### Version 2

1. 训练模型，采用所有数据(data.csv)作为训练集进行训练，仍采用valid.csv进行验证，为避免过拟合，设置训练epochs=1。

    ```
    python run_param_tuner.py --config config/FINAL_data_v1_tuner_config_02.yaml --gpu 0
    ```

2. 对test数据进行预测，并生成提交文件

   从实验结果中选取一组最好的验证集效果[val] AUC: 0.796111 - logloss: 0.203715，从[command]列提取训练命令行，并将`run_expid.py`改为`predict.py`进行预测。

    ```
    python predict.py --config config/FINAL_data_v1_tuner_config_02 --expid FINAL_data_v1_004_84322a67 --gpu 0
    ```
    
    对生成的结果文件test-1.txt进行压缩为submission.zip后提交。

3. 提交结果，测试AUC为0.7417
<div align="left">
    <img width="100%" src="./img/submit_0.7417.png">
</div>


### Version 3

1. 在Version 2的基础上，设置训练epochs=2。

    ```
    python run_param_tuner.py --config config/FINAL_data_v1_tuner_config_03.yaml --gpu 0
    ```

2. 实验最后没跑完，结果只出来一组，进行预测

   从实验结果中选取一组最好的验证集效果[val] AUC: 0.806603 - logloss: 0.200580，从[command]列提取训练命令行，并将`run_expid.py`改为`predict.py`进行预测。

    ```
    python predict.py --config config/FINAL_data_v1_tuner_config_03 --expid FINAL_data_v1_001_31c7f291 --gpu 0
    ```


4. 提交结果，测试AUC为0.74184，排名40
<div align="left">
    <img width="100%" src="./img/submit_0.74184.png">
</div>


### 后记

由于时间关系，一共只提交了5次。作为基线方案test_AUC=0.7417的分数还不错。其他可能需要尝试的方案：
+ 多值(序列)特征需要重点处理，这里可以采用DIN中的target attention进行建模 
+ 数据预处理默认采用min_categr_count=10，可进一步优化
+ Embedding维度没有仔细调优，除了Embedding维度大小也可考虑NFFM中的field-aware embedding方案。
+ 交叉特征/统计特征的添加
