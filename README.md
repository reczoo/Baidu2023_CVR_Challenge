## Baidu2023_CVR_Challenge

百度商业AI技术创新大赛：商业转化行为预测


```bash
cd ~/Baidu2023_CVR_Challenge/data/data205411
7zr x train_data.7z
mv 2023-cvr-contest-data/train_data ~/Baidu2023_CVR_Challenge/data/
cd ~/Baidu2023_CVR_Challenge/data/data204194
7zr x test_data.7z
mv test_data ~/Baidu2023_CVR_Challenge/data/

```

```bash
cd ~/Baidu2023_CVR_Challenge/
python data_prepare_v1.py
```

### Version 1
```
python run_param_tuner.py --config config/FINAL_data_v1_tuner_config_01.yaml --gpu 0
```

<div align="left">
    <img width="100%" src="./img/submit_0.74063.png">
</div>

### Version 2
```
python run_param_tuner.py --config config/FINAL_data_v1_tuner_config_01.yaml --gpu 0
```
<div align="left">
    <img width="100%" src="./img/submit_0.7417.png">
</div>
