# AU_EMOwPGM
Try to learn EMO-AU relation with PGM thoughts

- 截止到2022-10-18，已经完成：
    - 双支预测网络（AU检测和EMO识别）：
    ```
    models/TwoBrach.py
    ```
    - 规则学习网络
    ```
    models/rule_model.py
    ```
    通过运行以下代码可以完成双支预测网络的学习和对应数据集的规则学习：
    ```
    python main_datasetname.py
    ```
    或者在已有AU和EMO标签的情况下运行
    ```
    python main_rule.py
    ```
    - 规则增量学习网络
    ```
    continuous/rules_continuous.py
    ```
    通过运行以下代码可以完成规则上的增量学习：
    ```
    python continuous/main_continuous.py
    ```
- 结果
    - 单个数据集训练

    dataset | happy | sad | fear | anger | surprise | disgust | all
    :-: | :-: | :-: | :-: | :-:| :-: | :-: | :-: 
    BP4D | 99.86 | 81.17 | 96.70 | 77.19 | 28.65 | 46.42 | 78.17
    DISFA | 14.60| 100.00 | 97.75 | 0.10 | 100.00 | 100.00 | 70.42
    RAF-DB | 93.92| 100.00 | 0.00 | 0.00 | 100.00 | 0.00 | 83.44
    AffectNet | 100.00| 49.16 | 0.00 | 70.29 | 100.00 | 0.00 | 53.37

    - 数据集混合学习 （BP4D + RAF-DB + AffectNet）

    siatuation | happy | sad | fear | anger | surprise | disgust | all
    :-: | :-: | :-: | :-: | :-:| :-: | :-: | :-: 
    all_mixed | 100.00 | 69.28 | 63.14 | 78.10 | 65.96 | 10.08 | 69.01
    BP4D | 100.00| 73.33 | 66.78 | 80.57 | 60.66 | 10.96 | 70.05
    RAF-DB | 100.00 | 46.03 | 39.19 | 74.07 | 84.45 | 3.16 | 76.98
    AffectNet | 100.00 | 50.63 | 21.46 | 50.42 | 83.47 | 6.17 | 52.16

    - 数据集规则增量学习 （BP4D -> RAF-DB -> AffectNet）

    /media/data1/wf/AU_EMOwPGM/codes/continuous/save/continuous/2022-10-18/BRA/train.log
