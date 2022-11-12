1112更新
前80%的unseen sample 用来训练unseen的EMO2AU, 后面的sample用来KL散度拉近seen_trained和cat_trained_seen之间的分布

这种方式能够获取到unseen sample对seen_rule的影响，但是丢失了seen sample对unseen_rule的影响，可能是因为这一点，导致cat_trained虽然能够保持对seen的判断，但是unseen 的结果一直没有提升(相当于没训练上)

cat_trained在val_seen_sample上的结果：
2022-11-11 09:43:26,797:INFO: {'init_lr 0.0001'}
2022-11-11 09:46:06,603:INFO: {'Dataset: BP4D val_rules_loss: 0.31719, val_rules_acc: 99.57'}
2022-11-11 09:46:06,603:INFO: {'EMO Rules Val Acc-list:'}
2022-11-11 09:46:06,604:INFO: {'fear: 100.00, disgust: 98.79'}
2022-11-11 09:46:06,672:INFO: {'init_lr 0.0001'}
2022-11-11 09:46:12,599:INFO: {'Dataset: RAF-DB val_rules_loss: 0.42443, val_rules_acc: 88.70'}
2022-11-11 09:46:12,599:INFO: {'EMO Rules Val Acc-list:'}
2022-11-11 09:46:12,599:INFO: {'fear: 64.38, disgust: 100.00'}
2022-11-11 09:46:12,971:INFO: {'init_lr 0.0001'}
2022-11-11 09:47:41,509:INFO: {'Dataset: DISFA val_rules_loss: 0.45559, val_rules_acc: 85.76'}
2022-11-11 09:47:41,509:INFO: {'EMO Rules Val Acc-list:'}
2022-11-11 09:47:41,509:INFO: {'fear: 100.00, disgust: 69.00'}
2022-11-11 09:47:41,906:INFO: {'init_lr 0.0001'}
2022-11-11 09:48:43,708:INFO: {'Dataset: AffectNet val_rules_loss: 0.67852, val_rules_acc: 56.68'}
2022-11-11 09:48:43,708:INFO: {'EMO Rules Val Acc-list:'}
2022-11-11 09:48:43,708:INFO: {'fear: 12.39, disgust: 100.00'}

cat_trained在val_all_sample上的结果：
2022-11-11 22:27:01,512:INFO: {'init_lr 0.001'}
2022-11-11 22:32:54,998:INFO: {'Dataset: BP4D val_rules_loss: 0.58269, val_rules_acc: 42.70, val_rules_loss: 1.46253, val_rules_acc: 57.84'}
2022-11-11 22:32:54,998:INFO: {'EMO Rules Val Acc-list:'}
2022-11-11 22:32:54,998:INFO: {'happy: 99.83, sad: 59.93, anger: 94.83, surprise: 80.10, fear: 0.28, disgust: 12.50'}
2022-11-11 22:32:55,289:INFO: {'init_lr 0.001'}
2022-11-11 22:33:12,272:INFO: {'Dataset: RAF-DB val_rules_loss: 0.41780, val_rules_acc: 59.29, val_rules_loss: 1.39465, val_rules_acc: 64.32'}
2022-11-11 22:33:12,272:INFO: {'EMO Rules Val Acc-list:'}
2022-11-11 22:33:12,273:INFO: {'happy: 96.93, sad: 67.09, anger: 0.00, surprise: 1.55, fear: 21.92, disgust: 27.39'}
2022-11-11 22:33:12,704:INFO: {'init_lr 0.001'}
2022-11-11 22:36:10,576:INFO: {'Dataset: DISFA val_rules_loss: 0.58922, val_rules_acc: 43.77, val_rules_loss: 1.68441, val_rules_acc: 35.96'}
2022-11-11 22:36:10,576:INFO: {'EMO Rules Val Acc-list:'}
2022-11-11 22:36:10,577:INFO: {'happy: 98.35, sad: 100.00, anger: 0.48, surprise: 100.00, fear: 0.12, disgust: 0.05'}
2022-11-11 22:36:11,034:INFO: {'init_lr 0.001'}
2022-11-11 22:38:11,993:INFO: {'Dataset: AffectNet val_rules_loss: 0.59886, val_rules_acc: 39.75, val_rules_loss: 1.50375, val_rules_acc: 53.31'}
2022-11-11 22:38:11,994:INFO: {'EMO Rules Val Acc-list:'}
2022-11-11 22:38:11,994:INFO: {'happy: 100.00, sad: 56.84, anger: 61.45, surprise: 100.00, fear: 0.00, disgust: 0.22'}