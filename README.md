# Kaggle_SCTP_31th-solution

1. Seperate public/private/synthetic data from test data

using unique values from each raw feature, we can seperate test data into three part: public/private/synthetic, and removing synthetic data is importance for generating frequency-based feature.


2. Feature

(1) raw feature: var_0 ~ var_199

(2) drop count=1 values from each raw feature: var_0_repeat_2 ~ var_199_repeat_2 -> single LGBM should reach auc 0.922

(3) drop count=1/2 values from each raw feature: var_0_repeat_3 ~ var_199_repeat_3 -> boost auc by .0005


3. LGBM parameters

[2] shows that LGBM can reach same auc(.900) after shuffling each raw feature individually, meaning there are barely interactions between features. By setting lower feature_fraction and lower num_leaves will help model NOT to learn from fake interaction, so do [3]data augment.


Kernels:

[1] https://www.kaggle.com/yag320/list-of-fake-samples-and-public-private-lb-split

[2] https://www.kaggle.com/brandenkmurray/randomly-shuffled-data-also-works

[3] https://www.kaggle.com/jiweiliu/lgb-2-leaves-augment
