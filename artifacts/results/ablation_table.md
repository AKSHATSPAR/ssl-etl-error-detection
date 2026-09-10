| ablation        |   AUPRC |   Hit@1 |   MRR |   delta Hit@1 | verdict           |
|:----------------|--------:|--------:|------:|--------------:|:------------------|
| clean_train     |   0.952 |   0.814 | 0.892 |         0.009 | within seed noise |
| no_dag_bias     |   0.953 |   0.801 | 0.884 |        -0.005 | within seed noise |
| no_calendar     |   0.953 |   0.79  | 0.879 |        -0.015 | within seed noise |
| contrastive     |   0.953 |   0.787 | 0.88  |        -0.018 | within seed noise |
| longer_training |   0.948 |   0.756 | 0.861 |        -0.049 | OUTSIDE 2 sd      |