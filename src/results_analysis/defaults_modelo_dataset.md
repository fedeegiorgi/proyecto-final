# Parámetros default

### Original

| Dataset \ Params  | max_depth |
| ----------------- | --------- |
| Carbon            | XX        |
| House8L           | XX        |
| Wind              | XX        |

## Alternativa A

### IQR

| Dataset \ Params  | n_estimators | group_size | max_depth |
| ----------------- | ------------ | ---------- | --------- |
| Carbon            | 150          | 3          | 17        |
| House8L           | 150          | 3          | 17        |
| Wind              | 150          | 3          | 19        |

### Percentile Trimming

| Dataset \ Params  | n_estimators | group_size | percentile | max_depth |
| ----------------- | ------------ | ---------- | ---------- | --------- |
| Carbon            | 150          | 50         | 2          | 44        |
| House8L           | 150          | 50         | 2          | 40        |
| Wind              | 150          | 50         | 2          | 44        |

## Alternativa B

### OOB

| Dataset \ Params  | n_estimators | group_size | max_depth |
| ----------------- | ------------ | ---------- | --------- |
| Carbon            | 180          | 3          | 20        |
| House8L           | 180          | 3          | 32        |
| Wind              | 180          | 3          | 32        |

### OOB+IQR

| Dataset \ Params  | n_estimators | group_size | max_depth |
| ----------------- | ------------ | ---------- | --------- |
| Carbon            | 180          | 3          | 21        |
| House8L           | 180          | 3          | 42        |
| Wind              | 180          | 3          | 14        |

## Alternativa C

### FirstSplitsCombiner

| Dataset \ Params  | n_estimators | group_size | max_features |
| ----------------- | ------------ | ---------- | ------------ |
| Carbon            | 100          | 10         | log2         |
| House8L           | 100          | 10         | log2         |
| Wind              | 100          | 10         | log2         |

## Alternativa D

| Dataset \ Params  | n_estimators | group_size | initial_max_depth | max_depth |
| ----------------- | ------------ | ---------- | ----------------- | --------- |
| Carbon            |              |            |                   |           |
| House8L           |              |            |                   |           |
| Wind              |              |            |                   |           |