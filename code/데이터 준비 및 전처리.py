from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
# 데이터 불러오기
cancer = load_breast_cancer()
x = cancer.data
y = cancer.target

# 훈련 세트와 테스트 세트 나누기
x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)

# 훈련 세트로부터 검증 세트 나누기
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, stratify=y_train_all, test_size=0.2, random_state=42)

# 데이터 전처리하기 (표준화)
## 주의사항 : 전처리시 훈련데이터의 평균, 표준편차만을 이용하여 표준화해야한다.
train_mean = np.mean(x_train, axis=0)   # train_mean = u
train_std = np.std(x_train, axis=0)     # train_std = s

# 훈련 데이터 전처리하기
x_train_scaled = (x_train - train_mean) / train_std   # x_train_scaled = z

# 검증 데이터 전처리하기
x_val_scaled = (x_val - train_mean) / train_std

# 테스트 데이터 전처리하기
x_test_scaled = (x_test - train_mean) / train_std