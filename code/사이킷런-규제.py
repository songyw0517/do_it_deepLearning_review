from sklearn.linear_model import SGDClassifier

# 데이터 준비
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

cancer = load_breast_cancer()
x = cancer.data
y = cancer.target

# 데이터 나누기 (훈련 세트, 테스트 세트)
x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)

# 데이터 나누기 (훈련 세트, 검증 세트)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, stratify=y_train_all, test_size=0.2, random_state=42)

# 데이터 전처리
train_mean = np.mean(x_train, axis=0)   # train_mean = u
train_std = np.std(x_train, axis=0)     # train_std = s

## 훈련 데이터 전처리
x_train_scaled = (x_train - train_mean) / train_std   # x_train_scaled = z
## 검증 데이터 전처리
x_val_scaled = (x_val - train_mean) / train_std
## 테스트 데이터 전처리
x_test_scaled = (x_test - train_mean) / train_std

sgd = SGDClassifier(loss='log', penalty='l2', alpha=0.001, random_state=42)
sgd.fit(x_train_scaled, y_train)
print('l2 성능 = ',sgd.score(x_val_scaled, y_val))