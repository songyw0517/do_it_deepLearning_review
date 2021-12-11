'''
로지스틱회귀 모델과 거의 동일
유방암 데이터셋 사용
'''

import numpy as np
class SingleLayer:
    def __init__(self):
        self.w = None
        self.b = None
        self.losses = []                                                  # 손실을 저장하는 리스트
    
    def forpass(self, x):
        z = np.sum(x * self.w) + self.b     # 직선의 방정식을 계산
        # z = w1*x1 + w2*x2 ''' wn*xn + b
        return z
    
    def backprop(self, x, err):
        # 가중치와 절편에 대한 그레이디언트를 계산, 반환
        w_grad = x * err
        b_grad = 1 * err
        return w_grad, b_grad
    
    # 확률적 경사하강법을 적용한 훈련법
    def fit(self, x, y, epochs=100):
        self.w = np.ones(x.shape[1])        # 가중치를 초기화 합니다.
        self.b = 0                          # 절편을 초기화 합니다.
        for i in range(epochs):             # epochs만큼 반복합니다.
            loss = 0                                                      # 손실을 0으로 초기화합니다.
            indexes = np.random.permutation(np.arange(len(x)))            # 확률적 경사하강법을 적용하기 위한 인덱스 섞기
            for i in indexes:
                z = self.forpass(x[i])       # 정방향 계산
                a = self.activiation(z)     # 활성화 함수 적용
                err = -(y[i] - a)            # 오차 계산
                w_grad, b_grad = self.backprop(x[i], err) # 역방향 계산
                self.w -= w_grad            # 가중치 업데이트
                self.b -= b_grad            # 절편 업데이트

                # 안전한 로그 계산을 위해 클리핑한 후 손실을 누적합니다.
                a = np.clip(a, 1e-10, 1-1e-10)
                loss += -(y[i]*np.log(a)+(1-y[i])*np.log(1-a))
            self.losses.append(loss/len(y))
    
    def activiation(self, z):
        z = np.clip(z, -100, None)          # 안전한 np.exp() 계산을 위한 clip함수
        a = 1 / (1 + np.exp(-z))            # 시그모이드 계산
        return a
    
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]    # 선형 함수 적용
                                                                            # a = self.activiation(np.array(z))       # 활성화 함수를 뺌
        return np.array(z) > 0                  # 계단 함수 적용
    
    def score(self, x, y):                                                  # 성능을 보여주는 메서드
        return np.mean(self.predict(x) == y)                                # predict메서드에 입력 값을 넣고, y값과 비교하여 평균을 낸다.

# 데이터 준비
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
x = cancer.data
y = cancer.target

# 훈련 세트와 테스트 세트 나누기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)

# 단일층 신경망 생성
layer = SingleLayer()

# 신경망 훈련
layer.fit(x_train, y_train)

# 신경망 성능 테스트
print(layer.score(x_test, y_test))

# 신경망의 손실 리스트 그래프로 그리기
import matplotlib.pyplot as plt
plt.plot(layer.losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()