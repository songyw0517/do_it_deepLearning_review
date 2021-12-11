'''
유방암 데이터셋 사용
'''
import numpy as np
class LogisticNeuron:
    def __init__(self):
        self.w = None
        self.b = None
    
    def forpass(self, x):
        z = np.sum(x * self.w) + self.b     # 직선의 방정식을 계산
        # z = w1*x1 + w2*x2 ''' wn*xn + b
        return z
    
    def backprop(self, x, err):
        # 가중치와 절편에 대한 그레이디언트를 계산, 반환
        w_grad = x * err
        b_grad = 1 * err
        return w_grad, b_grad
    
    def fit(self, x, y, epochs=100):
        self.w = np.ones(x.shape[1])        # 가중치를 초기화 합니다.
        self.b = 0                          # 절편을 초기화 합니다.
        for i in range(epochs):             # epochs만큼 반복합니다.
            for x_i, y_i in zip(x, y):      # 모든 샘플에 대해 반복합니다.
                z = self.forpass(x_i)       # 정방향 계산
                a = self.activiation(z)     # 활성화 함수 적용
                err = -(y_i - a)            # 오차 계산
                w_grad, b_grad = self.backprop(x_i, err) # 역방향 계산
                self.w -= w_grad            # 가중치 업데이트
                self.b -= b_grad            # 절편 업데이트
    
    def activiation(self, z):
        z = np.clip(z, -100, None)          # 안전한 np.exp() 계산을 위한 clip함수
        a = 1 / (1 + np.exp(-z))            # 시그모이드 계산
        return a
    
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]    # 선형 함수 적용
        a = self.activiation(np.array(z))       # 활성화 함수 적용
        return a > 0.5                          # 계단 함수 적용

# 데이터 준비
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
x = cancer.data
y = cancer.target

# 박스 플롯으로 데이터 파악하기
import matplotlib.pyplot as plt
plt.boxplot(cancer.data)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
print("4번째 특성과 약41번째의 특성의 분산이 크다는 것을 알 수 있음")

# 훈련 세트와 테스트 세트 나누기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)

# 뉴런(모델) 생성 및 훈련
neuron = LogisticNeuron()
neuron.fit(x_train, y_train)

# 테스트 세트로 모델 성능 확인하기
print('성능 = ',np.mean(neuron.predict(x_test) == y_test))