import numpy as np
class SingleLayer:
    def __init__(self, learning_rate=0.1, l1=0, l2=0):
        self.w = None                       # 가중치
        self.b = None                       # 절편
        self.losses = []                    # 훈련 손실
        self.val_losses = []                # 검증 손실
        self.w_history = []                 # 가중치 기록
        self.lr = learning_rate             # 학습률
        self.l1 = l1                        # L1 규제 하이퍼 파라미터 적용
        self.l2 = l2                        # L2 규제 하이퍼 파라미터 적용
    
    def forpass(self, x):
        # 배치 경사 하강법을 위한 코드 (전체 데이터를 한번에 계산하는 방법)
        z = np.dot(x, self.w) + self.b
        return z
    
    def backprop(self, x, err):
        m = len(x) # 전체 샘플의 수, 각각의 그레이디언트는 전체 샘플의 수로 나눈다.

        # x : (샘플 수, 속성수), err : (샘플 수,) -> 점곱이 불가능하다.
        # x.T로 전치하여 계산
        # w_grad의 각 행은 각 특성에 따른 err이 곱해진 그레이디언트가 계산된다.
        # g1 = x1*e1 + x2*e2 + x3*e3 ... xn*en
        w_grad = np.dot(x.T, err) / m

        b_grad = np.sum(err) / m
        return w_grad, b_grad
    
    def fit(self, x, y, epochs=100, x_val=None, y_val=None):
        y = y.reshape(-1, 1)                # 타깃을 열 벡터로 바꿉니다.
        y_val = y_val.reshape(-1, 1)        # 검증용 타깃을 열 벡터로 바꿉니다.
        m = len(x)                          # 샘플의 개수를 저장합니다.
        self.w = np.ones((x.shape[1], 1))   # 가중치를 초기화 합니다.
        self.b = 0                          # 절편을 초기화 합니다.
        self.w_history.append(self.w.copy())# 가중치를 기록합니다.

        # epochs만큼 반복합니다.
        for i in range(epochs):
            z = self.forpass(x)
            a = self.activation(z)
            err = -(y - a)

            # 오차를 역전파하여 그레이디언트를 계산합니다.
            w_grad, b_grad = self.backprop(x, err)

            # 그레이디언트에서 페널티 항의 미분값을 더합니다.
            w_grad += (self.l1 * np.sign(self.w) + self.l2 * self.w) / m

            # 가중치와 절편을 업데이트 합니다.
            self.w -= self.lr * w_grad
            self.b -= self.lr * b_grad

            # 가중치를 기록합니다.
            self.w_history.append(self.w.copy())

            # 안전한 로그 계산을 위해 클리핑합니다.
            a = np.clip(a, 1e-10, 1-1e-10)

            # 로그 손실과 규제 손실을 더하여 리스트에 추가합니다.
            loss = np.sum(-(y*np.log(a) + (1-y)*np.log(1-a)))
            self.losses.append((loss + self.reg_loss()) / m)

            # 검증 세트에 대한 손실을 계산합니다.
            self.update_val_loss(x_val, y_val)
    
    def activation(self, z):
        z = np.clip(z, -100, None)          # 안전한 np.exp() 계산을 위한 clip함수
        a = 1 / (1 + np.exp(-z))            # 시그모이드 계산
        return a
    
    def predict(self, x):
        z = self.forpass(x)                 # 정방향 계산
        return z > 0                        # 예측 결과 반환


    def score(self, x, y):                      # 성능을 보여주는 메서드
        # 예측과 타깃 열 벡터를 비교하여 True의 비율을 반환
        return np.mean(self.predict(x) == y.reshape(-1, 1))

    # 로지스틱 손실 함수 계산에 페널티 항 추가하기
    def reg_loss(self):
        # 가중치에 규제를 적용합니다.
        return self.l1 * np.sum(np.abs(self.w)) + self.l2 / 2 * np.sum(self.w**2)
    
    
    # 검증 손실을 업데이트 합니다.
    def update_val_loss(self, x_val, y_val):
        z = self.forpass(x_val)             # 정방향 계산 수행
        a = self.activation(z)              # 활성화 함수 적용
        a = np.clip(a, 1e-10, 1-1e-10)      # 출력값을 클리핑
        # 로그 손실과 규제 손실을 더하여 리스트에 추가합니다.
        val_loss = np.sum(-(y_val * np.log(a) + (1-y_val)*np.log(1-a)))
        self.val_losses.append((val_loss + self.reg_loss()) / len(y_val))
