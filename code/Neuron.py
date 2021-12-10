class Neuron:
  def __init__(self):
    # 초기화 작업 수행
    self.w = 1.0
    self.b = 1.0

  # 필요한 메서드 추가

  def forpass(self, x):
    # 정방향 계산 메서드
    # y_hat을 계산하여 반환
    y_hat = x * self.w + self.b
    return y_hat
  
  def backprop(self, x, err):
    # 역방향 계산
    # w그레이디언트와 b그레이디언트 계산하여 반환
    w_grad = x * err
    b_grad = 1 * err
    return w_grad, b_grad

  def fit(self, x, y, epochs=100):
    for i in range(epochs):       # 에포크 반복
      for x_i, y_i in zip(x, y):  # 훈련세트 반복
        y_hat = self.forpass(x_i) # 정방향 계산 (예측)
        err = -(y_i - y_hat)      # 오차 계산
        w_grad, b_grad = self.backprop(x_i, err) # 역방향 계산
        self.w -= w_grad          # 가중치 업데이트
        self.b -= b_grad          # 절편 업데이트