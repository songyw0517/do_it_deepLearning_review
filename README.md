# 요약 정리

# Deep-Learning

# 간단 용어 정리
- 훈련데이터 : 여러개의 샘플과 속성으로 이루어짐 (샘플수, 속성수)
- 타깃데이터 : 샘플에 대한 값으로 이루어짐 (샘플에 대한 타깃수,)
- 예측값 : 모델이 예측한 값, $\hat{y}$
- 가중치 : w
- 절편 : b
- 변화율 : rate 또는 grad
- 경사하강법 : 어떤 '손실 함수'가 정의되었을 때 손실 함수의 값이 최소가 되는 지점을 찾는 방법
    - w와 b를 조정하는 것
<br><br>

# 손실함수
- 모델의 예측값과 타깃값의 차이를 함수로 정의한 것
## 손실함수의 종류
- <a id=squaredError style="text-decoration:none; color:inherit">제곱오차 (오차 역전파)</a>
    - $SE = (y-\hat{y})^2$
    - w_grad = $-(y-\hat{y})x$
    - w = w - w_grad = w + $(y-\hat{y})x$
    - b_gard = $-(y-\hat{y})$
    - b = b - b_grad = b + $(y-\hat{y})$
    <br><br>
- <a id=logisticLoss style="text-decoration:none; color:inherit">로지스틱 손실 함수</a>
    - $L = -(y\log{(a)} + (1-y)\log{(1-a))}$
      - y = 1 (양성 클래스인 경우) -> $L = -\log{(a)}$
      - y = 0 (음성 클래스인 경우) -> $L = -\log{(1-a)}$
    - w_grad = $-(y-a)x_i$
    - b_grad = $-(y-a)$

# 모델
- 훈련 데이터로 학습된 머신러닝 알고리즘
## 모델의 종류
- 선형 회귀 모델
    - 1차 함수의 모델
    - [제곱오차 (오차 역전파) 손실함수](#squaredError)를 사용할 수 있다.
- 로지스틱 회귀 모델
    - 이진 크로스 엔트로피(binary cross entropy) 또는 [로지스틱(logistic) 손실 함수](#logisticLoss)를 사용하는 모델
# 