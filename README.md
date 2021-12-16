# 요약 정리

# Deep-Learning

# 간단 용어 정리
- 훈련데이터 : 여러개의 샘플과 속성으로 이루어짐 (샘플수, 속성수)
- 타깃데이터 : 샘플에 대한 값으로 이루어짐 (샘플에 대한 타깃수,)
- 예측값 : 모델이 예측한 값, $\hat{y}$
- 가중치 : w
- 절편 : b
- 변화율 : rate 또는 grad
- 일반화 성능 : 훈련된 모델의 실전 성능
- 모델 파라미터 : 모델이 알아서 조정해주는 파라미터 - 가중치, 절편 등
- 하이퍼 파라미터 : 모델이 아닌, 사람이 지정해주어야 하는 파라미터 - 어떤 손실함수를 사용할 것인지 등
<br><br>

# 경사하강법
- 어떤 '손실 함수'가 정의되었을 때 손실 함수의 값이 최소가 되는 지점을 찾는 방법
- w와 b를 조정하는 것
## 경사하강법의 종류
- 확률적 경사 하강법 : 1개의 샘플을 중복되지 않게 무작위로 선택하여 그레이디언트를 계산하는 방법
- 배치 경사 하강법 : 전체 샘플을 모두 선택하여 그레이디언트를 계산하는 방법
- 미니 배치 경사 하강법 : 확률 경사하강법 + 배치 경사 하강법, 배치의 크기를 작게 적용하여 무작위로 선택하여 그레이디언트를 계산하는 방법


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
- <a id=hingeLoss style="text-decoration:none; color:inherit">힌지 손실 함수(서포트 벡터 머신 - SVM)</a>

# 모델
- 훈련 데이터로 학습된 머신러닝 알고리즘
## 모델의 종류
- 선형 회귀 모델
    - 1차 함수의 모델
    - [제곱오차 (오차 역전파) 손실함수](#squaredError)를 사용할 수 있다.
- 로지스틱 회귀 모델
    - 이진 크로스 엔트로피(binary cross entropy) 또는 [로지스틱(logistic) 손실 함수](#logisticLoss)를 사용하는 모델

# 데이터 준비 및 전처리
## 1. 데이터 준비
### 데이터 모양(data.shape, target.shape)
- 입력 데이터 => (샘플의 수, 특성)
- 타깃 데이터 => (샘플의 수,)
### 데이터 스케일 조정
- 표준화
    - $z = \frac{x-\mu}{s}$
    - 반드시 '훈련데이터 세트의 평균, 표준편차'만을 이용하여 '검증 데이터 세트'와 '테스트 데이터 세트'의 표준화를 진행해야한다.

# 일반적인 딥러닝 과정
1. 모델 과대적합
    - 모델이 과대적합이 되었는지 확인하는 과정 
    - (에포크, 손실 함수의 그래프로 확인)
2. 규제를 통한 적절한 모델로 조정

# 규제
- 가중치가 작을수록 일반화가 좋다.
- 규제를 통해 가중치를 작게 할 수 있다.
## 규제의 종류
- L1 규제 (라쏘 - Lasso)
    - L1 노름 = $\vert\vert{w}\vert\vert_1 = \sum\limits_{i=1}^{n}\vert{w_i}\vert$
    - $L = -(y\log(a) + (1-y)\log(1-a)) + α\sum\limits_{i=1}^{n}\vert{w_i}\vert$
    - $\frac{\delta}{\delta w}L = -(y-a)x + α * sign(w)$
    - <code> w_grad += alpha * np.sign(w)</code>
- L2 규제
    - L2노름(유클리드 거리) = $\vert\vert{w}\vert\vert_2 = \sqrt{\sum\limits_{i=1}^{n}\vert{w_i}\vert^2}$
    - $L = -(y\log(a) + (1-y)\log(1-a)) + \frac{1}{2}α\sum\limits_{i=l}^{n}\vert{w_i}\vert^2$ 
    - $\frac{\delta}{\delta w}L = -(y-a)x + α * w$
    - <code>w_grad += alpha * w</code>
