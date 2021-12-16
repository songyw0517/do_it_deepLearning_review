from model.SingleLayer_fin import SingleLayer
import numpy as np
class DualLayer(SingleLayer):
    def __init__(self, units=10, learning_rate=0.1, l1=0, l2=0):
        self.units = units
        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None
        self.a1 = None
        self.losses = []
        self.val_losses = []
        self.lr = learning_rate
        self.l1 = l1
        self.l2 = l2
    def forpass(self, x):
        z1 = np.dot(x, self.w1) + self.b1         # 첫 번째 층의 선형식을 계산
        self.a1 = self.activation(z1)             # 활성화 함수 적용
        z2 = np.dot(self.a1, self.w2) + self.b2   # 두번째 층의 선형식 계산
        return z2
    def backprop(self, x, err):
        m = len(x)
        # 출력층의 가중치와 절편에 대한 그레이디언트를 계산합니다.
        w2_grad = np.dot(self.a1.T, err)
        b2_grad = np.sum(err) / m
        # 시그모이드 함수까지 그레이디언트를 계산합니다.
        err_to_hidden = np.dot(err, self.w2.T) * self.a1 * (1 - self.a1)
        # 은닉층의 가중치와 절편에 대한 그레이디언트를 계산합니다.
        w1_grad = np.dot(x.T, err_to_hidden) / m
        b1_grad = np.sum(err_to_hidden, axis=0) / m
        return w1_grad, b1_grad, w2_grad, b2_grad
    
    def init_weights(self, n_features):
        self.w1 = np.ones((n_features, self.units))
        self.b1 = np.zeros(self.units)
        self.w2 = np.ones((self.units, 1))
        self.b2 = 0

    def init_weights_random(self, n_features):
        np.random.seed(42)
        self.w1 = np.random.normal(0, 1, 
                                (n_features, self.units)) # (특성 개수, 은닉층의 크기)
        self.b1 = np.zeros(self.units)
        self.w2 = np.random.normal(0, 1, (self.units,1)) # (은닉층의 크기, 1)
        self.b2 = 0

    def fit(self, x, y, epochs=100, x_val=None, y_val=None):
        y = y.reshape(-1, 1)                # 타깃을 열 벡터로 바꿉니다.
        y_val = y_val.reshape(-1, 1)        # 검증용 타깃을 열 벡터로 바꿉니다.
        m = len(x)                          # 샘플의 개수를 저장합니다.
        self.init_weights(x.shape[1])       # 은닉층과 출력층의 가중치를 초기화합니다.

        # epochs만큼 반복합니다.
        for i in range(epochs):
            a = self.training(x, y, m)
            # 안전한 로그 계산을 위해 클리핑합니다.
            a = np.clip(a, 1e-10, 1-1e-10)

            # 로그 손실과 규제 손실을 더하여 리스트에 추가합니다.
            loss = np.sum(-(y*np.log(a) + (1-y)*np.log(1-a)))
            self.losses.append((loss + self.reg_loss()) / m)

            # 검증 세트에 대한 손실을 계산합니다.
            self.update_val_loss(x_val, y_val)

    def training(self, x, y, m):
        z = self.forpass(x)
        a = self.activation(z)
        err = -(y - a)

        # 오차를 역전파하여 그레이디언트를 계산합니다.
        w1_grad, b1_grad, w2_grad, b2_grad = self.backprop(x, err)

        # 그레이디언트에서 페널티 항의 미분값을 뺍니다.
        w1_grad += (self.l1 * np.sign(self.w1) + self.l2 * self.w1) / m
        w2_grad += (self.l1 * np.sign(self.w2) + self.l2 * self.w2) / m

        # 은닉층의 가중치와 절편을 업데이트 합니다.
        self.w1 -= self.lr * w1_grad
        self.b1 -= self.lr * b1_grad

        # 출력층의 가중치와 절편을 업데이트 합니다.
        self.w2 -= self.lr * w2_grad
        self.b2 -= self.lr * b2_grad

        return a

    def reg_loss(self):
        # 은닉층과 출력층의 가중치에 규제를 적용합니다.
        return self.l1 * (np.sum(np.abs(self.w1)) + np.sum(np.abs(self.w2))) + \
            self.l2 / 2 * (np.sum(self.w1**2) + np.sum(self.w2**2))

if __name__ == '__main__':
    # 데이터 준비
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
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
    # StandardScaler로도 가능하다
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    dual_layer = DualLayer(l2=0.01)
    dual_layer.fit(x_train_scaled, y_train, x_val=x_val_scaled, y_val=y_val, epochs=20000)
    print(dual_layer.score(x_val_scaled, y_val))