from sklearn.linear_model import SGDClassifier

# 데이터 준비
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
x = cancer.data
y = cancer.target

# 훈련 세트와 테스트 세트 나누기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)


# sgd 분류기 생성
sgd = SGDClassifier(loss='log', max_iter=100, tol=1e-3, random_state=42)

# 모델 훈련
sgd.fit(x_train, y_train)

# 모델 성능 test
print('score = ', sgd.score(x_test, y_test))

# 모델 예측
print('predict = ',sgd.predict(x_test[0:10]))

