import numpy as np
def predict(self, x):
    z = self.forpass(x)
    return z > 0

def update_val_loss(self, x_val, y_val):
    z = self.forpass(x_val)
    a = self.activation(z)
    a = np.clip(a, 1e-10, 1-1e-10)
    # 로그 손실과 규제 손실을 더하여 리스트에 추가합니다.
    val_loss = np.sum(-(y_val * np.log(a) + (1-y_val)*np.log(1-a)))
    self.val_losses.append((val_loss + self.reg_loss()) / len(y_val))