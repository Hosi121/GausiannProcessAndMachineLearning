import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# ランダムなデータを生成
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 線形回帰モデルを作成し、データにフィットさせる
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# 回帰直線をプロットするためのデータを生成
X_new = np.array([[0], [2]])
y_predict = lin_reg.predict(X_new)

# 元のデータと回帰直線をプロット
plt.scatter(X, y)
plt.plot(X_new, y_predict, color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Example')
plt.show()

# 回帰係数と切片を表示
print(f'回帰係数: {lin_reg.coef_[0][0]}')
print(f'切片: {lin_reg.intercept_[0]}')