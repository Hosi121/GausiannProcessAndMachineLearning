import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# ランダムなデータを生成
np.random.seed(0)
X = np.random.rand(100, 2)  # 2つの特徴量を持つデータセット
y = 4 + 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100)

# 重回帰モデルを作成し、データにフィットさせる
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# 予測用のグリッドを作成
x0, x1 = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
X_new = np.c_[x0.ravel(), x1.ravel()]
y_predict = lin_reg.predict(X_new).reshape(x0.shape)

# 3Dプロット
fig = go.Figure(data=[
    go.Scatter3d(x=X[:, 0], y=X[:, 1], z=y, mode='markers', marker=dict(color='blue', size=5, opacity=0.5)),
    go.Surface(x=x0, y=x1, z=y_predict, opacity=0.5, colorscale='Reds')
])
fig.update_layout(title='Multiple Linear Regression', scene=dict(xaxis_title='X1', yaxis_title='X2', zaxis_title='y'))
fig.show()