# 你将使用单变量的线性回归来预测食品卡车的利润。假设你是a公司的CEO并正在考虑在不同的城市开设一家连锁餐厅。
# 该连锁店已经在多个城市拥有卡车，并且你拥有有关城市的人口和利润的数据。
# 你可以使用这些数据来帮助你选择接下来在那个城市发展。
# 文件ex1data1.txt包含了线性回归问题的数据集，其中第一列是城市人口，第二列是食品卡车在该城市的利润。共97个样本

import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def loaddata(filename):
    data = np.loadtxt(filename, delimiter=',')
    return data


def plotData(x, y):
    plt.plot(x, y, 'rx', ms=10)
    plt.xlabel('Population of City in 10,000')
    plt.ylabel('Profit in $10,000')
    plt.show()


def cost(x, y, theta):
    ly = np.size(y, 0)
    cost = (x.dot(theta) - y).dot(x.dot(theta) - y) / (2 * ly)
    return cost


# 迭代计算theta
def gradientDescent(x, y, theta, alpha, num_iters):
    m = np.size(y, 0)
    j_history = np.zeros((num_iters,))

    for i in range(num_iters):
        deltaJ = x.T.dot(x.dot(theta) - y) / m
        theta = theta - alpha * deltaJ
        j_history[i] = cost(x, y, theta)
        # print( j_history[i])
    return theta, j_history


if __name__ == '__main__':
    data = loaddata("ex1data1.txt")
    x = data[:, 0]  # 1*N
    y = data[:, 1]  # 1*N
    m = np.size(y, 0)
    x = np.vstack((np.ones((m,)), x)).T  # 97*2
    theta = np.zeros((2,))  # 初始化参数
    iterations = 1500  # 迭代次数
    alpha = 0.01  # 学习率
    j = cost(x, y, theta)
    print(j)
    theta, j_history = gradientDescent(x, y, theta, alpha, iterations)

    # plt.plot(x[:, 1], y, 'rx', ms=10, label='Training data')
    # plt.plot(x[:, 1], x.dot(theta), '-', label='Linear regression')
    # plt.xlabel('Population of City in 10,000')
    # plt.ylabel('Profit in $10,000')
    # plt.legend(loc='upper right')
    # plt.show()

    # 查看损失函数图像
    # iter = np.linspace(1,1500,1500)
    # plt.plot(iter, j_history, '-')
    # plt.xlabel("iteration")
    # plt.ylabel("cost value")
    # plt.show()

    # Predict values for population sizes of 35,000 and 70,000
    predict1 = np.array([1, 3.5]).dot(theta)
    print('For population = 35,000, we predict a profit of ', predict1 * 10000)
    predict2 = np.array([1, 7.0]).dot(theta)
    print('For population = 70,000, we predict a profit of ', predict2 * 10000)

    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    J_vals = np.zeros((np.size(theta0_vals, 0), np.size(theta1_vals, 0)))

    for i in range(np.size(theta0_vals, 0)):
        for j in range(np.size(theta1_vals, 0)):
            t = np.array([theta0_vals[i], theta1_vals[j]])
            J_vals[i, j] = cost(x, y, t)

    # 绘制三维图像
    theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(theta0_vals, theta1_vals, J_vals.T)
    ax.set_xlabel(r'$\theta$0')
    ax.set_ylabel(r'$\theta$1')

    # 绘制等高线图
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.contour(theta0_vals, theta1_vals, J_vals.T, np.logspace(-2, 3, 20))
    ax2.plot(theta[0], theta[1], 'rx', ms=10, lw=2)
    ax2.set_xlabel(r'$\theta$0')
    ax2.set_ylabel(r'$\theta$1')
    plt.show()
