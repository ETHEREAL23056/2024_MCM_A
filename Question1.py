import math

import numpy as np
import pandas as pd
from scipy.optimize import fsolve, root
from scipy.integrate import quad
import matplotlib.pyplot as plt


# 定义几何方程
# 以a x ^ 2 + b x cosx + cx + d cosx + e = 0的形式求解
def equation(x, a, b, c, d, e):
    return a * x ** 2 + (b * x + d) * np.cos(x) + c * x + e


# 定义阿基米德螺线的积分函数
def archimedes_integrand_function(t, a, b):
    r = a + b * t
    return np.sqrt(b ** 2 + r ** 2)


# 计算阿基米德螺线的长度
def get_archimedes_length(start, end, b):
    length, _ = quad(lambda x: archimedes_integrand_function(t=x, a=0, b=b), start, end)
    return length


# 定义目标函数：使其结果为固定弧长
def objective(t, end, b, L):
    return get_archimedes_length(start=end - t, end=end, b=b) - L


# 获取某时刻龙头的位置
def get_head_position(time_point, tail, b):
    theta_initial_guess = np.asarray([1])
    cover = time_point * 100
    d_theta = fsolve(lambda x: objective(x, end=tail, b=b, L=cover), x0=theta_initial_guess)
    return tail - d_theta, d_theta


# 获取某位置的速度
# start代表前面的点，end代表后面的点
def get_speed(start, end, v0):
    delta_theta = end - start
    return v0 * np.sqrt((1 + end ** 2) / (1 + start ** 2)) * np.abs(
        (start - end * np.cos(delta_theta) - start * end * np.sin(delta_theta)) / (
                end - start * np.cos(delta_theta) + start * end * np.sin(delta_theta)))


# 画出对应的位置
def draw_positions(b, positions):
    i = np.arange(0, 32 * np.pi, 0.01)
    x = b * i * np.cos(i)
    y = b * i * np.sin(i)
    plt.plot(x, y, color='g', linewidth=1)
    locations = []
    for position in positions:
        x = b * position * np.cos(position)
        y = b * position * np.sin(position)
        locations.append([float(x), float(y)])
        plt.scatter(x, y, c='r')
    for i in range(len(locations) - 1):
        print(np.sqrt((locations[i + 1][0] - locations[i][0]) ** 2 + (locations[i + 1][1] - locations[i][1]) ** 2))
        plt.plot([locations[i + 1][0], locations[i][0]], [locations[i + 1][1], locations[i][1]], c='b')
    plt.grid(linestyle="--")


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    dx = 55
    theta, l_head, l_tail, b_base = 2 * 16 * np.pi, 286, 165, dx / (2 * np.pi)

    # 求解龙头的位置-以极坐标的形式
    head_positions = []
    times = list(range(0, 301))
    # times.append(430)
    for i in times:
        pos, d = get_head_position(time_point=i, tail=theta, b=b_base)
        head_positions.append(pos)
    # 求解后续的位置
    head_backs, body_heads, body_backs = [], [], []
    # 求解后续的速度
    head_backs_v, body_backs_v = [], []
    for head_position in head_positions:
        # 求解龙头后半段的位置
        x_initial_guess = np.array([.5])
        delta = root(lambda x: equation(x, 1, -2 * head_position, 2 * head_position, -2 * head_position ** 2,
                                        2 * head_position ** 2 - l_head ** 2 / b_base ** 2),
                     x0=x_initial_guess, method="lm").x[0]
        head_back = head_position + delta
        head_backs.append(head_back)
        # 初始时龙头的速度为100cm/s固定
        head_back_v = get_speed(head_position, head_back, 100)
        head_backs_v.append(head_back_v)
        # 递推求解后续龙身的位置-第x节的尾部=第x+1节的头部
        front, end = head_back, 0
        body_head_positions, body_back_positions = [], []
        # 递推求解后续龙身的速度
        v = head_back_v
        body_vs = [100, v]
        for index in range(1, 223):
            delta = root(lambda x: equation(x, 1, -2 * front, 2 * front, -2 * front ** 2,
                                            2 * front ** 2 - l_tail ** 2 / b_base ** 2), x0=x_initial_guess,
                         method="lm").x[0]
            end = front + delta
            # 记录全部数据
            body_head_positions.append(front)
            body_back_positions.append(end)
            body_vs.append(get_speed(front, end, v))
            # 迭代更新
            v = get_speed(front, end, v)
            front = end
        body_head_positions.append(end)
        body_heads.append(body_head_positions)
        body_backs.append(body_back_positions)
        body_backs_v.append(body_vs)

    # 显示结果并存储在对应的文件中
    body_heads, body_backs = pd.DataFrame(body_heads), pd.DataFrame(body_backs)
    body_heads.insert(loc=0, column="head", value=head_positions)
    df_pos = dict()
    for i in range(body_heads.shape[0]):
        df_pos[i] = []
        for j in range(body_heads.shape[1]):
            po = body_heads.iloc[i, j]
            df_pos[i].append(b_base * po * math.cos(po) / 100)
            df_pos[i].append(b_base * po * math.sin(po) / 100)
    df_pos = pd.DataFrame.from_dict(df_pos)
    df_pos = df_pos.applymap(lambda x: '%.6f' % x)
    df_pos.to_csv("result1_position.csv", index=False)

    body_backs_v = pd.DataFrame(body_backs_v)
    df_vs = dict()
    for i in range(body_backs_v.shape[0]):
        df_vs[i] = []
        for j in range(body_backs_v.shape[1]):
            v = float(body_backs_v.iloc[i, j]) / 100
            df_vs[i].append(v)
    df_vs = pd.DataFrame.from_dict(df_vs)
    df_vs = df_vs.applymap(lambda x: '%.6f' % x)
    df_vs.to_csv("result1_velocity.csv", index=False)

    for i in [-1]:
        plt.figure("t = {time} s".format(time=times[i]))
        draw_positions(b=b_base, positions=body_heads.iloc[i, :])
    plt.show()
