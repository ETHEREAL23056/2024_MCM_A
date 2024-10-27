import math

import numpy as np
import pandas as pd
import pylab as p
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


# 碰撞检查
def collide_check(positions, b, width, length):
    node_list = positions.copy()
    for j in range(len(positions) - 1):
        position = positions[j]
        index = None
        goal = position + 2 * np.pi
        for K in range(len(node_list) - 1):
            if node_list[K + 1] >= goal >= node_list[K]:
                index = K
        if index is None:
            continue
        else:
            theta1, theta2 = node_list[index], node_list[index + 1]
            x_base, y_base = b * position * np.cos(position), b * position * np.sin(position)
            x_center, y_center = b * theta1 * np.cos(theta1), b * theta1 * p.sin(theta1)
            k = (theta1 * np.sin(theta1) - theta2 * np.sin(theta2)) / (
                    theta1 * np.cos(theta1) - theta2 * np.cos(theta2))
            s = np.abs(k * (x_base - x_center) + y_center - y_base) / np.sqrt(1 + k ** 2)
            # 后侧检验
            nex = positions[j + 1]
            x_next, y_next = b * nex * np.cos(nex), b * nex * np.sin(nex)
            k_front = (y_next - y_base) / (x_next - x_base)
            alpha_front = math.atan(math.fabs((k - k_front) / (1 + k * k_front)))
            if s <= width * (1 + np.cos(alpha_front)) + length * np.sin(alpha_front):
                return False
            # 前侧检验-除去龙头不用检验
            if j != 0:
                last = positions[j - 1]
                x_last, y_last = b * last * np.cos(last), b * last * np.sin(last)
                k_back = (y_last - y_base) / (x_last - x_base)
                alpha_back = math.atan(math.fabs((k - k_back) / (1 + k * k_back)))
                if s <= width * (1 + np.cos(alpha_back)) + length * np.sin(alpha_back):
                    return False
    return True


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

    dx, d, r, length = 55, 15, math.sqrt(27.5 ** 2 + 15 ** 2), 27.5
    theta, l_head, l_tail, b_base = 2 * 16 * np.pi, 286, 165, dx / (2 * np.pi)

    # 对时间进行枚举迭代
    head_positions = []
    times = np.linspace(300, 500, 201)
    for i in times:
        pos, _ = get_head_position(time_point=i, tail=theta, b=b_base)
        if pos < 0:
            break
        head_positions.append(pos)

    # 求解临界的位置-存储头部位置
    best_index = None
    for i in range(len(head_positions)):
        head_position = head_positions[i]
        body_head_positions = [head_position]
        # 求解龙头后半段的位置
        x_initial_guess = np.array([.5])
        delta = root(lambda x: equation(x, 1, -2 * head_position, 2 * head_position, -2 * head_position ** 2,
                                        2 * head_position ** 2 - l_head ** 2 / b_base ** 2),
                     x0=x_initial_guess, method="lm").x[0]
        if delta < 0:
            best_index = i - 1
            break
        head_back = head_position + delta
        # 递推求解后续龙身的位置-第x节的尾部=第x+1节的头部
        front, end = head_back, 0
        for index in range(1, 222):
            x_initial_guess = np.array([delta])
            delta = root(lambda x: equation(x, 1, -2 * front, 2 * front, -2 * front ** 2,
                                            2 * front ** 2 - l_tail ** 2 / b_base ** 2), x0=x_initial_guess,
                         method="lm").x[0]
            end = front + delta
            body_head_positions.append(front)
            front = end
        # 从内向外进行碰撞检查
        flag = collide_check(positions=body_head_positions, b=b_base, width=d, length=length)
        if flag is False:
            best_index = i
            break
    print(times[best_index])

    # 求解临界位置的状态
    head_position = head_positions[best_index]
    body_head_positions, body_vs = [head_position], [np.array([100])]
    # 求解龙头后半段的位置
    x_initial_guess = np.array([.5])
    delta = root(lambda x: equation(x, 1, -2 * head_position, 2 * head_position, -2 * head_position ** 2,
                                    2 * head_position ** 2 - l_head ** 2 / b_base ** 2),
                 x0=x_initial_guess, method="lm").x[0]
    head_back = head_position + delta
    # 初始时龙头的速度为100cm/s固定
    head_back_v = get_speed(head_position, head_back, 100)
    body_vs.append(head_back_v)
    # 递推求解后续龙身的位置-第x节的尾部=第x+1节的头部
    front, end = head_back, 0
    # 递推求解后续龙身的速度
    v = head_back_v
    for index in range(1, 223):
        x_initial_guess = np.array([delta])
        delta = root(lambda x: equation(x, 1, -2 * front, 2 * front, -2 * front ** 2,
                                        2 * front ** 2 - l_tail ** 2 / b_base ** 2), x0=x_initial_guess,
                     method="lm").x[0]
        end = front + delta
        body_head_positions.append(front)
        body_vs.append(get_speed(front, end, v))
        # 迭代更新
        v = get_speed(front, end, v)
        front = end
    body_head_positions.append(end)
    body_head_positions, body_vs = pd.DataFrame(body_head_positions), pd.DataFrame(body_vs)
    plt.figure("t = {time} s".format(time=times[best_index]))
    draw_positions(b=b_base, positions=body_head_positions.iloc[:, 0])
    plt.show()

    # 存储结果
    df_res = {'x': [], 'y': [], 'v': []}
    for i in range(body_head_positions.shape[0]):
        pos = body_head_positions.iloc[i, 0]
        df_res['x'].append(b_base * pos * math.cos(pos) / 100)
        df_res['y'].append(b_base * pos * math.sin(pos) / 100)
        df_res['v'].append(body_vs.iloc[i, 0] / 100)
    df_res = pd.DataFrame.from_dict(df_res)
    df_res = df_res.applymap(lambda x: '%.6f' % x)
    df_res.to_csv("result2.csv", index=False)
