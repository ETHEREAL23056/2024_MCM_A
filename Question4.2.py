import math
import copy
import numpy as np
import pandas as pd
import pylab as p
import sympy as sp
from sympy import S
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
    return get_archimedes_length(start=end, end=end + t, b=b) - L


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


# 用于计算后点在圆上的情况
def cal_to_circle_function(x_front, y_front, x_center, y_center, l, r):
    x, y = sp.symbols('x y')
    # 保持间距
    eq1 = sp.Eq((x - x_front) ** 2 + (y - y_front) ** 2 - l ** 2, 0)
    # 位于曲线上
    eq2 = sp.Eq((x - x_center) ** 2 + (y - y_center) ** 2 - r ** 2, 0)

    equations = [eq1, eq2]
    solutions = sp.nonlinsolve(equations, [x, y])
    return solutions


# 用于计算后点在螺线上的情况
def cal_to_rol_function(x_front, y_front, l, b, beg):
    t = sp.symbols('theta')
    # 保持间距且位于曲线之上
    eq = sp.Eq((b * t * sp.cos(t) - x_front) ** 2 + (b * t * sp.sin(t) - y_front) ** 2 - l ** 2, 0)
    solutions = sp.nsolve(eq, t, beg)
    return solutions


def cal_to_rol_function_end(x_front, y_front, l, b, beg):
    t = sp.symbols('theta')
    # 保持间距且位于曲线之上
    eq = sp.Eq((b * (t - math.pi) * sp.cos(t) - x_front) ** 2 + (
            b * (t - math.pi) * sp.sin(t) - y_front) ** 2 - l ** 2, 0)
    solutions = sp.nsolve(eq, t, beg)
    return solutions


# 所处在的相对位置判断
# 1-外螺线/2-第一部分/3-第二部分/4-内螺线
def function_check(x, y, start, b, x_big, y_big, r_big, x_small, y_small, r_small):
    theta = math.sqrt(x ** 2 + y ** 2) / b
    if math.fabs(math.tan(theta) - y / x) < 1e-6 and theta > start:
        return 1
    if math.fabs((x - x_big) ** 2 + (y - y_big) ** 2 - r_big ** 2) < 1e-6:
        return 2
    if math.fabs((x - x_small) ** 2 + (y - y_small) ** 2 - r_small ** 2) < 1e-6:
        return 3
    if math.fabs(math.tan(theta + math.pi) - y / x) < 1e-6 and theta + math.pi > start + math.pi:
        return 4


def get_ex_status(t):
    # 计算盘入之前的运动信息-Q1-对称备用
    theta_initial_guess = np.asarray([1])
    ex_head_positions = []
    for time_point in np.linspace(1, t + 1, int(2 * t)):
        cover = time_point * 100
        d_theta = fsolve(lambda x: objective(x, end=start, b=b_base, L=cover), x0=theta_initial_guess)
        begin_position = start + d_theta[0]
        ex_head_positions.append(begin_position)
    head_backs, body_heads, body_backs = [], [], []
    head_backs_v, body_backs_v = [], []
    for ex_head_position in ex_head_positions:
        x_initial_guess = np.array([.5])
        delta = root(lambda x: equation(x, 1, -2 * ex_head_position, 2 * ex_head_position, -2 * ex_head_position ** 2,
                                        2 * ex_head_position ** 2 - l_head ** 2 / b_base ** 2),
                     x0=x_initial_guess, method="lm").x[0]
        head_back = ex_head_position + delta
        head_backs.append(head_back)
        head_back_v = get_speed(ex_head_position, head_back, 1)
        head_backs_v.append(head_back_v)
        front, end = head_back, 0
        body_head_positions, body_back_positions = [], []
        v = head_back_v
        body_vs = [1, v]
        for index in range(1, 223):
            delta = root(lambda x: equation(x, 1, -2 * front, 2 * front, -2 * front ** 2,
                                            2 * front ** 2 - l_tail ** 2 / b_base ** 2), x0=x_initial_guess,
                         method="lm").x[0]
            end = front + delta
            body_head_positions.append(front)
            body_back_positions.append(end)
            body_vs.append(get_speed(front, end, v))
            v = get_speed(front, end, v)
            front = end
        body_head_positions.append(end)
        body_heads.append(body_head_positions)
        body_backs.append(body_back_positions)
        body_backs_v.append(body_vs)
    return ex_head_positions, body_heads, body_backs, body_backs_v


if __name__ == "__main__":
    # 定义基础信息
    dx, d, r, length, radius = 170, 15, math.sqrt(27.5 ** 2 + 15 ** 2), 27.5, 450
    l_head, l_tail, b_base = 286, 165, dx / (2 * np.pi)
    # 确定进入的位置
    start = radius / b_base
    x_head, y_head = b_base * start * math.cos(start), b_base * start * math.sin(start)
    x_center_big, y_center_big = (2 * b_base * math.sin(start) + b_base * start * math.cos(start)) / 3, (
            -2 * b_base * math.cos(start) + b_base * start * math.sin(start)) / 3
    x_center_small, y_center_small = (b_base * math.sin(start) - 2 * b_base * start * math.cos(start)) / 3, (
            -b_base * math.cos(start) - 2 * b_base * start * math.sin(start)) / 3
    x_p, y_p = -b_base * start * math.cos(start) / 3, -b_base * start * math.sin(start) / 3
    r1 = 2 * b_base * math.sqrt(1 + start ** 2) / 3
    r2 = b_base * math.sqrt(1 + start ** 2) / 3

    ex_head_positions, body_heads, body_backs, body_backs_v = get_ex_status(100)
    # # 显示结果并存储在对应的文件中
    # ex_head_positions, body_heads, body_backs, body_backs_v = get_ex_status(100)
    # body_heads, body_backs = pd.DataFrame(body_heads), pd.DataFrame(body_backs)
    # body_heads.insert(loc=0, column="head", value=ex_head_positions)
    # df_pos = dict()
    # for i in range(body_heads.shape[0]):
    #     df_pos[i] = []
    #     for j in range(body_heads.shape[1]):
    #         po = body_heads.iloc[i, j]
    #         df_pos[i].append(b_base * po * math.cos(po) / 100)
    #         df_pos[i].append(b_base * po * math.sin(po) / 100)
    # df_pos = pd.DataFrame.from_dict(df_pos)
    # df_pos = df_pos.applymap(lambda x: '%.6f' % x)
    # df_pos.to_csv("result4_position_ex.csv", index=False)
    #
    # body_backs_v = pd.DataFrame(body_backs_v)
    # df_vs = dict()
    # for i in range(body_backs_v.shape[0]):
    #     df_vs[i] = []
    #     for j in range(body_backs_v.shape[1]):
    #         v = float(body_backs_v.iloc[i, j]) / 100
    #         df_vs[i].append(v)
    # df_vs = pd.DataFrame.from_dict(df_vs)
    # df_vs = df_vs.applymap(lambda x: '%.6f' % x)
    # df_vs.to_csv("result4_velocity_ex.csv", index=False)

    # 确定进入的时间
    times = np.linspace(0, 100, 101)
    head_positions = []

    # 先单独求解龙头的坐标
    # 龙头走完第一部分的时间
    t1 = 4 * b_base * (math.pi - math.atan(start)) * math.sqrt(1 + start ** 2) / 300
    # 龙头走完第二部分的时间
    t2 = 2 * b_base * (math.pi - math.atan(start)) * math.sqrt(1 + start ** 2) / 100

    alpha = math.acos((start * math.cos(start) - math.sin(start)) / math.sqrt(1 + start ** 2))
    alpha = 2 * math.pi - math.acos((math.cos(alpha)))
    beta = math.acos((start * math.cos(start) + math.sin(start)) / math.sqrt(1 + start ** 2))
    beta = 2 * math.pi - math.acos(math.cos(beta))
    # 第一部分的角速度
    w1 = 300 / (2 * b_base * math.sqrt(1 + start ** 2))
    # 第二部分的角速度
    w2 = 300 / (b_base * math.sqrt(1 + start ** 2))

    for time in times:
        if time <= t1:
            # 龙头在第一部分上
            x_head = x_center_big + r1 * math.cos(alpha - w1 * time)
            y_head = y_center_big + r1 * math.sin(alpha - w1 * time)
            head_positions.append((x_head, y_head))
        elif time <= t2:
            # 龙头在第二部分上
            x_head = x_center_small + r2 * math.cos(beta + w2 * (time - t1))
            y_head = y_center_small + r2 * math.sin(beta + w2 * (time - t1))
            head_positions.append((x_head, y_head))
        else:
            # 龙头在对称的螺线上
            delta_t = time - t2
            cover = delta_t * 100
            theta_initial_guess = np.asarray([1])
            d_theta = fsolve(lambda x: objective(x, end=start + math.pi, b=b_base, L=cover), x0=theta_initial_guess)
            pos = start + math.pi + d_theta[0]
            head_positions.append(((pos - math.pi) * b_base * math.cos(pos), (pos - math.pi) * b_base * math.sin(pos)))

    # 再求龙尾的位置
    # 龙尾进入第一部分的时间
    t0 = 2 * math.asin(l_head / (2 * r1)) / w1
    # 龙尾走完第一部分的时间
    t3 = t1 + 2 * math.asin(l_head / (2 * r2)) / w2
    # 龙尾走完第二部分的时间
    theta = sp.symbols('theta')
    a_ = 1
    b_ = -2 * start
    c_ = 2 * start
    d_ = -2 * start ** 2
    e_ = 2 * start ** 2 - l_head ** 2 / b_base ** 2
    x_initial_guess = np.array([.5])
    delta = root(lambda x: equation(x, a=a_, b=b_, c=c_, d=d_, e=e_), x0=x_initial_guess, method="lm").x[0]
    head_position = start + math.pi + delta
    dl = get_archimedes_length(start=start + math.pi, end=head_position, b=b_base)
    t4 = t2 + dl / 100

    head_back_positions = []
    for i in range(len(head_positions)):
        x_head, y_head = head_positions[i]
        if times[i] <= t0:
            # 前-第一部分/后-螺线部分
            res = cal_to_rol_function(x_front=x_head, y_front=y_head, l=l_head, b=b_base, beg=start)
            head_back_positions.append((res * b_base * math.cos(res), res * b_base * math.sin(res)))
        elif times[i] <= t1:
            # 前后均位于第一部分上
            res = cal_to_circle_function(x_front=x_head, y_front=y_head, x_center=x_center_big, y_center=y_center_big,
                                         l=l_head, r=r1)
            res = res & S.Complexes
            for point in res:
                # 选择点作为结果
                k = point[1] / point[0]
                if (point[0] * x_head < 0 and k < y_head / x_head) or (point[0] * x_head > 0 and k > y_head / x_head):
                    x_head, y_head = point
                    head_back_positions.append(point)
        elif times[i] <= t3:
            # 前-第二部分/后-第一部分
            res = cal_to_circle_function(x_front=x_head, y_front=y_head, x_center=x_center_big, y_center=y_center_big,
                                         l=l_head, r=r1)
            res = res & S.Complexes
            for point in res:
                # 选择点作为结果
                k = point[1] / point[0]
                if (point[0] * x_head > 0 and k > y_head / x_head) or (point[0] * x_head < 0 and k < y_head / x_head):
                    x_head, y_head = point
                    head_back_positions.append(point)
        elif times[i] <= t2:
            # 前后均位于第二部分上
            res = cal_to_circle_function(x_front=x_head, y_front=y_head, x_center=x_center_small,
                                         y_center=y_center_small,
                                         l=l_head, r=r2)
            res = res & S.Complexes
            for point in res:
                # 选择点作为结果
                k = point[1] / point[0]
                if (point[0] * x_head > 0 and k < y_head / x_head) or (point[0] * x_head < 0 and k > y_head / x_head):
                    x_head, y_head = point
                    head_back_positions.append(point)
        elif times[i] <= t4:
            # 前-螺线/后-第二部分
            res = cal_to_circle_function(x_front=x_head, y_front=y_head, x_center=x_center_small,
                                         y_center=y_center_small, l=l_head, r=r2)
            res = res & S.Complexes
            for point in list(res)[::-1]:
                # 选择点作为结果
                k = point[1] / point[0]
                if (point[0] * x_head > 0 and k < y_head / x_head) or (point[0] * x_head < 0 and k > y_head / x_head):
                    x_head, y_head = point
                    head_back_positions.append(point)
                    break
        else:
            # 均在螺线上
            ro = math.sqrt(x_head ** 2 + y_head ** 2) / b_base
            x_initial_guess = np.array([.5])
            delta = \
                root(lambda x: equation(x, 1, 2 * ro, -2 * ro, -2 * ro ** 2, 2 * ro ** 2 - l_head ** 2 / b_base ** 2),
                     x0=x_initial_guess, method="lm").x[0]
            head_back = ro - delta + math.pi
            head_back_positions.append(
                ((head_back - math.pi) * b_base * math.cos(head_back),
                 (head_back - math.pi) * b_base * math.sin(head_back)))

    # 可视化呈现
    plt.scatter(x_center_big, y_center_big, c='b')
    plt.scatter(x_center_small, y_center_small, c='y')
    circle_big = plt.Circle((x_center_big, y_center_big), r1, fill=False)
    circle_small = plt.Circle((x_center_small, y_center_small), r2, fill=False)
    plt.gcf().gca().add_artist(circle_big)
    plt.gcf().gca().add_artist(circle_small)
    for point in ex_head_positions:
        plt.scatter(point * b_base * math.cos(point), point * b_base * math.sin(point), c='g')
    for point in head_positions:
        plt.scatter(point[0], point[1], c='r')
    for point in head_back_positions:
        plt.scatter(point[0], point[1], c='g')
    for i in range(len(head_back_positions)):
        print((head_back_positions[i][0] - head_positions[i][0]) ** 2 + (
                head_back_positions[i][1] - head_positions[i][1]) ** 2)
    plt.axis('equal')
    plt.grid(linestyle="--")
    plt.show()

    # 求解后续各个点的运动状态
    ex_head_positions, body_heads, body_backs, body_backs_v = get_ex_status(100 + t2)
    arr = np.linspace(0, 100 + t2 + 1, int(2 * (100 + t2)))
    res_positions, res_v = {}, {}
    for i in range(101):
        if i < t2:
            res_v[i] = body_backs_v[i]
        else:
            d_x = i - t2
            index = None
            for j in range(arr.shape[0] - 1):
                if arr[j] <= d_x <= arr[j + 1]:
                    if d_x - arr[j] >= arr[j + 1] - d_x:
                        index = j + 1
                    else:
                        index = j
            res_positions[i] = body_heads[index]
            res_v[i] = body_backs_v[index]
    df_vs = pd.DataFrame.from_dict(res_v)
    df_vs = df_vs.applymap(lambda x: '%.6f' % x)
    df_vs.to_csv("result4_velocity_ba.csv", index=False)
