from dataclasses import dataclass
import numpy as np

# dh表
@dataclass(frozen=True)
class SCARAParameters:
    # a参数，表示延x方向上z轴的偏距，实际可理解为杆长
    a0: float = 0.0
    a1: float = 1.0
    a2: float = 0.8
    # alpha参数，表示延x方向上z轴的偏角
    alpha0: float = 0.0
    alpha1: float = 0.0
    alpha2: float = 0.0
    # d参数，表示沿z轴方向上x轴的偏距
    d1: float = 0.4
    d2: float = 0.1
    d3_min: float = -0.5
    d3_max: float = 0.0
    # thta参数为关节角，下面为角度限制范围
    theta1_min: float = -np.pi * 2
    theta1_max: float = np.pi * 2
    theta2_min: float = -np.pi * 2
    theta2_max: float = np.pi * 2


P = SCARAParameters()


# 转换矩阵
def DHTransform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    # 改进型dh对应的变换矩阵
    T = np.array([
        [ct, -st, 0, a],
        [st * ca, ct * ca, -sa, -sa * d],
        [st * sa, ct * sa, ca, ca * d],
        [0, 0, 0, 1]
    ])

    return T


# 前向运动学，返回一个转换矩阵
# 三个参数的含义:关节1旋转角，关节2旋转角，关节3位移
def forward_kinematics(theta1: float, theta2: float, d3: float) -> np.ndarray:
    theta1, theta2, d3 = clamp_joint_values(theta1, theta2, d3)
    # a,alpha为i-1,d,theta为i
    T0_1 = DHTransform(P.a0, P.alpha0, P.d1, theta1)
    T1_2 = DHTransform(P.a1, P.alpha1, P.d2, theta2)
    T2_3 = DHTransform(P.a2, P.alpha2, d3, 0.0)

    # @为矩阵乘法符号
    T0_3 = T0_1 @ T1_2 @ T2_3
    return T0_3
    # 目前算出来的结果有如下现象
    # Xt = -Y实， Yt = X实，也就是说，算出来的结果为实际结果绕z轴旋转了90度，
    # 为什么会有这样的结果，是因为该urdf再初始化加载scara时，默认的朝向是向着-y轴的，也就是绕z轴旋转了90度的结果
    # 所以，要想让结果和pybullet中一致，需要将算出来的矩阵再绕z轴旋转-90度;或者说，在初始状态时，就把scara的朝向改为朝向x轴


# 加入范围限制
def clamp_joint_values(theta1: float, theta2: float, d3: float):
    """夹紧关节值到物理范围。返回 (theta1, theta2, d3_clamped)。"""
    t1 = min(max(theta1, P.theta1_min), P.theta1_max)
    t2 = min(max(theta2, P.theta2_min), P.theta2_max)
    d3c = min(max(d3, P.d3_min), P.d3_max)
    return t1, t2, d3c


# 逆运动学
def inverse_kinematics(x: float, y: float, z: float, elbow: str = "down"):
    # 由于坐标系问题,需要先做一下变换
    x, y = -y, x
    # 平面距离
    r2 = x * x + y * y
    r = np.sqrt(r2)

    # 可达性判定
    L1 = P.a1
    L2 = P.a2
    max_r = L1 + L2
    min_r = abs(L1 - L2)
    planar_ok = (min_r - 1e-9) <= r <= (max_r + 1e-9)

    # cos(theta2)
    cos_t2 = (r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    cos_t2 = max(min(cos_t2, 1.0), -1.0)
    base = np.arccos(cos_t2)
    theta2 = base if elbow == "up" else -base

    # theta1
    theta1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))

    # d3
    d3_raw = z - (P.d1 + P.d2)
    vertical_ok = (P.d3_min - 1e-9) <= d3_raw <= (P.d3_max + 1e-9)

    # clamp
    theta1_c, theta2_c, d3_c = clamp_joint_values(theta1, theta2, d3_raw)
    reachable = planar_ok and vertical_ok
    return theta1_c, theta2_c, d3_c, reachable


if __name__ == "__main__":
    # 简单测试：目标点来自正向再求逆，看误差
    tests = [
        (0.5, 0.5, 0.4),
        (0.3, -1.2, 0.25),
        (1.5, 0.0, 0.2),
        (0.2, -0.5, 0.35),
        (0.3, 0.4, 0.3),
    ]
    for (x, y, z) in tests:
        th1, th2, d3, ok = inverse_kinematics(x, y, z, elbow="down")
        T = forward_kinematics(th1, th2, d3)
        fk_pos = (T[0, 3], T[1, 3], T[2, 3])
        err = np.linalg.norm(np.array(fk_pos) - np.array([x, y, z]))
        print(
            f"Target=({x:.3f},{y:.3f},{z:.3f}) -> IK (t1={th1:.3f}, t2={th2:.3f}, d3={d3:.3f}, reachable={ok}) | FK pos={tuple(fk_pos)} | err={err:.3e}")