import torch
import numpy as np

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 10)
plt.ion()

target_point = np.array([-3.0, 0])
anchor_point = np.array([0, 0])

is_running = True


def button_press_event(event):
    global target_point
    target_point = np.array([event.xdata, event.ydata])


def press(event):
    global is_running
    print('press', event.key)
    if event.key == 'escape':
        is_running = False  # quits app


fig, _ = plt.subplots()
fig.canvas.mpl_connect('button_press_event', button_press_event)
fig.canvas.mpl_connect('key_press_event', press)

length_joint = 2.0
theta_1 = np.deg2rad(-10)
theta_2 = np.deg2rad(-10)
theta_3 = np.deg2rad(-10)


def rotation(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s],
                  [s, c]])

    return R


def d_rotation(theta):
    dc = -np.sin(theta)
    ds = np.cos(theta)
    dR = np.array([[dc, -ds],
                   [ds, dc]])
    return dR



loss = 0
step = 0.01
coef = 0.1
while is_running:
    plt.clf()
    plt.title(
        f'loss: {round(loss, 4)}  theta_1: {round(np.rad2deg(theta_1))} theta_2: {round(np.rad2deg(theta_2))}  theta_3: {round(np.rad2deg(theta_3))}')

    R_1 = rotation(theta_1)
    dR1 = d_rotation(theta_1)

    R_2 = rotation(theta_2)
    dR2 = d_rotation(theta_2)

    R_3 = rotation(theta_3)
    dR3 = d_rotation(theta_3)

    joints = []

    segment = np.array([0.0, length_joint])

    joints.append(anchor_point)
    joint = R_1 @ segment
    joints.append(joint)

    joints.append(joint)
    joint = R_1 @ (segment + R_2 @ segment)
    joints.append(joint)

    joints.append(joint)
    joint = R_1 @ R_2 @ (R_3 @ segment + segment + segment)
    joints.append(joint)

    loss = np.sum((target_point - joint) ** 2)

    d_theta1 = np.sum(
        dR1 @ segment * -2 * (target_point - joint) - coef * (joint - joint[0]))
    theta_1 -= d_theta1 * step

    d_theta2 = np.sum(
        R_1 @ dR2 @ segment * -2 * (target_point - joint) - coef * (joint - joint[0]))
    theta_2 -= d_theta2 * step

    d_theta3 = np.sum(
        R_1 @ R_2 @ dR3 @ segment * -2 * (target_point - joint) - coef * (joint - joint[0]))
    theta_3 -= d_theta3 * step

    np_joints = np.array(joints)
    if len(np_joints):
        plt.plot(np_joints[:, 0], np_joints[:, 1])
    plt.scatter(target_point[0], target_point[1], s=50, c='r')

    plt.xlim(-10, 10)
    plt.ylim(0, 10)
    plt.draw()
    plt.pause(1e-3)
    # break
# input('end')
