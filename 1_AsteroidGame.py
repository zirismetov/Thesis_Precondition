import math
import numpy as np

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 10)
plt.ion()


def rotation_mat(degrees):
    theta = np.radians(degrees)
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s, 0],
                  [s, c, 0],
                  [0, 0, 1]])

    return R


def translation_mat(dx, dy):
    T = np.array([[1, 0, dx],
                  [0, 1, dy],
                  [0, 0, 1]])
    return T


def scale_mat(sx, sy):
    S = np.array([[sx, 0, 0],
                  [0, sy, 0],
                  [0, 0, 1]])

    return S


def dot(X, Y):
    # v = np.dot(X,Y)
    if Y.ndim >= 2:
        Z = np.zeros([X.shape[0],Y.shape[1]])
        for x in range(Z.shape[0]):
            for y in range(Z.shape[1]):
                result = 0
                for i in range(len(Y)):
                     result += X[x][i] * Y[i][y]
                Z[x][y] = result
    else:
        Z = np.zeros([X.shape[0],Y.ndim])
        for x in range(Z.shape[0]):
            for y in range(Z.shape[1]):
                result = 0
                for i in range(len(Y)):
                     result += X[x][i] * Y[i]
                Z[x][y] = result
        Z = np.squeeze(Z)
    # print(Z.shape, Z.ndim)
    # print(Z)
    return Z


def vec2d_to_vec3d(vec2):
    I = np.array([
        [1, 0],
        [0, 1],
        [0, 0]
    ])
    vec3 = dot(I, vec2) + np.array([0, 0 ,1])
    return vec3


def vec3d_to_vec2d(vec3):
    I = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ])
    vec2 = dot(I, vec3)
    return vec2


class Character():
    def __init__(self):
        super().__init__()
        self.__angle = 0
        self.geometry = []
        self.color = 'b'
        self.T = np.identity(3)
        self.C = np.identity(3)
        self.R = np.identity(3)
        self.S = np.identity(3)

        self.pos = np.array([0.0, 0.0])

        self.dir_init = np.array([0.0, 0.1])
        self.dir = np.array(self.dir_init)
        self.speed = 0.1
        self.generate_geometry()

    def set_angle(self, angle):
        self.__angle = angle  # encapsulation
        self.R = rotation_mat(self.__angle)

    def get_angle(self):
        return self.__angle

    def draw(self):
        x_values = []
        y_values = []
        ship_ratio = scale_mat(0.5, 1)

        for vec2d in self.geometry:
            vec3d = vec2d_to_vec3d(vec2d)

            vec3d = dot(self.C, vec3d)

            vec2d = vec3d_to_vec2d(vec3d)
            x_values.append(vec2d[0])
            y_values.append(vec2d[1])
        plt.plot(x_values, y_values, c=self.color)

        new_x = self.pos[0] + self.speed * np.cos(np.radians(self.get_angle() + 90))
        new_y = self.pos[1] + self.speed * np.sin(np.radians(self.get_angle() + 90))
        self.pos = np.array([new_x, new_y])

        self.T = translation_mat(self.pos[0], self.pos[1])
        self.C = dot(self.T, self.R)
        self.C = dot(self.C, ship_ratio)

class Asteroid(Character):
    def __init__(self):
        super().__init__()
        self.pos = np.array([np.random.randint(-8, 8), np.random.randint(-8, 8)])
        self.color = np.random.choice(['y', 'b', 'g'])
        self.speed = np.random.uniform(0.1, 0.5)
        self.rand_dir = np.radians(np.random.randint(0, 360))

    def generate_geometry(self):
        a = []
        pimult2 = 2 * math.pi
        r = np.random.uniform(0.5, 1.0)
        theta = 0
        while theta <= pimult2:
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            a.append([x, y])
            if theta < pimult2:
                ran = np.random.randint(5, 25)
                step = pimult2 / ran
                theta += step
                if theta > pimult2:
                    theta = pimult2
            elif theta == pimult2:
                break
        self.geometry = np.array(a)

    def draw(self):
        x_values = []
        y_values = []
        for vec2d in self.geometry:
            vec3d = vec2d_to_vec3d(vec2d)

            vec3d = dot(self.C, vec3d)

            vec2d = vec3d_to_vec2d(vec3d)
            x_values.append(vec2d[0])
            y_values.append(vec2d[1])

        plt.plot(x_values, y_values, c=self.color)

        new_x = self.pos[0] + self.speed * np.cos(self.rand_dir)
        new_y = self.pos[1] + self.speed * np.sin(self.rand_dir)
        if new_x > 10:
            new_x = 10
            self.rand_dir *= np.pi
            if np.cos(self.rand_dir) > 0:
                self.speed *= -1
        elif new_x < -10:
            new_x = -10
            self.rand_dir /= np.pi
            if np.cos(self.rand_dir) < 0:
                self.speed *= -1
        if new_y >= 10 or new_y <= -10:
            self.rand_dir = -self.rand_dir

        self.pos = np.array([new_x, new_y])

        self.T = translation_mat(self.pos[0], self.pos[1])
        self.C = dot(self.T, self.R)


class Player(Character):
    def __init__(self):
        super().__init__()

    def generate_geometry(self):
        self.geometry = np.array([
            [-1, 0],
            [1, 0],
            [0, 1],
            [-1, 0]
        ])


characters = [Player()]
for i in range(10):
    characters.append(Asteroid())
player = characters[0]
is_running = True


def press(event):
    global is_running, player
    print('press', event.key)
    if event.key == 'p':
        is_running = False  # quits app
    elif event.key == 'right':
        player.set_angle(player.get_angle() - 5)
    elif event.key == 'left':
        player.set_angle(player.get_angle() + 5)


fig, _ = plt.subplots()
fig.canvas.mpl_connect('key_press_event', press)

while is_running:
    plt.clf()
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    for character in characters:  # polymorhism
        character.draw()

    plt.draw()
    plt.pause(1e-2)
