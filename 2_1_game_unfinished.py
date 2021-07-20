import numpy as np

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)
plt.ion()


def rotation_mat(degrees):
    # R = np.identity(3)
    theta = np.radians(degrees)
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c , -s, 0],
                 [s , c , 0],
                 [0 , 0 , 1]])
    #TODO
    return R

def translation_mat(dx, dy):
    T = np.array([[1, 0, dx],
                  [0, 1, dy],
                  [0, 0, 1]])
    #TODO
    return T

def scale_mat(sx, sy):
    S = np.identity(3)
    #TODO
    return S

def dot(X, Y):
    try:
        Z = np.dot(X, Y)
        # Z = sum([X[i][0] * Y[i] for i in range(len(Y[i]))])

        return Z
    except NameError:
        print("errpr")

def vec2d_to_vec3d(vec2):
    I = np.array([
        [1, 0],
        [0, 1],
        [0, 0]
    ])
    # 0 0 1 top down view ?
    vec3 = dot(I, vec2) + np.array([0, 0 ,1])
    return vec3


def vec3d_to_vec2d(vec3):
    I = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ])
    vec2 = dot(I, vec3)
    return vec2

A = np.array([
    [1,2,3,4],
    [1,2,3,4]
])
B = np.array([
    [1,2,3],
    [1,2,3],
    [1,2,3],
    [1,2,3]
])

C = dot(A, B)

vec2 = np.array([1.0, 0])
vec3 = vec2d_to_vec3d(vec2)
vec2 = vec3d_to_vec2d(vec3)
# exit()

class Character(object):
    def __init__(self):
        super().__init__()
        self.__angle = 3 * np.pi

        self.geometry = []
        self.color = 'r'

        self.C = np.identity(3)
        self.R = np.identity(3)
        self.S = np.identity(3)
        self.T = np.identity(3)

        self.pos = np.array([5.0, 0.0])
        self.dir_init = np.array([0.0, 1.0])
        self.dir = np.array(self.dir_init)
        self.speed = 0.1

        self.generate_geometry()
        self.T = translation_mat(0.0, 0.0)
    def set_angle(self, angle):
        self.__angle = angle # encapsulation
        self.R = rotation_mat(self.__angle)


    def get_angle(self):
        return self.__angle

    def set_T(self, T):
        self.T = translation_mat(0.0, T)

    def get_T(self):
        return self.T

    def draw(self):
        x_values = []
        y_values = []
        for vec2d in self.geometry:
            vec3d = vec2d_to_vec3d(vec2d)

            self.C = dot(self.T, self.R)
            vec3d = dot(self.C, vec3d)
            vec2d = vec3d_to_vec2d(vec3d)
            x_values.append(vec2d[0])
            y_values.append(vec2d[1])

        plt.plot(x_values, y_values, c=self.color)


class Asteroid(Character):
    def __init__(self):
        super().__init__()

    def generate_geometry(self):
        self.geometry = []


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


characters = []
characters.append(Player())
# ???? -1 why?
player = characters[0]

is_running = True
def press(event):
    global is_running, player
    print('press', event.key)
    if event.key == 'escape':
        is_running = False # quits app
    elif event.key == 'right':
        player.set_angle(player.get_angle() - 5)
    elif event.key == 'left':
        player.set_angle(player.get_angle() + 5)
    # elif event.key == 'up':
    #     player.set_T(player.get_T() + 5)
    # elif event.key == 'down':
    #     player.set_angle(player.get_angle() + 5)

fig, _ = plt.subplots()
fig.canvas.mpl_connect('key_press_event', press)

while is_running:
    plt.clf()

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    for character in characters: # polymorhism
        character.draw()

    plt.draw()
    plt.pause(1e-2)