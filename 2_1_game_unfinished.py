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
    result = np.array([])
    try:
        # Z = np.dot(X, Y)
        # return Z
        for i in range(len(X)):
            for j in range(len(Y[0])):
                result[i][j] = getCell(matrixA, matrixB, i, j)
            print(result[i])
    except NameError:
        print("errpr")

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
        self.C = np.identity(3)
        self.R = np.identity(3)
        self.S = np.identity(3)


        self.pos = np.array([0.0, 0.0])

        self.dir_init = np.array([0.0, 0.1])
        self.dir = np.array(self.dir_init)

        self.speed = 0.1

        self.generate_geometry()

    def set_angle(self, angle):
        self.__angle = angle # encapsulation

        self.R = rotation_mat(self.__angle)
        # self.
        self.pos = np.add(self.pos, self.dir)
        self.T = translation_mat(self.pos[0], self.pos[1])
        self.C = dot(self.T, self.R)

    def get_angle(self):
        return self.__angle


    def draw(self):
        x_values = []
        y_values = []

        for vec2d in self.geometry:

            vec3d = vec2d_to_vec3d(vec2d)

            # self.R = rotation_mat(5)

            vec3d = dot(self.C, vec3d)
            vec2d = vec3d_to_vec2d(vec3d)
            x_values.append(vec2d[0])
            y_values.append(vec2d[1])

            # x += 0.2

        plt.plot(x_values, y_values, c=self.color)



    # def move(self):
    #     self.T = translation_mat()


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
player = characters[0]

is_running = True
def press(event):
    global is_running, player
    print('press', event.key)
    if event.key == 'p':
        is_running = False # quits app
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

    for character in characters: # polymorhism
        character.draw()

    plt.draw()
    plt.pause(1e-2)