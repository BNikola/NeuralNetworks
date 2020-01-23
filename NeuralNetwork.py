import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('training_set.csv', index_col=0)
print(df)
print(df.sample(frac=1).reset_index(drop=True))

def plot_shot(ax, g, angle, speed, basket_distance, player_distance):
    ax.set_title('Angle: ' + str(np.round(np.rad2deg(angle), 3)) + ' speed: ' + str(np.round(speed, 3)) + ' basket: ' + str(
        np.round(basket_distance, 3)) + ' player: ' + str(np.round(player_distance, 3)))
    ax.set_xlabel('Distance')
    ax.set_ylabel('Height')

    tmax = ((2 * speed) * np.sin(angle)) / g
    time_mat = tmax * np.linspace(0, 1, 100)[:, None]

    x = ((speed * time_mat) * np.cos(angle))
    y = ((speed * time_mat) * np.sin(angle)) - ((0.5 * g) * (time_mat ** 2)) + 2.5

    ax.plot(x, y)
    ax.bar(player_distance, 2.9, color='orange')
    # ax.bar(player_distance, height=)
    plt.bar(player_distance, height=0.2, bottom=2.9, color='red')
    ax.bar(basket_distance, 3.05, color='lime')
    plt.show()

class NeuralNetwork(object):
    def __init__(self):
        # parameters
        self.input_size = 2
        self.output_size = 2
        self.hidden_size = 6 #(int)(500 / (5 * 4))
        self.gama = 0.01
        self.max_iter = 100000
        self.max_error = 0.001

        # weights
        # self.W1 = np.array([[3., 1., 1., 4.], [2.,4.,1.,-5.]])
        # self.W2 = np.array([[1.,3.,1.,-16.], [-2.,-3.,-1.,16.]]).T
        self.W1 = np.random.rand(self.input_size, self.hidden_size)
        self.W2 = np.random.rand(self.hidden_size, self.output_size)

        # self.W1 = np.loadtxt('W1_62.txt')
        # self.W2 = np.loadtxt('W2_62.txt')

    def sigmoid(self, s, deriv=False):
        if deriv == True:
            return s * (1 - s)
        return 1 / (1 + np.exp(-s))

    def feed_forward(self, input):
        self.z = np.dot(input, self.W1)  # dot product of X (input) and first set of weights (3x2)
        self.z2 = self.sigmoid(self.z)  # activation function
        self.z3 = np.dot(self.z2, self.W2)  # dot product of hidden layer (z2) and second set of weights (3x1)
        output = self.sigmoid(self.z3)
        return output

    def back_propagation(self, input, desired_output, output):
        # output layer
        # todo - maybe change this error
        self.output_error = desired_output - output
        self.output_delta = self.output_error * self.sigmoid(output, deriv=True)

        # W2 delta
        self.W2_delta = np.zeros([len(self.output_delta), len(self.z2)])
        for i in range(0, len(self.output_delta)):
            for j in range(0, len(self.z2)):
                self.W2_delta[i][j] = self.gama * self.output_delta[i] * self.z2[j]
        self.W2 += self.W2_delta.T

        # hidden layer
        self.z2_error = self.output_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoid(self.z2, deriv=True)

        self.W1_delta = np.zeros([len(self.z2_delta), len(input)])
        for i in range (0, len(self.z2_delta)):
            for j in range (0, len(input)):
                self.W1_delta[i][j] = self.gama * self.z2_delta[i] * input[j]
        self.W1 += self.W1_delta.T

    def train(self, input, desired_output):
        iter = 0
        E = 0
        while (iter < self.max_iter) or (E < self.max_error):
            E = 0
            for i, j in zip(input, desired_output):
                out = self.feed_forward(i)
                Ep = np.sum(np.square(j - out)) * 0.5
                E = E + Ep
                # print(i)
                if Ep >= self.max_error:
                    self.back_propagation(i, j, out)
            print(str(iter) + 'Greska: ' + str(E))
            iter += 1
        print('Gotovo')
        print(str(i) + '\t' + str(j))
        np.savetxt('W1_100.txt', self.W1)
        np.savetxt('W2_100.txt', self.W2)
        # print(out)
        print('--------')

# region TEST
print('------------')
NN = NeuralNetwork()

# desired_output = np.transpose(np.array([df['initial_angle'], df['initial_speed']]))
# forward = 0
# g = 9.81
# fig, ax = plt.subplots()
# for i, j in zip(input[:20], desired_output[:20]):
#     print('$--------------')
#     print(i)
#     print(j)
#     print('$--------------')
#     forward = NN.feed_forward(i)
#     print(forward)
#     NN.back_propagation(i, j, forward)
#     plot_shot(ax, g, forward[0], forward[1], i[0], i[1])
#     ax.cla()
# print('----------')
# print(forward)
def sigmoid2(s, deriv=False):
    if deriv == True:
        return s * (1 - s)
    return 1 / (1 + np.exp(-s))
# def sig_r(s):
#     return np.log(s / (1 - s))
# SV = np.vectorize(sigmoid2)
# SRV = np.vectorize(sig_r)
# endregion
for i in range(1):
    df.sample(frac=1).reset_index(drop=True)
    df_n = (df - df.min()) / (df.max() - df.min())
    input = np.transpose(np.array([df_n['shot_distance'], df_n['player_distance']]))
    desired_output = np.transpose(np.array([df_n['initial_angle'], df_n['initial_speed']]))
    # input = np.transpose(np.array([df['shot_distance'] / np.max(df['shot_distance']), df['player_distance'] / np.max(df['player_distance'])]))
    # desired_output = np.transpose(np.array([df['initial_angle'] / np.max(df['initial_angle']), df['initial_speed'] / np.max(df['initial_speed'])]))

    # NN.train(input, desired_output)

inp = np.array([6.75, 1.9])
inp2 = np.array([(inp[0] - df['shot_distance'].min()) / (df['shot_distance'].max() - df['shot_distance'].min()),\
                 (inp[1] - df['player_distance'].min()) / (df['player_distance'].max() - df['player_distance'].min())])

out = NN.feed_forward(inp2)
g = 9.81
fig, ax = plt.subplots()
print(out[1])
print(out[0])
# print(sig_r(out[1]))
print(13.2 + out[1])
print(out[1] * 13.2 + out[1])
print(out[1] * (df['initial_speed'].max() - df['initial_speed'].min()) + df['initial_speed'].min())

# print(NN.W1)
# print(NN.W2)

plot_shot(ax, g,\
          out[0] * (df['initial_angle'].max() - df['initial_angle'].min()) + df['initial_angle'].min(),\
          out[1] * (df['initial_speed'].max() - df['initial_speed'].min()) + df['initial_speed'].min(), inp[0], inp[1])
print(str(inp) + '\t' + str(out))


# scale input
# input = np.array([[1,1], [0.1, 0]])
# output = np.array([[1,0], [0, 1]])
# NN.train(input, output)
# print(NN.feed_forward([0.5, 0.5]))


#     [df['shot_distance'] / np.max(df['shot_distance']), df['player_distance'] / np.max(df['player_distance'])])
# output = np.array(
#     [df['initial_angle'] / np.max(df['initial_angle']), df['initial_speed'] / np.max(df['initial_speed'])])
# NN.train(input, output)

# region Presentation test
# input = [1,1]
# output = [1, 0]
# NN = NeuralNetwork()
# ff = NN.feed_forward(input)
# bp = NN.back_propagation(input, output, ff)

# print('Drugi put\n-------------------------')
# input = [0.1,0]
# output = [0,1]
# ff = NN.feed_forward(input)
# bp = NN.back_propagation(input, output, ff)
# print(ff)
#
#
#
# print('Treci put\n-------------------------')
# input = [0.5, 0.5]
# # output = [0,1]
# ff = NN.feed_forward(input)
# # bp = NN.back_propagation(input, output, ff)
# print(np.round(ff, 4))
# Endregion

# input = np.array([df['shot_distance'], df['player_distance']])
# output = np.array([df['initial_angle'], df['initial_speed']])


# input2 = np.array([df['shot_distance'], df['player_distance']])
# print([df['shot_distance']/np.max(df['shot_distance']), df['player_distance']/np.max(df['player_distance'])])
# for i, j in zip(input.T, input2.T):
#     print(str(i[0]) + ' ' + str(i[1]))
#     print(str(j[0]) + ' ' + str(j[1]))
#     print(str(i[0] * np.max(df['shot_distance'])) + ' ' + str(i[1] * np.max(df['player_distance'])))
#     print('---------')






# for k in range(50):
#     df.sample(frac=1).reset_index(drop=True)
#     # scale input
#     input = np.array(
#         [df['shot_distance'] / np.max(df['shot_distance']), df['player_distance'] / np.max(df['player_distance'])])
#     output = np.array(
#         [df['initial_angle'] / np.max(df['initial_angle']), df['initial_speed'] / np.max(df['initial_speed'])])
#     for i,j in zip(input.T, output.T):
#         print('Greska: ' + str(np.sum(np.square(j - NN.feed_forward(i))) * 0.5))
#         NN.train(i, j)


# fig, ax = plt.subplots()
# ax.set_xlim([0, 20])
# g = 9.81
# # print('----------')
# # print(i)
# # print(j)
# # # plot_shot(ax, g, angle, speed, basket_distance, player_distance)
# # # plot_shot(ax, g, 0.81185048, 10.3647713, 10.38636364, 2.09090909)
# inp = np.array([12, 1.5])
# out = NN.feed_forward(inp)
# print('------')
# print(out[0]* np.max(df['initial_angle']), out[1]* np.max(df['initial_speed']))
# plot_shot(ax, g, out[0]* np.max(df['initial_angle']), out[1]* np.max(df['initial_speed']), inp[0], inp[1])

# out = NN.feed_forward([0.5, 0.5])
# print(out)
# print(NN.W1)
# print(NN.W2)
