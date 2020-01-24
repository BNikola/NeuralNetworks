import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('training_set.csv', index_col=0)
df = df.sample(frac=1).reset_index(drop=True)
print(df)

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
    ax.bar(basket_distance, 3.05, color='lime', width=0.4)
    plt.show()

class NeuralNetwork(object):
    def __init__(self):
        # parameters
        self.input_size = 2
        self.output_size = 2
        self.hidden_size = 8
        self.gama = 1.1
        self.max_iter = 50000
        self.max_error = 0.00001

        # weights - the network has only 3 layers: input, output and one hidden layer
        # initialize weights with random values
        # self.W1 = np.random.rand(self.input_size, self.hidden_size)
        # self.W2 = np.random.rand(self.hidden_size, self.output_size)

        # initialize weights from txt file (after training)
        self.W1 = np.loadtxt('W1_2_42.txt')
        self.W2 = np.loadtxt('W2_2_42.txt')

    def sigmoid(self, s, deriv=False):
        if deriv == True:
            return s * (1 - s)
        return 1 / (1 + np.exp(-s))

    def feed_forward(self, input):
        self.z = np.dot(input, self.W1)  # dot product of input and first set of weights
        self.z2 = self.sigmoid(self.z)  # activation function
        self.z3 = np.dot(self.z2, self.W2)  # dot product of hidden layer (z2) and second set of weights
        output = self.sigmoid(self.z3)
        return output

    def back_propagation(self, input, desired_output, output):
        # output layer
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
        for i in range(0, len(self.z2_delta)):
            for j in range(0, len(input)):
                self.W1_delta[i][j] = self.gama * self.z2_delta[i] * input[j]
        self.W1 += self.W1_delta.T

    def train(self, training_set):
        iter = 0
        E = 0
        E_old = 1
        cnt = 0
        while (iter < self.max_iter) or (E < self.max_error):
            # training_set = training_set.sample(frac=1).reset_index(drop=True)
            input = np.transpose(np.array([training_set['shot_distance'], training_set['player_distance']]))
            desired_output = np.transpose(np.array([training_set['initial_angle'], training_set['initial_speed']]))
            if (cnt > 100) and E != 0:
                if self.gama > 0.1:
                    self.gama = self.gama - 0.02
                    E_old = np.round(E, 2)
                    cnt = 0
            # elif cnt > 200:
            #     self.gama += 0.02
            E = 0
            for i, j in zip(input, desired_output):
                out = self.feed_forward(i)
                # temp error
                Ep = np.sum(np.square(j - out)) * 0.5
                E = E + Ep
                if Ep >= self.max_error:
                    self.back_propagation(i, j, out)
            # if cnt % 10 == 0:
            print(str(iter) + '\tGama: ' + str(np.round(self.gama,3)) + '\tGreska:\t' + str(np.round(E,15)) + '\t' + str(E_old) + '\t' + str(cnt))
            # if cnt > 3001:
            #     self.gama = 0.01
            iter += 1
            cnt += 1

        print(str(i) + '\t' + str(j))

        # save weights in file
        np.savetxt('W1_2_42.txt', self.W1)
        np.savetxt('W2_2_42.txt', self.W2)

NN = NeuralNetwork()

df_n = (df - df.min()) / (df.max() - df.min())
# df_n = ((df - df.min()) / (df.max() - df.min())) * (0.5 - 0.731059) + 0.5

NN.train(df_n)

inp = np.array([18, 1.86])

inp2 = np.array([(inp[0] - df['shot_distance'].min()) / (df['shot_distance'].max() - df['shot_distance'].min()),\
                 (inp[1] - df['player_distance'].min()) / (df['player_distance'].max() - df['player_distance'].min())])

out = NN.feed_forward(inp2)

g = 9.81
fig, ax = plt.subplots()
print(out[1] * (df['initial_speed'].max() - df['initial_speed'].min()) + df['initial_speed'].min())
print(str(inp) + '\t' + str(out))

plot_shot(ax, g,\
          out[0] * (df['initial_angle'].max() - df['initial_angle'].min()) + df['initial_angle'].min(),\
          out[1] * (df['initial_speed'].max() - df['initial_speed'].min()) + df['initial_speed'].min(),\
          inp[0], inp[1])
