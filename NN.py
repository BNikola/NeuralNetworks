import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('training_set2.csv', index_col=0)
df = df.sample(frac=1).reset_index(drop=True)
print(df)

def plot_shot(ax, g, angle, speed, basket_distance, player_distance):
    # ax.set_xlim([0, 20])
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
        self.hidden_size = 10
        self.gama = 0.01
        self.max_iter = 50000
        self.max_error = 0.0001


        # weights - the network has only 3 layers: input, output and one hidden layer
        # initialize weights with random values
        # self.W1 = np.array([[3., 1., 1., 4.], [2.,4.,1.,-5.]])
        # self.W2 = np.array([[1.,3.,1.,-16.], [-2.,-3.,-1.,16.]]).T
        # self.W1 = np.random.rand(self.input_size, self.hidden_size)
        # self.W2 = np.random.rand(self.hidden_size, self.output_size)

        # initialize weights from txt file (after training)
        self.W1 = np.loadtxt('W1_2_94.txt')
        self.W2 = np.loadtxt('W2_2_94.txt')

    def sigmoid(self, s, deriv=False):
        if deriv == True:
            return s * (1 - s)
        return 1 / (1 + np.exp(-s))

    def tanh(self, x, deriv=False):
        if deriv == True:
            return 1 - x ** 2
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def feed_forward(self, input):
        self.z = np.dot(input, self.W1)  # dot product of input and first set of weights
        self.z2 = self.sigmoid(self.z)  # activation function
        self.z3 = np.dot(self.z2, self.W2)  # dot product of hidden layer (z2) and second set of weights - net of last layer
        output = self.sigmoid(self.z3)  # activation function

        return output

    def back_propagation(self, input, desired_output, output):
        # output layer
        self.output_error = desired_output - output * 20
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

    # def train(self, training_set):

    def train(self, input, desired_output):
        iter = 0
        E = 23
        E_old = 1
        cnt = 0
        while (iter < self.max_iter):
            if E < self.max_error and E != 23:
                break
            # training_set = training_set.sample(frac=1).reset_index(drop=True)
            # input = np.transpose(np.array([training_set['shot_distance'], training_set['player_distance']]))
            # desired_output = np.transpose(np.array([training_set['initial_speed'], training_set['initial_angle']]))
            # if (cnt > 100) and E != 0:
            #     if self.gama > 0.02:
            #         self.gama = self.gama - 0.01
            #         E_old = np.round(E, 2)
            #         cnt = 0
            # elif cnt > 200:
            #     self.gama += 0.02
            E = 0
            for i, j in zip(input, desired_output):
                out = self.feed_forward(i)
                # temp error
                Ep = np.sum(np.square(j - out*20)) * 0.5
                E = E + Ep
                if Ep >= self.max_error:
                    self.back_propagation(i, j, out)
            # if cnt % 10 == 0:
            #     print('----------------------\nW1')
            #     print(self.W1)
            #     print('----------------------\nW2')
            #     print(self.W2)
            #     print('----------------------\n1net')
            #     print(self.z)
            #     print('----------------------\n1y')
            #     print(self.z2)
            #     print('----------------------\n2net')
            #     print(self.z3)
            print(str(iter) + '\tGama: ' + str(np.round(self.gama,3)) + '\tGreska:\t' + str(np.round(E,15)) + '\t' + str(E_old) + '\t' + str(cnt))
            # if cnt > 3001:
            #     self.gama = 0.01
            iter += 1
            # cnt += 1

        # print(str(i) + '\t' + str(j))



        # save weights in file
        np.savetxt('W1_2_95.txt', self.W1)
        np.savetxt('W2_2_95.txt', self.W2)

    def throw(self, angle, speed):
        x1 = ( np.tan(angle) + np.sqrt( np.tan(angle)**2 - (2 * 9.81 * 0.55) / ( speed**2 * np.cos(angle)**2 ) ) ) / ( 9.81 / (speed**2 * np.cos(angle)**2 ) )
        x2 = ( np.tan(angle) - np.sqrt( np.tan(angle)**2 - (2 * 9.81 * 0.55) / ( speed**2 * np.cos(angle)**2 ) ) ) / ( 9.81 / (speed**2 * np.cos(angle)**2 ) )

        return np.max([x1, x2])

    def check_shot(self, start_range, end_range):
        return np.abs(start_range - end_range) < 0.1086

NN = NeuralNetwork()
# input = np.array([[1,1], [0.1, 0]])
# output = np.array([[1,0], [0, 1]])
# NN.train(input, output)
# print(NN.feed_forward([0.5, 0.5]))

#
# df_n = (df - df.min()) / (df.max() - df.min())
# # df_n = ((df - df.min()) / (df.max() - df.min())) * (0.5 - 0.731059) + 0.5
#


input = np.transpose(np.array([df['shot_distance'], df['player_distance']]))
desired_output = np.transpose(np.array([df['initial_angle'], df['initial_speed']]))
print(input.shape)
print(desired_output.shape)
print(input[0])
print(desired_output[0])
for i, j in zip(input[:10], desired_output[:10]):
    print(str(NN.throw(j[0], j[1])) + ' \t ' + str(i))

# NN.train(input, desired_output)


result = pd.DataFrame()


sd_start = 6.75
sd_end = 18
pd_start = 1.5
pd_end = 3
shot_distance = np.linspace(sd_start, sd_end, num=100)
player_distance = np.linspace(pd_start, pd_end, num=100)


np.random.shuffle(shot_distance)
np.random.shuffle(player_distance)

inp2 = np.array([shot_distance, player_distance]).T

print('---------')
counter = 0
for i in inp2:
    out2 = NN.feed_forward(i)
    shot = NN.throw(out2[0]*20, out2[1]*20)
    print(str(i[0]) + '\t' + str(shot) + '\t' + str(i[0] - 0.1086) + '\t' + str(i[0] + 0.1086) + '\t' + str(NN.check_shot(i[0], shot)))
    if NN.check_shot(i[0], shot) == True:
        counter += 1
print('Percent')
print(counter)
print('---------')

inp = np.array([16, 1.7])
out = NN.feed_forward(inp)
print(NN.throw(out[0]*20, out[1]*20))
print(out)
g = 9.81
fig, ax = plt.subplots()
plot_shot(ax, g, out[0]*20, out[1]*20, inp[0], inp[1])


# inp2 = np.array([(inp[0] - df['shot_distance'].min()) / (df['shot_distance'].max() - df['shot_distance'].min()),\
#                  (inp[1] - df['player_distance'].min()) / (df['player_distance'].max() - df['player_distance'].min())])
#
# out = NN.feed_forward(inp2)
#
# g = 9.81
# fig, ax = plt.subplots()
# print(out[1] * (df['initial_speed'].max() - df['initial_speed'].min()) + df['initial_speed'].min())
# print(str(inp) + '\t' + str(out * (df[['initial_angle', 'initial_speed']].max() - df[['initial_angle', 'initial_speed']].min()) + df[['initial_angle', 'initial_speed']].min()))
#
# plot_shot(ax, g,\
#           out[0] * (df['initial_angle'].max() - df['initial_angle'].min()) + df['initial_angle'].min(),\
#           out[1] * (df['initial_speed'].max() - df['initial_speed'].min()) + df['initial_speed'].min(),\
#           inp[0], inp[1])
