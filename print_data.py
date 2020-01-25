import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


g = 9.81
angle           = 0.8010428892224671
speed           = 13.336489231902078
basket_distance = 7.75
player_distance = 1.9


def check_basket(x, angle, v):
    return (x * np.tan(angle)) - (9.81/2) * (x**2) / (v**2 * np.cos(angle))

fig, ax = plt.subplots()
ax.set_xlim([0, 20])
plt.ion()
def plot_shot(ax, g, angle, speed, basket_distance, player_distance):
    p = 13.9 + np.random.uniform()
    # speed = speed2 * p
    ax.set_title('Angle: ' + str(np.round(np.rad2deg(angle), 3)) + ' speed: ' + str(np.round(speed, 3)) + ' basket: ' + str(
        np.round(basket_distance, 3)) + ' player: ' + str(np.round(player_distance, 3)) + ' p: ' + str(p))
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
    ax.bar(basket_distance, 3.05, color='lime', width=0.4575)

    ax.scatter(x[95], y[95], s=24)
    # plt.show()
    plt.draw()
    plt.pause(1)
    ax.cla()

df = pd.read_csv('training_set2.csv')
for i in range(50):
    # plot_shot(ax, g, angle, speed, basket_distance, player_distance)
    plot_shot(ax, g, df['initial_angle'][i], df['initial_speed'][i], df['shot_distance'][i], df['player_distance'][i])
plt.show()
