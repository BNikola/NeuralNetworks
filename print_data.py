import numpy as np
import matplotlib.pyplot as plt


g = 9.81
angle           = 0.97982908
speed           = 0.66271114
basket_distance = 7.75
player_distance = 1.9


fig, ax = plt.subplots()
ax.set_xlim([0, 20])
plt.ion()

def plot_shot(ax, g, angle, speed2, basket_distance, player_distance):
    for i in range(10):
        p = 13.9 + np.random.uniform()
        speed = speed2 * p
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
        ax.bar(basket_distance, 3.05, color='lime')
        # plt.show()
        plt.draw()
        plt.pause(1.5)
        ax.cla()


plot_shot(ax, g, angle, speed, basket_distance, player_distance)
plt.show()
