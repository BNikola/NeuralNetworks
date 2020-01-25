import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

h = 2.5     # height of the player
H = 3.05    # height of the basket
dh = H - h  # difference in height
g = 9.81    # gravitation


# distances of shot (sd) and player (pd)
sd_start = 6.75
sd_end = 18
pd_start = 1.5
pd_end = 3

# returns perfect initial angle and speed
def perfect_shot(basket_distance, player_distance):
    initial_angle = np.arctan((dh + np.sqrt(dh**2 + basket_distance**2)) / basket_distance)
    initial_speed = np.sqrt(g * (dh + np.sqrt(dh**2 + basket_distance**2)))
    return initial_angle, initial_speed

def perfect_shot2(distance, defender):
    angle = (0.38 - distance / 135) * np.pi
    velocity = distance * np.sqrt(g / (distance * np.tan(angle) - H + h) / 2) / np.cos(angle)
    return angle, velocity

# generate 50 values for basket distance and player distance
shot_distance = np.linspace(sd_start, sd_end, num=200)
player_distance = np.linspace(pd_start, pd_end, num=200)
# b_s = (18 - 6.75) / 1000
# p_s = 1.5 / 1000
#
# shot_distance = np.array([6.75 + b_s * i for i in range(0, 1001)])
# player_distance = np.array([1.5 + p_s * i for i in range(0, 1001)])

random.shuffle(shot_distance)
random.shuffle(player_distance)

initial_angle, initial_speed = perfect_shot(shot_distance, player_distance)

df = pd.DataFrame()
df['shot_distance'] = shot_distance
df['player_distance'] = player_distance
df['initial_angle'] = initial_angle
df['initial_speed'] = initial_speed

print(df)

pd.DataFrame.to_csv(df, "training_set2.csv", sep=',')