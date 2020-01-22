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

# generate 50 values for basket distance and player distance
shot_distance = np.linspace(sd_start, sd_end, num=500)
player_distance = np.linspace(pd_start, pd_end, num=500)

random.shuffle(shot_distance)
random.shuffle(player_distance)

initial_angle, initial_speed = perfect_shot(shot_distance, player_distance)

df = pd.DataFrame()
df['shot_distance'] = shot_distance
df['player_distance'] = player_distance
df['initial_angle'] = initial_angle
df['initial_speed'] = initial_speed

print(df)

pd.DataFrame.to_csv(df, "training_set", sep=',')