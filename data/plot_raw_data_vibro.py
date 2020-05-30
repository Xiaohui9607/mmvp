import numpy as np
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import glob
import matplotlib.pyplot as plt

crop_stategy = {
    'crush': [16, -5],
    'grasp': [0, -10],
    'lift_slow': [0, -3],
    'shake': [0, -1],
    'poke': [2, -5],
    'push': [2, -5],
    'tap': [0, -5],
    'low_drop': [0, -1],
    'hold': [0, -1],
}

behavior = 'low_drop'
n_frames_low_drop_can_coke = 22

path = "/Users/ramtin/PycharmProjects/data/CY101/rc_data/can_coke/trial_1/exec_1/low_drop" \
       "/vibro/b877428f-4080-11e0-bd19-000c2930dc90-1301423955849128165.tsv"

vibro_list = open(path).readlines()
vibro_list = [list(map(int, vibro.strip().split('\t'))) for vibro in vibro_list]
vibro_list = np.array(vibro_list)
vibro_time = vibro_list[:, 0]
vibro_data = vibro_list[:, 1:]

binsn_low_drop_can_coke = (np.array([1301424011268295424,
                                     1301424011361628160,
                                     1301424011454960896,
                                     1301424011548293632,
                                     1301424011641626368,
                                     1301424011734959104,
                                     1301424011828291840,
                                     1301424011921624576,
                                     1301424012014957312,
                                     1301424012108290048,
                                     1301424012201622784,
                                     1301424012294955520,
                                     1301424012388288256,
                                     1301424012481620992,
                                     1301424012574953728,
                                     1301424012668286464,
                                     1301424012761619200,
                                     1301424012854951936,
                                     1301424012948284672,
                                     1301424013041617408,
                                     1301424013134950144]), 1301424013228284928)

binsn_low_drop_can_coke, end_time = binsn_low_drop_can_coke
end_time -= binsn_low_drop_can_coke[0]
binsn_low_drop_can_coke -= binsn_low_drop_can_coke[0]

v_h_ratio = vibro_time[-1] / end_time
binsn_low_drop_can_coke = binsn_low_drop_can_coke * v_h_ratio

# groups = np.digitize(vibro_time, bins, right=False)
# vibro_data = [vibro_data[np.where(groups == idx)] for idx in range(1, n_frames + 1)]
# vibro_data = [np.vstack([np.resize(vibro[:, 0], (128,)),
#                          np.resize(vibro[:, 1], (128,)),
#                          np.resize(vibro[:, 2], (128,))]).T[np.newaxis, ...]
#               for vibro in vibro_data]
# # haplist = [np.pad(ht, [[0, 48 - ht.shape[0]], [0, 0]], mode='edge')[np.newaxis, ...] for ht in haplist]
# vibro_data = vibro_data[crop_stategy[behavior][0]:crop_stategy[behavior][1]]
# print(vibro_data)

x = vibro_data[2000:4000,0]
y = vibro_data[2000:4000,1]
z = vibro_data[2000:4000,2]
std = np.sqrt(x**2+y**2+z**2)
timestep = np.arange(len(x))

fig = plt.figure(figsize=(20.0, 8.0))
plt.plot(timestep, x, label="x-axis accelerometer")
plt.plot(timestep, y, label="y-axis accelerometer")
plt.plot(timestep, z, label="z-axis accelerometer")
plt.plot(timestep, std, label="magnitude deviation of accelerometer")

plt.ylabel(ylabel="Acceleration", fontsize=18)
plt.xlabel(xlabel="Time (ms)", fontsize=18)
plt.title("Raw 3-axis accelerometer readings", fontsize=18)
plt.legend(fontsize=12)
# plt.show()
plt.savefig('vibro_low_drop_can_coke.png', dpi=300, bbox_inches='tight')
plt.close()