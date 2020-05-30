import numpy as np
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import glob
import matplotlib.pyplot as plt

HAPTIC_MAX = [-10.478915, 1.017272, 6.426756, 5.950242, 0.75426, -0.013009, 0.034224]
HAPTIC_MIN = [-39.090578, -21.720063, -10.159031, -4.562487, -1.456323, -1.893409, -0.080752]
# we do not normalize the cpos
HAPTIC_MEAN = [-25.03760727, -8.2802204, -5.49065186, 2.53891808, -0.6424120, -1.22525292, -0.04463354, 0.0, 0.0, 0.0]
HAPTIC_STD = [4.01142790e+01, 2.29780167e+01, 2.63156072e+01, 7.54091499e+00, 3.40810983e-01, 3.23891355e-01, 1.65208189e-03, 1.0, 1.0, 1.0]

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

behavior='low_drop'
n_frames_low_drop_can_coke=22

path1= '/Users/ramtin/PycharmProjects/data/CY101/rc_data/can_coke/trial_1/exec_1/low_drop/proprioception/ttrq0.txt'
path2='/Users/ramtin/PycharmProjects/data/CY101/rc_data/can_coke/trial_1/exec_1/low_drop/proprioception/cpos0.txt'

haplist1 = open(path1, 'r').readlines()
haplist2 = open(path2, 'r').readlines()
haplist = [list(map(float, v.strip().split('\t'))) + list(map(float, w.strip().split('\t')))[1:] for v, w in
           zip(haplist1, haplist2)]
haplist = np.array(haplist)
time_duration = (haplist[-1][0] - haplist[0][0]) / n_frames_low_drop_can_coke
bins = np.arange(haplist[0][0], haplist[-1][0], time_duration)

end_time = haplist[-1][0]
groups = np.digitize(haplist[:, 0], bins, right=False)

# print("%d," %end_time)
# for bin in bins:
#     print("%d," %bin)

haplist = [haplist[np.where(groups == idx)][..., 1:][:48] for idx in range(1, n_frames_low_drop_can_coke + 1)]
haplist = [np.pad(ht, [[0, 48 - ht.shape[0]], [0, 0]], mode='edge')[np.newaxis, ...] for ht in haplist]
haplist = haplist[crop_stategy[behavior][0]:crop_stategy[behavior][1]]

# print(haplist[0][0]) # one frame

haptic_across_all_frames = []
for i in range(0, 21):
    haptic_across_all_frames.append(np.mean(haplist[i][0], axis=0)) #extend

haptic_across_all_frames = np.array(haptic_across_all_frames)

#for normalizing
for index, a_frame in enumerate(haptic_across_all_frames):
    a_frame[:-3] = (a_frame[:-3]-np.array(HAPTIC_MIN))/(np.array(HAPTIC_MAX)-np.array(HAPTIC_MIN))
    haptic_across_all_frames[index] = a_frame
#for normalizing

# haptic_data = np.array(haptic_across_all_frames).T.reshape(10,21) # extend
haptic_data = haptic_across_all_frames.T

fig = plt.figure(figsize=(20.0, 8.0))
plt.imshow(haptic_data) #for gray_scale: cmap=plt.cm.gray

ax = plt.gca()
ax.set_xticks(np.arange(0, 21+1, 2))
ax.set_yticks(np.arange(0, 10+1, 1))
ax.set_xticklabels(np.arange(1, 21+1 , 2))
ax.set_yticklabels(np.arange(1, 10+1, 1))

plt.ylabel(ylabel="Joints [1-7], End Effector [8-10]", fontsize=18)
plt.xlabel(xlabel="Frames", fontsize=18)
plt.title("Visualization of Haptic Features", fontsize=18)

plt.colorbar()
# plt.show()
plt.savefig("haptic_low_drop_can_coke.png", dpi=300, bbox_inches='tight')
plt.close()