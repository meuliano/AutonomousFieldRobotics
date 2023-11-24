import bagpy
from bagpy import bagreader
import pandas as pd
import matplotlib.pyplot as plt

b = bagreader('/home/meuli/src/eece7150/HW5/2023-10-19-14-14-38-filtered.bag')

# get the list of topics
print(b.topic_table)

data_gps = b.message_by_topic('/gps/fix')
data_imu = b.message_by_topic('/imu/imu_uncompensated')
d_gps = pd.read_csv(data_gps)
d_imu = pd.read_csv(data_imu)


# Use “/gps/fix” and “/imu/imu_uncompensated”foryouralgorithm.

# get all the messages of type velocity
# velmsgs   = b.vel_data()
# veldf = pd.read_csv(velmsgs[0])
# plt.plot(d_gps['Time'], d_gps['linear.x'])

# # quickly plot velocities
# b.plot_vel(save_fig=True)

# # you can animate a timeseries data
# bagpy.animate_timeseries(veldf['Time'], veldf['linear.x'], title='Velocity Timeseries Plot')
