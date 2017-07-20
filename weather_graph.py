import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import preprocess_data as pre_data

aqp_path = "datasets/AQP_temperatures/AQP_temperatures.csv"


df = pre_data.preprocess_aqp_dataset()

m =	np.array(df['date_time'].tail(200))
s = np.array(df['temperature'].tail(200))

print df.tail(50)

plt.plot(m, s)
plt.xlabel('time')
plt.ylabel('celcius')
#plt.ylabel('wind_speed')
#plt.ylabel('dew_point')
plt.title('Weather of Arequipa ')
plt.show()



