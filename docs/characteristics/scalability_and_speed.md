Scalability
===========

``` {.python}
from sklearn import cluster, datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os.path

import warnings
warnings.filterwarnings('ignore')
```

``` {.python}
# install DenMune clustering algorithm using pip command from the offecial Python repository, PyPi
# from https://pypi.org/project/denmune/
!pip install denmune

# then import it
from denmune import DenMune
```

``` {.python}
# Denmune's Paramaters
knn = 25 # k-nearest neighbor, the only parameter required by the algorithm
data_scale = []

for n in range(1000, 100000, 1000):
    n_samples = n
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)

    data= noisy_circles[0]
    data_labels = noisy_circles[1]
    dm = DenMune(train_data=data, k_nearest=knn, rgn_tsne=True)
    labels, validity = dm.fit_predict(show_noise=True, show_analyzer=False, show_plots=False)
    time_exec = dm.analyzer['exec_time']['DenMune']
    data_scale.append([n, time_exec ])

    print('data size:',n ,  'time:' , round(time_exec,4), 'seconds')
```

::: {.parsed-literal}
data size: 1000 time: 0.4518 seconds data size: 2000 time: 0.93 seconds
data size: 3000 time: 1.3754 seconds data size: 4000 time: 2.0891
seconds data size: 5000 time: 2.8772 seconds data size: 6000 time:
4.5046 seconds data size: 7000 time: 5.7184 seconds data size: 8000
time: 4.836 seconds data size: 9000 time: 7.793 seconds data size: 10000
time: 8.3138 seconds data size: 11000 time: 8.7401 seconds data size:
12000 time: 9.8531 seconds data size: 13000 time: 11.2796 seconds data
size: 14000 time: 13.4036 seconds data size: 15000 time: 16.6113 seconds
data size: 16000 time: 14.4252 seconds data size: 17000 time: 20.697
seconds data size: 18000 time: 18.1152 seconds data size: 19000 time:
22.1096 seconds data size: 20000 time: 25.8013 seconds data size: 21000
time: 26.6907 seconds data size: 22000 time: 27.0235 seconds data size:
23000 time: 27.3918 seconds data size: 24000 time: 38.0108 seconds data
size: 25000 time: 41.3266 seconds data size: 26000 time: 36.7593 seconds
data size: 27000 time: 42.6916 seconds data size: 28000 time: 41.0344
seconds data size: 29000 time: 42.878 seconds data size: 30000 time:
50.9385 seconds data size: 31000 time: 51.326 seconds data size: 32000
time: 54.6266 seconds data size: 33000 time: 50.0233 seconds data size:
34000 time: 59.8251 seconds data size: 35000 time: 51.2865 seconds data
size: 36000 time: 62.331 seconds data size: 37000 time: 59.3316 seconds
data size: 38000 time: 67.6423 seconds data size: 39000 time: 72.0803
seconds data size: 40000 time: 70.2149 seconds data size: 41000 time:
71.7297 seconds data size: 42000 time: 74.2931 seconds data size: 43000
time: 76.8439 seconds data size: 44000 time: 93.3641 seconds data size:
45000 time: 93.5944 seconds data size: 46000 time: 74.4506 seconds data
size: 47000 time: 94.0584 seconds data size: 48000 time: 105.9512
seconds data size: 49000 time: 94.7943 seconds data size: 50000 time:
88.705 seconds data size: 51000 time: 110.4996 seconds data size: 52000
time: 119.498 seconds data size: 53000 time: 125.8244 seconds data size:
54000 time: 117.446 seconds data size: 55000 time: 129.3928 seconds data
size: 56000 time: 135.6952 seconds data size: 57000 time: 137.2575
seconds data size: 58000 time: 146.3149 seconds data size: 59000 time:
131.5086 seconds data size: 60000 time: 160.2656 seconds data size:
61000 time: 160.4223 seconds data size: 62000 time: 149.5028 seconds
data size: 63000 time: 153.2287 seconds data size: 64000 time: 163.9789
seconds data size: 65000 time: 174.5982 seconds data size: 66000 time:
190.9555 seconds data size: 67000 time: 185.289 seconds data size: 68000
time: 238.9069 seconds data size: 69000 time: 182.9564 seconds data
size: 70000 time: 209.4002 seconds data size: 71000 time: 243.5443
seconds data size: 72000 time: 208.053 seconds data size: 73000 time:
220.6103 seconds data size: 74000 time: 217.0223 seconds data size:
75000 time: 223.4226 seconds data size: 76000 time: 233.5772 seconds
data size: 77000 time: 238.5042 seconds data size: 78000 time: 245.0213
seconds data size: 79000 time: 221.3705 seconds data size: 80000 time:
244.24 seconds data size: 81000 time: 274.9946 seconds data size: 82000
time: 253.6804 seconds data size: 83000 time: 286.6225 seconds data
size: 84000 time: 266.2466 seconds data size: 85000 time: 317.2572
seconds data size: 86000 time: 325.8073 seconds data size: 87000 time:
303.0139 seconds data size: 88000 time: 338.6655 seconds data size:
89000 time: 343.8219 seconds data size: 90000 time: 347.6194 seconds
data size: 91000 time: 325.1733 seconds data size: 92000 time: 359.6347
seconds data size: 93000 time: 350.8712 seconds data size: 94000 time:
379.433 seconds data size: 95000 time: 334.2113 seconds data size: 96000
time: 338.2628 seconds data size: 97000 time: 371.5167 seconds data
size: 98000 time: 347.7595 seconds data size: 99000 time: 391.2152
seconds
:::

``` {.python}
# compouting moving average to smooth the curve
x, y = zip(*data_scale)
window = 5
cumsum, moving_aves = [0], []

for i, n in enumerate(y, 1):
    cumsum.append(cumsum[i-1] + n)
    if i>=window:
        moving_ave = (cumsum[i] - cumsum[i-window])/window
        #can do stuff with moving_ave here
        moving_aves.append(moving_ave)
y = moving_aves        
```

``` {.python}
# Creating figure and axis objects using subplots()
fig, ax = plt.subplots(figsize=[20, 8])
ax.plot(x[:-window+1], y, marker='.', linewidth=2, label='DenMune Scalability')
plt.xticks(rotation=60)
ax.set_xlabel('Dataset size')
ax.set_ylabel('Time in seconds')
plt.legend()
plt.show()
```

![image](images/scalability/output_5_0.png)
