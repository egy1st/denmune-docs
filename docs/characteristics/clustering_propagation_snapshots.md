Clustering Propagation Snapshots
================================

``` {.python}
import pandas as pd
import time
import os.path
import glob

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
# clone datasets from our repository datasets
if not os.path.exists('datasets'):
  !git clone https://github.com/egy1st/datasets
```

::: {.parsed-literal}
Cloning into \'datasets\'\... remote: Enumerating objects: 52, done.\[K
remote: Counting objects: 100% (52/52), done.\[K remote: Compressing
objects: 100% (43/43), done.\[K remote: Total 52 (delta 8), reused 49
(delta 8), pack-reused 0\[K Unpacking objects: 100% (52/52), done.
:::

``` {.python}
#@title  { run: "auto", vertical-output: true, form-width: "50%" }
dataset = "t7.10k" #@param ["t4.8k", "t5.8k", "t7.10k", "t8.8k"]
show_noize_checkbox = True #@param {type:"boolean"}
data_path = 'datasets/denmune/chameleon/' 

# train file
data_file = data_path + dataset +'.csv'
X_train = pd.read_csv(data_file, sep=',', header=None)
```

``` {.python}
from itertools import chain

# Denmune's Paramaters
knn = 39 # number of k-nearest neighbor, the only parameter required by the algorithm

# create list of differnt snapshots of the propagation
snapshots = chain([0], range(2,5), range(5,50,5), range(50, 100, 10), range(100,500,50), range(500,1000, 100), range(1000,3000, 250),range(3000,5500,500))

from IPython.display import clear_output
for snapshot in snapshots:
    print ("itration", snapshot )
    #clear_output(wait=True)
    dm = DenMune(train_data=X_train, k_nearest=knn, rgn_tsne=False, prop_step=snapshot)
    labels, validity = dm.fit_predict(show_analyzer=False, show_noise=False)    
```

::: {.parsed-literal}
itration 0
:::

![image](images/prop_snapshots/output_5_1.png)

::: {.parsed-literal}
itration 2
:::

![image](images/prop_snapshots/output_5_3.png)

::: {.parsed-literal}
itration 3
:::

![image](images/prop_snapshots/output_5_5.png)

::: {.parsed-literal}
itration 4
:::

![image](images/prop_snapshots/output_5_7.png)

::: {.parsed-literal}
itration 5
:::

![image](images/prop_snapshots/output_5_9.png)

::: {.parsed-literal}
itration 10
:::

![image](images/prop_snapshots/output_5_11.png)

::: {.parsed-literal}
itration 15
:::

![image](images/prop_snapshots/output_5_13.png)

::: {.parsed-literal}
itration 20
:::

![image](images/prop_snapshots/output_5_15.png)

::: {.parsed-literal}
itration 25
:::

![image](images/prop_snapshots/output_5_17.png)

::: {.parsed-literal}
itration 30
:::

![image](images/prop_snapshots/output_5_19.png)

::: {.parsed-literal}
itration 35
:::

![image](images/prop_snapshots/output_5_21.png)

::: {.parsed-literal}
itration 40
:::

![image](images/prop_snapshots/output_5_23.png)

::: {.parsed-literal}
itration 45
:::

![image](images/prop_snapshots/output_5_25.png)

::: {.parsed-literal}
itration 50
:::

![image](images/prop_snapshots/output_5_27.png)

::: {.parsed-literal}
itration 60
:::

![image](images/prop_snapshots/output_5_29.png)

::: {.parsed-literal}
itration 70
:::

![image](images/prop_snapshots/output_5_31.png)

::: {.parsed-literal}
itration 80
:::

![image](images/prop_snapshots/output_5_33.png)

::: {.parsed-literal}
itration 90
:::

![image](images/prop_snapshots/output_5_35.png)

::: {.parsed-literal}
itration 100
:::

![image](images/prop_snapshots/output_5_37.png)

::: {.parsed-literal}
itration 150
:::

![image](images/prop_snapshots/output_5_39.png)

::: {.parsed-literal}
itration 200
:::

![image](images/prop_snapshots/output_5_41.png)

::: {.parsed-literal}
itration 250
:::

![image](images/prop_snapshots/output_5_43.png)

::: {.parsed-literal}
itration 300
:::

![image](images/prop_snapshots/output_5_45.png)

::: {.parsed-literal}
itration 350
:::

![image](images/prop_snapshots/output_5_47.png)

::: {.parsed-literal}
itration 400
:::

![image](images/prop_snapshots/output_5_49.png)

::: {.parsed-literal}
itration 450
:::

![image](images/prop_snapshots/output_5_51.png)

::: {.parsed-literal}
itration 500
:::

![image](images/prop_snapshots/output_5_53.png)

::: {.parsed-literal}
itration 600
:::

![image](images/prop_snapshots/output_5_55.png)

::: {.parsed-literal}
itration 700
:::

![image](images/prop_snapshots/output_5_57.png)

::: {.parsed-literal}
itration 800
:::

![image](images/prop_snapshots/output_5_59.png)

::: {.parsed-literal}
itration 900
:::

![image](images/prop_snapshots/output_5_61.png)

::: {.parsed-literal}
itration 1000
:::

![image](images/prop_snapshots/output_5_63.png)

::: {.parsed-literal}
itration 1250
:::

![image](images/prop_snapshots/output_5_65.png)

::: {.parsed-literal}
itration 1500
:::

![image](images/prop_snapshots/output_5_67.png)

::: {.parsed-literal}
itration 1750
:::

![image](images/prop_snapshots/output_5_69.png)

::: {.parsed-literal}
itration 2000
:::

![image](images/prop_snapshots/output_5_71.png)

::: {.parsed-literal}
itration 2250
:::

![image](images/prop_snapshots/output_5_73.png)

::: {.parsed-literal}
itration 2500
:::

![image](images/prop_snapshots/output_5_75.png)

::: {.parsed-literal}
itration 2750
:::

![image](images/prop_snapshots/output_5_77.png)

::: {.parsed-literal}
itration 3000
:::

![image](images/prop_snapshots/output_5_79.png)

::: {.parsed-literal}
itration 3500
:::

![image](images/prop_snapshots/output_5_81.png)

::: {.parsed-literal}
itration 4000
:::

![image](images/prop_snapshots/output_5_83.png)

::: {.parsed-literal}
itration 4500
:::

![image](images/prop_snapshots/output_5_85.png)

::: {.parsed-literal}
itration 5000
:::

![image](images/prop_snapshots/output_5_87.png)

``` {.python}
from PIL import Image

# collect immages for each snapshot automatically by the algorithm in a folder named propagation
images = []
prop_folder = 'propagation'
img_files = os.listdir(prop_folder)
img_files = [os.path.join(prop_folder, f) for f in img_files]
sorted_files = sorted (img_files, key=os.path.getmtime) 
for filename in sorted_files:
  im = Image.open(filename)
  images.append(im)

# create annimated gif to show evolution of the propagation
images[0].save('propagation.gif', save_all=True, append_images=images[1:], optimize=False, duration=800, loop=1)
```
