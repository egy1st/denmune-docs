Iris Dataset
============

.. code:: python3

    import time
    import os.path
    import requests
    import pandas as pd

.. code:: python3

    # install DenMune clustering algorithm using pip command from the offecial Python repository, PyPi
    # from https://pypi.org/project/denmune/
    !pip install denmune
    
    # now import it
    from denmune import DenMune

.. code:: python3

    dataset = 'iris' # let us take iris dataset as an example
    
    url = "https://zerobytes.one/denmune_data/"
    file_ext = ".txt"
    ground_ext = "-gt"
    
    dataset_url = url + dataset + file_ext
    groundtruth_url = url + dataset + ground_ext  + file_ext
    
    data_path = 'data/' # change it to whatever you put your data, set it to ''; so it will retrive from current folder
    if  not os.path.isfile(data_path + dataset + file_ext):
        req = requests.get(dataset_url)
        with open(data_path + dataset + file_ext, 'wb') as f:
            f.write(req.content)
            
    if  not os.path.isfile(data_path + dataset + ground_ext + file_ext):
        req = requests.get(groundtruth_url)
        with open(data_path + dataset +  ground_ext + file_ext, 'wb') as f:
            f.write(req.content)       

.. code:: python3

    # Denmune's Paramaters
    # DenMune(dataset=dataset, k_nearest=n, data_path=data_path, verpose=verpose_mode, show_plot=show_plot, show_noise=show_noise)
    verpose_mode = True # view in-depth analysis of time complexity and outlier detection, num of clusters
    show_plot = True  # show plots on/off
    show_noise = True # show noise and outlier on/off
    
    # loop's parameters
    start = 3
    step = 1
    end=15
    
    # Validity indexes' parameters
    validity_val = -1
    best_k = 0
    best_val = -1
    
    validity_idx = 1 # Acc=1, F1-score=2,  NMI=3, AMI=4, ARI=5,  Homogeneity=6, and Completeness=7
    df = pd.DataFrame(columns =['K', 'ACC', 'F1', 'NMI', 'AMI', 'ARI','Homogeneity', 'Completeness', 'Time' ])
    
    
    for n in range(start, end+1, step):
        start_time = time.time()
        dm = DenMune(dataset=dataset, k_nearest=n, data_path=data_path, verpose=verpose_mode, show_noise=show_noise)
        labels_true, labels_pred = dm.output_Clusters()
        if show_plot == True and n==start:
            # Let us plot the groundtruth of this dataset which is reduced to 2-d using t-SNE
            print ("Dataset\'s Groundtruht")
            dm.plot_clusters(labels_true, ground=True)
            print('\n', "=====" * 20 , '\n')       
                   
        end_time = time.time()
        
        validity_indexes = dm.validate_Clusters(labels_true, labels_pred)
        validity_val = validity_indexes[validity_idx]
        validity_indexes[0] = n
        validity_indexes[8] = end_time - start_time
        
        df = df.append(pd.Series(validity_indexes, index=df.columns ), ignore_index=True)
        
        if (best_val < validity_val):
            best_val = validity_val
            best_k = n
            # Let us show results where only an improve in accuracy is detected
        if show_plot:
                dm.plot_clusters(labels_pred, show_noise=show_noise)
        print ('k=' , n, ':Validity score is:', validity_val , 'but best score is', best_val, 'at k=', best_k , end='     ')
                
        if not verpose_mode:
            print('\r', end='')
        else:
            print('\n', "=====" * 20 , '\n')


.. parsed-literal::

    using NGT, Proximity matrix has been calculated  in:  0.0019257068634033203  seconds
    Dataset's Groundtruht



.. image:: datasets/iris/output_3_1.png


.. parsed-literal::

    
     ==================================================================================================== 
    
    There are 13 outlier point(s) in black (noise of type-1) represent 9% of total points
    There are 8 weak point(s) in light grey (noise of type-2) represent 5% of total points
    DenMune detected 25 clusters 
    



.. image:: datasets/iris/output_3_3.png


.. parsed-literal::

    k= 3 :Validity score is: 26 but best score is 26 at k= 3     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.0013012886047363281  seconds
    There are 5 outlier point(s) in black (noise of type-1) represent 3% of total points
    There are 10 weak point(s) in light grey (noise of type-2) represent 7% of total points
    DenMune detected 12 clusters 
    



.. image:: datasets/iris/output_3_5.png


.. parsed-literal::

    k= 4 :Validity score is: 49 but best score is 49 at k= 4     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.0016024112701416016  seconds
    There are 2 outlier point(s) in black (noise of type-1) represent 1% of total points
    There are 9 weak point(s) in light grey (noise of type-2) represent 6% of total points
    DenMune detected 9 clusters 
    



.. image:: datasets/iris/output_3_7.png


.. parsed-literal::

    k= 5 :Validity score is: 67 but best score is 67 at k= 5     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.0015184879302978516  seconds
    There are 2 outlier point(s) in black (noise of type-1) represent 1% of total points
    There are 5 weak point(s) in light grey (noise of type-2) represent 3% of total points
    DenMune detected 7 clusters 
    



.. image:: datasets/iris/output_3_9.png


.. parsed-literal::

    k= 6 :Validity score is: 84 but best score is 84 at k= 6     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.0016658306121826172  seconds
    There are 2 outlier point(s) in black (noise of type-1) represent 1% of total points
    There are 2 weak point(s) in light grey (noise of type-2) represent 1% of total points
    DenMune detected 5 clusters 
    



.. image:: datasets/iris/output_3_11.png


.. parsed-literal::

    k= 7 :Validity score is: 120 but best score is 120 at k= 7     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.0016393661499023438  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 2 weak point(s) in light grey (noise of type-2) represent 1% of total points
    DenMune detected 4 clusters 
    



.. image:: datasets/iris/output_3_13.png


.. parsed-literal::

    k= 8 :Validity score is: 130 but best score is 130 at k= 8     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.0018620491027832031  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 2 weak point(s) in light grey (noise of type-2) represent 1% of total points
    DenMune detected 4 clusters 
    



.. image:: datasets/iris/output_3_15.png


.. parsed-literal::

    k= 9 :Validity score is: 122 but best score is 130 at k= 8     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.0017979145050048828  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 2 weak point(s) in light grey (noise of type-2) represent 1% of total points
    DenMune detected 4 clusters 
    



.. image:: datasets/iris/output_3_17.png


.. parsed-literal::

    k= 10 :Validity score is: 112 but best score is 130 at k= 8     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.00472259521484375  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 2 weak point(s) in light grey (noise of type-2) represent 1% of total points
    DenMune detected 3 clusters 
    



.. image:: datasets/iris/output_3_19.png


.. parsed-literal::

    k= 11 :Validity score is: 133 but best score is 133 at k= 11     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.006127357482910156  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 3 clusters 
    



.. image:: datasets/iris/output_3_21.png


.. parsed-literal::

    k= 12 :Validity score is: 134 but best score is 134 at k= 12     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.004782199859619141  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 3 clusters 
    



.. image:: datasets/iris/output_3_23.png


.. parsed-literal::

    k= 13 :Validity score is: 134 but best score is 134 at k= 12     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.0021009445190429688  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 3 clusters 
    



.. image:: datasets/iris/output_3_25.png


.. parsed-literal::

    k= 14 :Validity score is: 135 but best score is 135 at k= 14     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.0020799636840820312  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 3 clusters 
    



.. image:: datasets/iris/output_3_27.png


.. parsed-literal::

    k= 15 :Validity score is: 134 but best score is 135 at k= 14     
     ==================================================================================================== 
    



.. parsed-literal::

    <Figure size 432x288 with 0 Axes>


.. code:: python3

    # It is time to save the results
    results_path = 'results/'  # change it to whatever you output results to, set it to ''; so it will output to current folder
    para_file = 'denmune'+ '_para_'  + dataset + '.csv'
    df.sort_values(by=['ACC', 'F1', 'NMI', 'ARI'] , ascending=False, inplace=True)   
    df.to_csv(results_path + para_file, index=False, sep='\t', header=True)

.. code:: python3

    df # it is sorted now and saved




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>K</th>
          <th>ACC</th>
          <th>F1</th>
          <th>NMI</th>
          <th>AMI</th>
          <th>ARI</th>
          <th>Homogeneity</th>
          <th>Completeness</th>
          <th>Time</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>11</th>
          <td>14.0</td>
          <td>135.0</td>
          <td>0.897698</td>
          <td>0.797989</td>
          <td>0.795421</td>
          <td>0.745504</td>
          <td>0.786923</td>
          <td>0.809369</td>
          <td>0.025140</td>
        </tr>
        <tr>
          <th>9</th>
          <td>12.0</td>
          <td>134.0</td>
          <td>0.890531</td>
          <td>0.790679</td>
          <td>0.788012</td>
          <td>0.732298</td>
          <td>0.778177</td>
          <td>0.803589</td>
          <td>0.029522</td>
        </tr>
        <tr>
          <th>10</th>
          <td>13.0</td>
          <td>134.0</td>
          <td>0.890531</td>
          <td>0.790679</td>
          <td>0.788012</td>
          <td>0.732298</td>
          <td>0.778177</td>
          <td>0.803589</td>
          <td>0.027311</td>
        </tr>
        <tr>
          <th>12</th>
          <td>15.0</td>
          <td>134.0</td>
          <td>0.890531</td>
          <td>0.790679</td>
          <td>0.788012</td>
          <td>0.732298</td>
          <td>0.778177</td>
          <td>0.803589</td>
          <td>0.022892</td>
        </tr>
        <tr>
          <th>8</th>
          <td>11.0</td>
          <td>133.0</td>
          <td>0.891029</td>
          <td>0.779845</td>
          <td>0.775379</td>
          <td>0.730006</td>
          <td>0.790165</td>
          <td>0.769792</td>
          <td>0.022020</td>
        </tr>
        <tr>
          <th>5</th>
          <td>8.0</td>
          <td>130.0</td>
          <td>0.920343</td>
          <td>0.820395</td>
          <td>0.816057</td>
          <td>0.817625</td>
          <td>0.922692</td>
          <td>0.738517</td>
          <td>0.016305</td>
        </tr>
        <tr>
          <th>6</th>
          <td>9.0</td>
          <td>122.0</td>
          <td>0.835264</td>
          <td>0.736792</td>
          <td>0.730139</td>
          <td>0.678943</td>
          <td>0.799032</td>
          <td>0.683547</td>
          <td>0.016929</td>
        </tr>
        <tr>
          <th>4</th>
          <td>7.0</td>
          <td>120.0</td>
          <td>0.868994</td>
          <td>0.797335</td>
          <td>0.790210</td>
          <td>0.776531</td>
          <td>0.963419</td>
          <td>0.680094</td>
          <td>0.094015</td>
        </tr>
        <tr>
          <th>7</th>
          <td>10.0</td>
          <td>112.0</td>
          <td>0.838612</td>
          <td>0.725948</td>
          <td>0.719470</td>
          <td>0.671153</td>
          <td>0.830880</td>
          <td>0.644547</td>
          <td>0.027200</td>
        </tr>
        <tr>
          <th>3</th>
          <td>6.0</td>
          <td>84.0</td>
          <td>0.715190</td>
          <td>0.681570</td>
          <td>0.669026</td>
          <td>0.493952</td>
          <td>0.963419</td>
          <td>0.527306</td>
          <td>0.014004</td>
        </tr>
        <tr>
          <th>2</th>
          <td>5.0</td>
          <td>67.0</td>
          <td>0.616738</td>
          <td>0.592427</td>
          <td>0.574010</td>
          <td>0.359015</td>
          <td>0.911344</td>
          <td>0.438854</td>
          <td>0.014182</td>
        </tr>
        <tr>
          <th>1</th>
          <td>4.0</td>
          <td>49.0</td>
          <td>0.487165</td>
          <td>0.532330</td>
          <td>0.505790</td>
          <td>0.264754</td>
          <td>0.881604</td>
          <td>0.381276</td>
          <td>0.014421</td>
        </tr>
        <tr>
          <th>0</th>
          <td>3.0</td>
          <td>26.0</td>
          <td>0.295019</td>
          <td>0.434869</td>
          <td>0.373416</td>
          <td>0.103594</td>
          <td>0.850955</td>
          <td>0.292062</td>
          <td>0.074632</td>
        </tr>
      </tbody>
    </table>
    </div>



