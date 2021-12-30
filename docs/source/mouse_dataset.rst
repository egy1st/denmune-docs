Mouse Datase
============

.. code:: ipython3

    import time
    import os.path
    import requests
    import pandas as pd

.. code:: ipython3

    # install DenMune clustering algorithm using pip command from the offecial Python repository, PyPi
    # from https://pypi.org/project/denmune/
    !pip install denmune
    
    # now import it
    from denmune import DenMune

.. code:: ipython3

    dataset = 'mouse' # let us take Mouse dataset as an example
    
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

.. code:: ipython3

    # Denmune's Paramaters
    # DenMune(dataset=dataset, k_nearest=n, data_path=data_path, verpose=verpose_mode, show_plot=show_plot, show_noise=show_noise)
    verpose_mode = True # view in-depth analysis of time complexity and outlier detection, num of clusters
    show_plot = True  # show plots on/off
    show_noise = True # show noise and outlier on/off
    
    # loop's parameters
    start = 5
    step = 1
    end=13
    
    # Validity indexes' parameters
    validity_val = -1
    best_k = 0
    best_val = -1
    
    validity_idx = 2 # Acc=1, F1-score=2,  NMI=3, AMI=4, ARI=5,  Homogeneity=6, and Completeness=7
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

    using NGT, Proximity matrix has been calculated  in:  0.005628824234008789  seconds
    Dataset's Groundtruht



.. image:: datasets/mouse/output_3_1.png


.. parsed-literal::

    
     ==================================================================================================== 
    
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 21 weak point(s) in light grey (noise of type-2) represent 4% of total points
    DenMune detected 24 clusters 
    



.. image:: datasets/mouse/output_3_3.png


.. parsed-literal::

    k= 5 :Validity score is: 0.49039093884166574 but best score is 0.49039093884166574 at k= 5     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.00585174560546875  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 15 weak point(s) in light grey (noise of type-2) represent 3% of total points
    DenMune detected 15 clusters 
    



.. image:: datasets/mouse/output_3_5.png


.. parsed-literal::

    k= 6 :Validity score is: 0.4064557249579739 but best score is 0.49039093884166574 at k= 5     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.005789279937744141  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 21 weak point(s) in light grey (noise of type-2) represent 4% of total points
    DenMune detected 11 clusters 
    



.. image:: datasets/mouse/output_3_7.png


.. parsed-literal::

    k= 7 :Validity score is: 0.8333651083172287 but best score is 0.8333651083172287 at k= 7     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.006821632385253906  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 17 weak point(s) in light grey (noise of type-2) represent 3% of total points
    DenMune detected 11 clusters 
    



.. image:: datasets/mouse/output_3_9.png


.. parsed-literal::

    k= 8 :Validity score is: 0.7205043800382904 but best score is 0.8333651083172287 at k= 7     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.00662994384765625  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 3 weak point(s) in light grey (noise of type-2) represent 1% of total points
    DenMune detected 6 clusters 
    



.. image:: datasets/mouse/output_3_11.png


.. parsed-literal::

    k= 9 :Validity score is: 0.956823452493869 but best score is 0.956823452493869 at k= 9     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.010288000106811523  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 3 weak point(s) in light grey (noise of type-2) represent 1% of total points
    DenMune detected 4 clusters 
    



.. image:: datasets/mouse/output_3_13.png


.. parsed-literal::

    k= 10 :Validity score is: 0.9839611023896739 but best score is 0.9839611023896739 at k= 10     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.0076448917388916016  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 3 weak point(s) in light grey (noise of type-2) represent 1% of total points
    DenMune detected 4 clusters 
    



.. image:: datasets/mouse/output_3_15.png


.. parsed-literal::

    k= 11 :Validity score is: 0.9859507397315909 but best score is 0.9859507397315909 at k= 11     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.00804758071899414  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 3 weak point(s) in light grey (noise of type-2) represent 1% of total points
    DenMune detected 4 clusters 
    



.. image:: datasets/mouse/output_3_17.png


.. parsed-literal::

    k= 12 :Validity score is: 0.983919505038908 but best score is 0.9859507397315909 at k= 11     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.007779359817504883  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 7 weak point(s) in light grey (noise of type-2) represent 1% of total points
    DenMune detected 4 clusters 
    



.. image:: datasets/mouse/output_3_19.png


.. parsed-literal::

    k= 13 :Validity score is: 0.981923342471317 but best score is 0.9859507397315909 at k= 11     
     ==================================================================================================== 
    



.. parsed-literal::

    <Figure size 432x288 with 0 Axes>


.. code:: ipython3

    # It is time to save the results
    results_path = 'results/'  # change it to whatever you output results to, set it to ''; so it will output to current folder
    para_file = 'denmune'+ '_para_'  + dataset + '.csv'
    df.sort_values(by=['F1', 'NMI', 'ARI'] , ascending=False, inplace=True)   
    df.to_csv(results_path + para_file, index=False, sep='\t', header=True)

.. code:: ipython3

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
          <th>6</th>
          <td>11.0</td>
          <td>492.0</td>
          <td>0.985951</td>
          <td>0.949327</td>
          <td>0.948731</td>
          <td>0.972490</td>
          <td>0.948414</td>
          <td>0.950242</td>
          <td>0.131318</td>
        </tr>
        <tr>
          <th>5</th>
          <td>10.0</td>
          <td>491.0</td>
          <td>0.983961</td>
          <td>0.937424</td>
          <td>0.936688</td>
          <td>0.966055</td>
          <td>0.935552</td>
          <td>0.939304</td>
          <td>0.074179</td>
        </tr>
        <tr>
          <th>7</th>
          <td>12.0</td>
          <td>491.0</td>
          <td>0.983920</td>
          <td>0.942282</td>
          <td>0.941603</td>
          <td>0.966237</td>
          <td>0.940354</td>
          <td>0.944218</td>
          <td>0.062571</td>
        </tr>
        <tr>
          <th>8</th>
          <td>13.0</td>
          <td>488.0</td>
          <td>0.981923</td>
          <td>0.925591</td>
          <td>0.924713</td>
          <td>0.953574</td>
          <td>0.939309</td>
          <td>0.912268</td>
          <td>0.067738</td>
        </tr>
        <tr>
          <th>4</th>
          <td>9.0</td>
          <td>464.0</td>
          <td>0.956823</td>
          <td>0.856576</td>
          <td>0.854273</td>
          <td>0.858343</td>
          <td>0.942980</td>
          <td>0.784678</td>
          <td>0.051092</td>
        </tr>
        <tr>
          <th>2</th>
          <td>7.0</td>
          <td>367.0</td>
          <td>0.833365</td>
          <td>0.646641</td>
          <td>0.638226</td>
          <td>0.595048</td>
          <td>0.873006</td>
          <td>0.513495</td>
          <td>0.045316</td>
        </tr>
        <tr>
          <th>3</th>
          <td>8.0</td>
          <td>291.0</td>
          <td>0.720504</td>
          <td>0.620970</td>
          <td>0.612466</td>
          <td>0.437110</td>
          <td>0.901818</td>
          <td>0.473508</td>
          <td>0.126676</td>
        </tr>
        <tr>
          <th>0</th>
          <td>5.0</td>
          <td>193.0</td>
          <td>0.490391</td>
          <td>0.473207</td>
          <td>0.452669</td>
          <td>0.174208</td>
          <td>0.885879</td>
          <td>0.322825</td>
          <td>0.112229</td>
        </tr>
        <tr>
          <th>1</th>
          <td>6.0</td>
          <td>170.0</td>
          <td>0.406456</td>
          <td>0.472052</td>
          <td>0.456412</td>
          <td>0.183621</td>
          <td>0.703445</td>
          <td>0.355209</td>
          <td>0.046375</td>
        </tr>
      </tbody>
    </table>
    </div>


