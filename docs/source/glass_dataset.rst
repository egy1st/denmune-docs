Glass Datasets
==================

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

    dataset = 'glass' # let us take Glass dataset as an example
    
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
    start = 2
    step = 1
    end=10
    
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

    using NGT, Proximity matrix has been calculated  in:  0.0025110244750976562  seconds
    Dataset's Groundtruht



.. image:: datasets/glass/output_3_1.png


.. parsed-literal::

    
     ==================================================================================================== 
    
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 50 clusters 
    



.. image:: datasets/glass/output_3_3.png


.. parsed-literal::

    k= 2 :Validity score is: 0.2375600902830109 but best score is 0.2375600902830109 at k= 2     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.0030586719512939453  seconds
    There are 8 outlier point(s) in black (noise of type-1) represent 4% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 26 clusters 
    



.. image:: datasets/glass/output_3_5.png


.. parsed-literal::

    k= 3 :Validity score is: 0.36396584442776486 but best score is 0.36396584442776486 at k= 3     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.0038933753967285156  seconds
    There are 6 outlier point(s) in black (noise of type-1) represent 3% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 19 clusters 
    



.. image:: datasets/glass/output_3_7.png


.. parsed-literal::

    k= 4 :Validity score is: 0.41684192757612853 but best score is 0.41684192757612853 at k= 4     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.003717660903930664  seconds
    There are 2 outlier point(s) in black (noise of type-1) represent 1% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 11 clusters 
    



.. image:: datasets/glass/output_3_9.png


.. parsed-literal::

    k= 5 :Validity score is: 0.4562103910870234 but best score is 0.4562103910870234 at k= 5     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.004549741744995117  seconds
    There are 1 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 16 weak point(s) in light grey (noise of type-2) represent 7% of total points
    DenMune detected 10 clusters 
    



.. image:: datasets/glass/output_3_11.png


.. parsed-literal::

    k= 6 :Validity score is: 0.49588157935097343 but best score is 0.49588157935097343 at k= 6     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.004144430160522461  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 13 weak point(s) in light grey (noise of type-2) represent 6% of total points
    DenMune detected 8 clusters 
    



.. image:: datasets/glass/output_3_13.png


.. parsed-literal::

    k= 7 :Validity score is: 0.5417712505562974 but best score is 0.5417712505562974 at k= 7     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.0032761096954345703  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 7 weak point(s) in light grey (noise of type-2) represent 3% of total points
    DenMune detected 7 clusters 
    



.. image:: datasets/glass/output_3_15.png


.. parsed-literal::

    k= 8 :Validity score is: 0.5272764618559012 but best score is 0.5417712505562974 at k= 7     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.003448963165283203  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 2 weak point(s) in light grey (noise of type-2) represent 1% of total points
    DenMune detected 7 clusters 
    



.. image:: datasets/glass/output_3_17.png


.. parsed-literal::

    k= 9 :Validity score is: 0.4361904623485281 but best score is 0.5417712505562974 at k= 7     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.003370523452758789  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 6 weak point(s) in light grey (noise of type-2) represent 3% of total points
    DenMune detected 6 clusters 
    



.. image:: datasets/glass/output_3_19.png


.. parsed-literal::

    k= 10 :Validity score is: 0.4275862128413732 but best score is 0.5417712505562974 at k= 7     
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
          <th>5</th>
          <td>7.0</td>
          <td>105.0</td>
          <td>0.541771</td>
          <td>0.402761</td>
          <td>0.364177</td>
          <td>0.264027</td>
          <td>0.462331</td>
          <td>0.356790</td>
          <td>0.038398</td>
        </tr>
        <tr>
          <th>6</th>
          <td>8.0</td>
          <td>106.0</td>
          <td>0.527276</td>
          <td>0.380386</td>
          <td>0.344334</td>
          <td>0.240828</td>
          <td>0.419384</td>
          <td>0.348023</td>
          <td>0.030751</td>
        </tr>
        <tr>
          <th>4</th>
          <td>6.0</td>
          <td>87.0</td>
          <td>0.495882</td>
          <td>0.387788</td>
          <td>0.339877</td>
          <td>0.207087</td>
          <td>0.487230</td>
          <td>0.322057</td>
          <td>0.036412</td>
        </tr>
        <tr>
          <th>3</th>
          <td>5.0</td>
          <td>83.0</td>
          <td>0.456210</td>
          <td>0.366724</td>
          <td>0.315996</td>
          <td>0.173227</td>
          <td>0.449674</td>
          <td>0.309612</td>
          <td>0.032428</td>
        </tr>
        <tr>
          <th>7</th>
          <td>9.0</td>
          <td>84.0</td>
          <td>0.436190</td>
          <td>0.366186</td>
          <td>0.331234</td>
          <td>0.171317</td>
          <td>0.409536</td>
          <td>0.331134</td>
          <td>0.033949</td>
        </tr>
        <tr>
          <th>8</th>
          <td>10.0</td>
          <td>83.0</td>
          <td>0.427586</td>
          <td>0.372920</td>
          <td>0.342055</td>
          <td>0.166891</td>
          <td>0.408232</td>
          <td>0.343230</td>
          <td>0.038563</td>
        </tr>
        <tr>
          <th>2</th>
          <td>4.0</td>
          <td>68.0</td>
          <td>0.416842</td>
          <td>0.378511</td>
          <td>0.300766</td>
          <td>0.138580</td>
          <td>0.521643</td>
          <td>0.297014</td>
          <td>0.036200</td>
        </tr>
        <tr>
          <th>1</th>
          <td>3.0</td>
          <td>54.0</td>
          <td>0.363966</td>
          <td>0.396516</td>
          <td>0.301184</td>
          <td>0.097928</td>
          <td>0.606552</td>
          <td>0.294528</td>
          <td>0.026664</td>
        </tr>
        <tr>
          <th>0</th>
          <td>2.0</td>
          <td>34.0</td>
          <td>0.237560</td>
          <td>0.384558</td>
          <td>0.225363</td>
          <td>0.040990</td>
          <td>0.663697</td>
          <td>0.270705</td>
          <td>0.139132</td>
        </tr>
      </tbody>
    </table>
    </div>


