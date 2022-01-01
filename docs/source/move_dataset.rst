Libras Movement dataset
========================

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

    dataset = 'move' # let us take Libras movement dataset as an example
    
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
    start = 3
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

    using NGT, Proximity matrix has been calculated  in:  0.0065157413482666016  seconds
    Dataset's Groundtruht



.. image:: datasets/move/output_3_1.png


.. parsed-literal::

    
     ==================================================================================================== 
    
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 33 weak point(s) in light grey (noise of type-2) represent 9% of total points
    DenMune detected 50 clusters 
    



.. image:: datasets/move/output_3_3.png


.. parsed-literal::

    k= 3 :Validity score is: 0.4142348645571525 but best score is 0.4142348645571525 at k= 3     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.006665229797363281  seconds
    There are 8 outlier point(s) in black (noise of type-1) represent 2% of total points
    There are 21 weak point(s) in light grey (noise of type-2) represent 6% of total points
    DenMune detected 41 clusters 
    



.. image:: datasets/move/output_3_5.png


.. parsed-literal::

    k= 4 :Validity score is: 0.3873107979641555 but best score is 0.4142348645571525 at k= 3     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.006779193878173828  seconds
    There are 2 outlier point(s) in black (noise of type-1) represent 1% of total points
    There are 23 weak point(s) in light grey (noise of type-2) represent 6% of total points
    DenMune detected 24 clusters 
    



.. image:: datasets/move/output_3_7.png


.. parsed-literal::

    k= 5 :Validity score is: 0.3569652002786488 but best score is 0.4142348645571525 at k= 3     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.007827281951904297  seconds
    There are 1 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 19 weak point(s) in light grey (noise of type-2) represent 5% of total points
    DenMune detected 20 clusters 
    



.. image:: datasets/move/output_3_9.png


.. parsed-literal::

    k= 6 :Validity score is: 0.3366497245909011 but best score is 0.4142348645571525 at k= 3     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.006575107574462891  seconds
    There are 1 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 12 weak point(s) in light grey (noise of type-2) represent 3% of total points
    DenMune detected 14 clusters 
    



.. image:: datasets/move/output_3_11.png


.. parsed-literal::

    k= 7 :Validity score is: 0.3561433694438948 but best score is 0.4142348645571525 at k= 3     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.006296634674072266  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 31 weak point(s) in light grey (noise of type-2) represent 9% of total points
    DenMune detected 12 clusters 
    



.. image:: datasets/move/output_3_13.png


.. parsed-literal::

    k= 8 :Validity score is: 0.3501005681603664 but best score is 0.4142348645571525 at k= 3     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.006155252456665039  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 19 weak point(s) in light grey (noise of type-2) represent 5% of total points
    DenMune detected 11 clusters 
    



.. image:: datasets/move/output_3_15.png


.. parsed-literal::

    k= 9 :Validity score is: 0.34252654386635384 but best score is 0.4142348645571525 at k= 3     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.008177518844604492  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 17 weak point(s) in light grey (noise of type-2) represent 5% of total points
    DenMune detected 11 clusters 
    



.. image:: datasets/move/output_3_17.png


.. parsed-literal::

    k= 10 :Validity score is: 0.34275501442730516 but best score is 0.4142348645571525 at k= 3     
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
          <th>0</th>
          <td>3.0</td>
          <td>109.0</td>
          <td>0.414235</td>
          <td>0.646454</td>
          <td>0.504685</td>
          <td>0.243622</td>
          <td>0.766996</td>
          <td>0.558655</td>
          <td>0.216693</td>
        </tr>
        <tr>
          <th>1</th>
          <td>4.0</td>
          <td>105.0</td>
          <td>0.387311</td>
          <td>0.663924</td>
          <td>0.543933</td>
          <td>0.280881</td>
          <td>0.775327</td>
          <td>0.580513</td>
          <td>0.195564</td>
        </tr>
        <tr>
          <th>2</th>
          <td>5.0</td>
          <td>118.0</td>
          <td>0.356965</td>
          <td>0.652170</td>
          <td>0.573484</td>
          <td>0.324351</td>
          <td>0.691686</td>
          <td>0.616926</td>
          <td>0.086581</td>
        </tr>
        <tr>
          <th>4</th>
          <td>7.0</td>
          <td>139.0</td>
          <td>0.356143</td>
          <td>0.672236</td>
          <td>0.626687</td>
          <td>0.391285</td>
          <td>0.655574</td>
          <td>0.689768</td>
          <td>0.100869</td>
        </tr>
        <tr>
          <th>5</th>
          <td>8.0</td>
          <td>139.0</td>
          <td>0.350101</td>
          <td>0.645810</td>
          <td>0.604253</td>
          <td>0.375294</td>
          <td>0.609102</td>
          <td>0.687228</td>
          <td>0.092846</td>
        </tr>
        <tr>
          <th>7</th>
          <td>10.0</td>
          <td>143.0</td>
          <td>0.342755</td>
          <td>0.641107</td>
          <td>0.601974</td>
          <td>0.368364</td>
          <td>0.595668</td>
          <td>0.694050</td>
          <td>0.110062</td>
        </tr>
        <tr>
          <th>6</th>
          <td>9.0</td>
          <td>142.0</td>
          <td>0.342527</td>
          <td>0.636100</td>
          <td>0.596491</td>
          <td>0.363439</td>
          <td>0.590300</td>
          <td>0.689604</td>
          <td>0.186325</td>
        </tr>
        <tr>
          <th>3</th>
          <td>6.0</td>
          <td>118.0</td>
          <td>0.336650</td>
          <td>0.656211</td>
          <td>0.592238</td>
          <td>0.344386</td>
          <td>0.673948</td>
          <td>0.639384</td>
          <td>0.104372</td>
        </tr>
      </tbody>
    </table>
    </div>


