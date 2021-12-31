Flame Dataset
==============

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

    dataset = 'flame' # let us take Jain dataset as an example
    
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

    using NGT, Proximity matrix has been calculated  in:  0.0040798187255859375  seconds
    Dataset's Groundtruht



.. image:: datasets/flame/output_3_1.png


.. parsed-literal::

    
     ==================================================================================================== 
    
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 43 clusters 
    



.. image:: datasets/flame/output_3_3.png


.. parsed-literal::

    k= 2 :Validity score is: 0.31182636077372916 but best score is 0.31182636077372916 at k= 2     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.004065752029418945  seconds
    There are 2 outlier point(s) in black (noise of type-1) represent 1% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 7 clusters 
    



.. image:: datasets/flame/output_3_5.png


.. parsed-literal::

    k= 3 :Validity score is: 0.5065426029962546 but best score is 0.5065426029962546 at k= 3     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.0032880306243896484  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 2 clusters 
    



.. image:: datasets/flame/output_3_7.png


.. parsed-literal::

    k= 4 :Validity score is: 0.979192037470726 but best score is 0.979192037470726 at k= 4     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.004467964172363281  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 2 clusters 
    



.. image:: datasets/flame/output_3_9.png


.. parsed-literal::

    k= 5 :Validity score is: 0.5152051783097216 but best score is 0.979192037470726 at k= 4     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.005274772644042969  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 2 clusters 
    



.. image:: datasets/flame/output_3_11.png


.. parsed-literal::

    k= 6 :Validity score is: 0.9833732057416268 but best score is 0.9833732057416268 at k= 6     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.005054473876953125  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 2 clusters 
    



.. image:: datasets/flame/output_3_13.png


.. parsed-literal::

    k= 7 :Validity score is: 0.9916452733313198 but best score is 0.9916452733313198 at k= 7     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.005687236785888672  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 2 clusters 
    



.. image:: datasets/flame/output_3_15.png


.. parsed-literal::

    k= 8 :Validity score is: 1.0 but best score is 1.0 at k= 8     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.004950761795043945  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 2 clusters 
    



.. image:: datasets/flame/output_3_17.png


.. parsed-literal::

    k= 9 :Validity score is: 0.9916866028708136 but best score is 1.0 at k= 8     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.006613016128540039  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 2 clusters 
    



.. image:: datasets/flame/output_3_19.png


.. parsed-literal::

    k= 10 :Validity score is: 0.9875152224824355 but best score is 1.0 at k= 8     
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
          <td>8.0</td>
          <td>240.0</td>
          <td>1.000000</td>
          <td>1.000000</td>
          <td>1.000000</td>
          <td>1.000000</td>
          <td>1.000000</td>
          <td>1.000000</td>
          <td>0.046381</td>
        </tr>
        <tr>
          <th>7</th>
          <td>9.0</td>
          <td>238.0</td>
          <td>0.991687</td>
          <td>0.935864</td>
          <td>0.935658</td>
          <td>0.966656</td>
          <td>0.939118</td>
          <td>0.932631</td>
          <td>0.039253</td>
        </tr>
        <tr>
          <th>5</th>
          <td>7.0</td>
          <td>238.0</td>
          <td>0.991645</td>
          <td>0.935464</td>
          <td>0.935256</td>
          <td>0.966611</td>
          <td>0.931996</td>
          <td>0.938958</td>
          <td>0.042520</td>
        </tr>
        <tr>
          <th>8</th>
          <td>10.0</td>
          <td>237.0</td>
          <td>0.987515</td>
          <td>0.899366</td>
          <td>0.899043</td>
          <td>0.950178</td>
          <td>0.900956</td>
          <td>0.897782</td>
          <td>0.140920</td>
        </tr>
        <tr>
          <th>4</th>
          <td>6.0</td>
          <td>236.0</td>
          <td>0.983373</td>
          <td>0.875217</td>
          <td>0.874817</td>
          <td>0.933872</td>
          <td>0.878260</td>
          <td>0.872194</td>
          <td>0.041173</td>
        </tr>
        <tr>
          <th>2</th>
          <td>4.0</td>
          <td>235.0</td>
          <td>0.979192</td>
          <td>0.847495</td>
          <td>0.847005</td>
          <td>0.917664</td>
          <td>0.848992</td>
          <td>0.846002</td>
          <td>0.029567</td>
        </tr>
        <tr>
          <th>3</th>
          <td>5.0</td>
          <td>155.0</td>
          <td>0.515205</td>
          <td>0.024231</td>
          <td>0.016415</td>
          <td>0.012750</td>
          <td>0.013007</td>
          <td>0.176739</td>
          <td>0.041492</td>
        </tr>
        <tr>
          <th>1</th>
          <td>3.0</td>
          <td>137.0</td>
          <td>0.506543</td>
          <td>0.122928</td>
          <td>0.100085</td>
          <td>0.033245</td>
          <td>0.128593</td>
          <td>0.117741</td>
          <td>0.032018</td>
        </tr>
        <tr>
          <th>0</th>
          <td>2.0</td>
          <td>48.0</td>
          <td>0.311826</td>
          <td>0.241364</td>
          <td>0.199074</td>
          <td>0.028343</td>
          <td>0.747887</td>
          <td>0.143903</td>
          <td>0.111955</td>
        </tr>
      </tbody>
    </table>
    </div>


