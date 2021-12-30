Pathbased Dataset
=================

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

    dataset = 'pathbased' # let us take Pathbased dataset as an example
    
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

    using NGT, Proximity matrix has been calculated  in:  0.003040790557861328  seconds
    Dataset's Groundtruht



.. image:: datasets/pathbased/output_3_1.png


.. parsed-literal::

    
     ==================================================================================================== 
    
    There are 7 outlier point(s) in black (noise of type-1) represent 2% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 33 clusters 
    



.. image:: datasets/pathbased/output_3_3.png


.. parsed-literal::

    k= 3 :Validity score is: 0.38905812773403325 but best score is 0.38905812773403325 at k= 3     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.0027437210083007812  seconds
    There are 5 outlier point(s) in black (noise of type-1) represent 2% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 19 clusters 
    



.. image:: datasets/pathbased/output_3_5.png


.. parsed-literal::

    k= 4 :Validity score is: 0.5946664942598677 but best score is 0.5946664942598677 at k= 4     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.004683017730712891  seconds
    There are 2 outlier point(s) in black (noise of type-1) represent 1% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 7 clusters 
    



.. image:: datasets/pathbased/output_3_7.png


.. parsed-literal::

    k= 5 :Validity score is: 0.8584971213251468 but best score is 0.8584971213251468 at k= 5     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.005736112594604492  seconds
    There are 1 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 3 clusters 
    



.. image:: datasets/pathbased/output_3_9.png


.. parsed-literal::

    k= 6 :Validity score is: 0.9781764895594682 but best score is 0.9781764895594682 at k= 6     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.0030863285064697266  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 5 clusters 
    



.. image:: datasets/pathbased/output_3_11.png


.. parsed-literal::

    k= 7 :Validity score is: 0.8485152019285412 but best score is 0.9781764895594682 at k= 6     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.0059092044830322266  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 3 clusters 
    



.. image:: datasets/pathbased/output_3_13.png


.. parsed-literal::

    k= 8 :Validity score is: 0.9632669749767612 but best score is 0.9781764895594682 at k= 6     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.005806684494018555  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 3 clusters 
    



.. image:: datasets/pathbased/output_3_15.png


.. parsed-literal::

    k= 9 :Validity score is: 0.9632669749767612 but best score is 0.9781764895594682 at k= 6     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.003769397735595703  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 5 clusters 
    



.. image:: datasets/pathbased/output_3_17.png


.. parsed-literal::

    k= 10 :Validity score is: 0.8023816299192644 but best score is 0.9781764895594682 at k= 6     
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
          <th>3</th>
          <td>6.0</td>
          <td>293.0</td>
          <td>0.978176</td>
          <td>0.907532</td>
          <td>0.906654</td>
          <td>0.937889</td>
          <td>0.914701</td>
          <td>0.900476</td>
          <td>0.027950</td>
        </tr>
        <tr>
          <th>5</th>
          <td>8.0</td>
          <td>289.0</td>
          <td>0.963267</td>
          <td>0.853170</td>
          <td>0.852261</td>
          <td>0.891123</td>
          <td>0.851352</td>
          <td>0.854996</td>
          <td>0.033110</td>
        </tr>
        <tr>
          <th>6</th>
          <td>9.0</td>
          <td>289.0</td>
          <td>0.963267</td>
          <td>0.853170</td>
          <td>0.852261</td>
          <td>0.891123</td>
          <td>0.851352</td>
          <td>0.854996</td>
          <td>0.029872</td>
        </tr>
        <tr>
          <th>2</th>
          <td>5.0</td>
          <td>236.0</td>
          <td>0.858497</td>
          <td>0.734994</td>
          <td>0.729705</td>
          <td>0.725052</td>
          <td>0.896653</td>
          <td>0.622722</td>
          <td>0.042256</td>
        </tr>
        <tr>
          <th>4</th>
          <td>7.0</td>
          <td>232.0</td>
          <td>0.848515</td>
          <td>0.729561</td>
          <td>0.726569</td>
          <td>0.697346</td>
          <td>0.846194</td>
          <td>0.641185</td>
          <td>0.026819</td>
        </tr>
        <tr>
          <th>7</th>
          <td>10.0</td>
          <td>233.0</td>
          <td>0.802382</td>
          <td>0.633090</td>
          <td>0.628855</td>
          <td>0.581756</td>
          <td>0.701576</td>
          <td>0.576786</td>
          <td>0.031490</td>
        </tr>
        <tr>
          <th>1</th>
          <td>4.0</td>
          <td>132.0</td>
          <td>0.594666</td>
          <td>0.525774</td>
          <td>0.506783</td>
          <td>0.305610</td>
          <td>0.893260</td>
          <td>0.372520</td>
          <td>0.021981</td>
        </tr>
        <tr>
          <th>0</th>
          <td>3.0</td>
          <td>75.0</td>
          <td>0.389058</td>
          <td>0.443954</td>
          <td>0.409240</td>
          <td>0.136267</td>
          <td>0.890032</td>
          <td>0.295735</td>
          <td>0.084656</td>
        </tr>
      </tbody>
    </table>
    </div>


