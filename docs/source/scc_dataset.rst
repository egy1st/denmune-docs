SCC Datasets
======================

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

    dataset = 'scc' # let us take SCC dataset as an example
    
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

    using NGT, Proximity matrix has been calculated  in:  0.007254600524902344  seconds
    Dataset's Groundtruht



.. image:: datasets/scc/output_3_1.png


.. parsed-literal::

    
     ==================================================================================================== 
    
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 129 clusters 
    



.. image:: datasets/scc/output_3_3.png


.. parsed-literal::

    k= 2 :Validity score is: 0.09835024418357752 but best score is 0.09835024418357752 at k= 2     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.0072324275970458984  seconds
    There are 18 outlier point(s) in black (noise of type-1) represent 3% of total points
    There are 26 weak point(s) in light grey (noise of type-2) represent 4% of total points
    DenMune detected 73 clusters 
    



.. image:: datasets/scc/output_3_5.png


.. parsed-literal::

    k= 3 :Validity score is: 0.3286584848053895 but best score is 0.3286584848053895 at k= 3     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.007908821105957031  seconds
    There are 6 outlier point(s) in black (noise of type-1) represent 1% of total points
    There are 23 weak point(s) in light grey (noise of type-2) represent 4% of total points
    DenMune detected 43 clusters 
    



.. image:: datasets/scc/output_3_7.png


.. parsed-literal::

    k= 4 :Validity score is: 0.49446653318259903 but best score is 0.49446653318259903 at k= 4     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.010877847671508789  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 18 weak point(s) in light grey (noise of type-2) represent 3% of total points
    DenMune detected 29 clusters 
    



.. image:: datasets/scc/output_3_9.png


.. parsed-literal::

    k= 5 :Validity score is: 0.5442734499878952 but best score is 0.5442734499878952 at k= 5     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.013741016387939453  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 21 weak point(s) in light grey (noise of type-2) represent 4% of total points
    DenMune detected 18 clusters 
    



.. image:: datasets/scc/output_3_11.png


.. parsed-literal::

    k= 6 :Validity score is: 0.7246681384283954 but best score is 0.7246681384283954 at k= 6     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.010501861572265625  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 26 weak point(s) in light grey (noise of type-2) represent 4% of total points
    DenMune detected 15 clusters 
    



.. image:: datasets/scc/output_3_13.png


.. parsed-literal::

    k= 7 :Validity score is: 0.7624303963017123 but best score is 0.7624303963017123 at k= 7     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.010219335556030273  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 21 weak point(s) in light grey (noise of type-2) represent 4% of total points
    DenMune detected 12 clusters 
    



.. image:: datasets/scc/output_3_15.png


.. parsed-literal::

    k= 8 :Validity score is: 0.595266505504429 but best score is 0.7624303963017123 at k= 7     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.010495662689208984  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 20 weak point(s) in light grey (noise of type-2) represent 3% of total points
    DenMune detected 12 clusters 
    



.. image:: datasets/scc/output_3_17.png


.. parsed-literal::

    k= 9 :Validity score is: 0.5836640455744316 but best score is 0.7624303963017123 at k= 7     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.012023210525512695  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 15 weak point(s) in light grey (noise of type-2) represent 2% of total points
    DenMune detected 13 clusters 
    



.. image:: datasets/scc/output_3_19.png


.. parsed-literal::

    k= 10 :Validity score is: 0.5672771672771673 but best score is 0.7624303963017123 at k= 7     
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
          <td>399.0</td>
          <td>0.762430</td>
          <td>0.786899</td>
          <td>0.779852</td>
          <td>0.680220</td>
          <td>0.937075</td>
          <td>0.678209</td>
          <td>0.136638</td>
        </tr>
        <tr>
          <th>4</th>
          <td>6.0</td>
          <td>378.0</td>
          <td>0.724668</td>
          <td>0.771757</td>
          <td>0.762786</td>
          <td>0.648432</td>
          <td>0.944800</td>
          <td>0.652289</td>
          <td>0.222926</td>
        </tr>
        <tr>
          <th>6</th>
          <td>8.0</td>
          <td>335.0</td>
          <td>0.595267</td>
          <td>0.754502</td>
          <td>0.747738</td>
          <td>0.575578</td>
          <td>0.843129</td>
          <td>0.682736</td>
          <td>0.131110</td>
        </tr>
        <tr>
          <th>7</th>
          <td>9.0</td>
          <td>328.0</td>
          <td>0.583664</td>
          <td>0.749817</td>
          <td>0.742916</td>
          <td>0.567887</td>
          <td>0.836848</td>
          <td>0.679183</td>
          <td>0.132345</td>
        </tr>
        <tr>
          <th>8</th>
          <td>10.0</td>
          <td>321.0</td>
          <td>0.567277</td>
          <td>0.747440</td>
          <td>0.739842</td>
          <td>0.565242</td>
          <td>0.839463</td>
          <td>0.673599</td>
          <td>0.217946</td>
        </tr>
        <tr>
          <th>3</th>
          <td>5.0</td>
          <td>293.0</td>
          <td>0.544273</td>
          <td>0.705337</td>
          <td>0.686910</td>
          <td>0.528638</td>
          <td>0.918451</td>
          <td>0.572497</td>
          <td>0.119797</td>
        </tr>
        <tr>
          <th>2</th>
          <td>4.0</td>
          <td>221.0</td>
          <td>0.494467</td>
          <td>0.636824</td>
          <td>0.604386</td>
          <td>0.339141</td>
          <td>0.930864</td>
          <td>0.483953</td>
          <td>0.108957</td>
        </tr>
        <tr>
          <th>1</th>
          <td>3.0</td>
          <td>123.0</td>
          <td>0.328658</td>
          <td>0.572873</td>
          <td>0.513081</td>
          <td>0.160912</td>
          <td>0.941885</td>
          <td>0.411612</td>
          <td>0.112248</td>
        </tr>
        <tr>
          <th>0</th>
          <td>2.0</td>
          <td>38.0</td>
          <td>0.098350</td>
          <td>0.469504</td>
          <td>0.343993</td>
          <td>0.044928</td>
          <td>0.826038</td>
          <td>0.327953</td>
          <td>0.232555</td>
        </tr>
      </tbody>
    </table>
    </div>


