Ecoli Dataset
================

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

    dataset = 'ecoli' # let us take Ecoli dataset as an example
    
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

    using NGT, Proximity matrix has been calculated  in:  0.004878997802734375  seconds
    Dataset's Groundtruht



.. image:: datasets/ecoli/output_3_1.png


.. parsed-literal::

    
     ==================================================================================================== 
    
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 74 clusters 
    



.. image:: datasets/ecoli/output_3_3.png


.. parsed-literal::

    k= 2 :Validity score is: 0.15789277126213216 but best score is 0.15789277126213216 at k= 2     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.005110263824462891  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 11 weak point(s) in light grey (noise of type-2) represent 3% of total points
    DenMune detected 33 clusters 
    



.. image:: datasets/ecoli/output_3_5.png


.. parsed-literal::

    k= 3 :Validity score is: 0.3445717937800129 but best score is 0.3445717937800129 at k= 3     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.007807731628417969  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 16 weak point(s) in light grey (noise of type-2) represent 5% of total points
    DenMune detected 19 clusters 
    



.. image:: datasets/ecoli/output_3_7.png


.. parsed-literal::

    k= 4 :Validity score is: 0.4410243835729711 but best score is 0.4410243835729711 at k= 4     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.004739522933959961  seconds
    There are 1 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 9 weak point(s) in light grey (noise of type-2) represent 3% of total points
    DenMune detected 8 clusters 
    



.. image:: datasets/ecoli/output_3_9.png


.. parsed-literal::

    k= 5 :Validity score is: 0.6970568624801579 but best score is 0.6970568624801579 at k= 5     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.006937265396118164  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 4 weak point(s) in light grey (noise of type-2) represent 1% of total points
    DenMune detected 7 clusters 
    



.. image:: datasets/ecoli/output_3_11.png


.. parsed-literal::

    k= 6 :Validity score is: 0.7101310753802333 but best score is 0.7101310753802333 at k= 6     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.006165742874145508  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 6 clusters 
    



.. image:: datasets/ecoli/output_3_13.png


.. parsed-literal::

    k= 7 :Validity score is: 0.7023761083140417 but best score is 0.7101310753802333 at k= 6     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.005532026290893555  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 6 clusters 
    



.. image:: datasets/ecoli/output_3_15.png


.. parsed-literal::

    k= 8 :Validity score is: 0.7744417807485581 but best score is 0.7744417807485581 at k= 8     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.0053102970123291016  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 4 clusters 
    



.. image:: datasets/ecoli/output_3_17.png


.. parsed-literal::

    k= 9 :Validity score is: 0.7087311569686376 but best score is 0.7744417807485581 at k= 8     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.006548166275024414  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 4 clusters 
    



.. image:: datasets/ecoli/output_3_19.png


.. parsed-literal::

    k= 10 :Validity score is: 0.7077638497764626 but best score is 0.7744417807485581 at k= 8     
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
          <td>273.0</td>
          <td>0.774442</td>
          <td>0.718515</td>
          <td>0.708602</td>
          <td>0.764308</td>
          <td>0.676107</td>
          <td>0.766600</td>
          <td>0.049580</td>
        </tr>
        <tr>
          <th>4</th>
          <td>6.0</td>
          <td>232.0</td>
          <td>0.710131</td>
          <td>0.696202</td>
          <td>0.682739</td>
          <td>0.576016</td>
          <td>0.725418</td>
          <td>0.669249</td>
          <td>0.057938</td>
        </tr>
        <tr>
          <th>7</th>
          <td>9.0</td>
          <td>260.0</td>
          <td>0.708731</td>
          <td>0.693628</td>
          <td>0.686100</td>
          <td>0.735369</td>
          <td>0.617588</td>
          <td>0.791023</td>
          <td>0.136931</td>
        </tr>
        <tr>
          <th>8</th>
          <td>10.0</td>
          <td>260.0</td>
          <td>0.707764</td>
          <td>0.689723</td>
          <td>0.682099</td>
          <td>0.730317</td>
          <td>0.612488</td>
          <td>0.789247</td>
          <td>0.061325</td>
        </tr>
        <tr>
          <th>5</th>
          <td>7.0</td>
          <td>233.0</td>
          <td>0.702376</td>
          <td>0.657676</td>
          <td>0.645678</td>
          <td>0.552982</td>
          <td>0.663172</td>
          <td>0.652270</td>
          <td>0.053092</td>
        </tr>
        <tr>
          <th>3</th>
          <td>5.0</td>
          <td>223.0</td>
          <td>0.697057</td>
          <td>0.670618</td>
          <td>0.653434</td>
          <td>0.549727</td>
          <td>0.723473</td>
          <td>0.624960</td>
          <td>0.047864</td>
        </tr>
        <tr>
          <th>2</th>
          <td>4.0</td>
          <td>119.0</td>
          <td>0.441024</td>
          <td>0.545320</td>
          <td>0.505488</td>
          <td>0.231977</td>
          <td>0.758403</td>
          <td>0.425710</td>
          <td>0.126078</td>
        </tr>
        <tr>
          <th>1</th>
          <td>3.0</td>
          <td>85.0</td>
          <td>0.344572</td>
          <td>0.458034</td>
          <td>0.387350</td>
          <td>0.110004</td>
          <td>0.730313</td>
          <td>0.333644</td>
          <td>0.042913</td>
        </tr>
        <tr>
          <th>0</th>
          <td>2.0</td>
          <td>37.0</td>
          <td>0.157893</td>
          <td>0.381214</td>
          <td>0.234958</td>
          <td>0.024755</td>
          <td>0.692580</td>
          <td>0.262983</td>
          <td>0.122756</td>
        </tr>
      </tbody>
    </table>
    </div>


