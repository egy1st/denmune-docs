Aggregation Dataset
===================

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

    dataset = 'aggregation' # let us take Aggregation dataset as an example
    
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

    using NGT, Proximity matrix has been calculated  in:  0.007555723190307617  seconds
    Dataset's Groundtruht



.. image:: datasets/aggregation/output_3_1.png


.. parsed-literal::

    
     ==================================================================================================== 
    
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 150 clusters 
    



.. image:: datasets/aggregation/output_3_3.png


.. parsed-literal::

    k= 2 :Validity score is: 0.10834819591164768 but best score is 0.10834819591164768 at k= 2     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.007392406463623047  seconds
    There are 16 outlier point(s) in black (noise of type-1) represent 2% of total points
    There are 19 weak point(s) in light grey (noise of type-2) represent 2% of total points
    DenMune detected 43 clusters 
    



.. image:: datasets/aggregation/output_3_5.png


.. parsed-literal::

    k= 3 :Validity score is: 0.5970123672945185 but best score is 0.5970123672945185 at k= 3     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.008713006973266602  seconds
    There are 7 outlier point(s) in black (noise of type-1) represent 1% of total points
    There are 5 weak point(s) in light grey (noise of type-2) represent 1% of total points
    DenMune detected 20 clusters 
    



.. image:: datasets/aggregation/output_3_7.png


.. parsed-literal::

    k= 4 :Validity score is: 0.8971268233976194 but best score is 0.8971268233976194 at k= 4     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.007703065872192383  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 3 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 7 clusters 
    



.. image:: datasets/aggregation/output_3_9.png


.. parsed-literal::

    k= 5 :Validity score is: 0.9929458471817214 but best score is 0.9929458471817214 at k= 5     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.008939266204833984  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 7 clusters 
    



.. image:: datasets/aggregation/output_3_11.png


.. parsed-literal::

    k= 6 :Validity score is: 0.9962034083064701 but best score is 0.9962034083064701 at k= 6     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.010357379913330078  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 7 clusters 
    



.. image:: datasets/aggregation/output_3_13.png


.. parsed-literal::

    k= 7 :Validity score is: 0.9962034083064701 but best score is 0.9962034083064701 at k= 6     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.009857654571533203  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 7 clusters 
    



.. image:: datasets/aggregation/output_3_15.png


.. parsed-literal::

    k= 8 :Validity score is: 0.9962034083064701 but best score is 0.9962034083064701 at k= 6     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.009833574295043945  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 7 clusters 
    



.. image:: datasets/aggregation/output_3_17.png


.. parsed-literal::

    k= 9 :Validity score is: 0.9974644121825882 but best score is 0.9974644121825882 at k= 9     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.013320684432983398  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 7 clusters 
    



.. image:: datasets/aggregation/output_3_19.png


.. parsed-literal::

    k= 10 :Validity score is: 0.9974644121825882 but best score is 0.9974644121825882 at k= 9     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.09010028839111328  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 7 clusters 
    



.. image:: datasets/aggregation/output_3_21.png


.. parsed-literal::

    k= 11 :Validity score is: 0.9949579337028716 but best score is 0.9974644121825882 at k= 9     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.011791706085205078  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 7 clusters 
    



.. image:: datasets/aggregation/output_3_23.png


.. parsed-literal::

    k= 12 :Validity score is: 0.9974706059239578 but best score is 0.9974706059239578 at k= 12     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.011208295822143555  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 7 clusters 
    



.. image:: datasets/aggregation/output_3_25.png


.. parsed-literal::

    k= 13 :Validity score is: 0.9974706059239578 but best score is 0.9974706059239578 at k= 12     
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
          <th>10</th>
          <td>12.0</td>
          <td>786.0</td>
          <td>0.997471</td>
          <td>0.991530</td>
          <td>0.991408</td>
          <td>0.994888</td>
          <td>0.992384</td>
          <td>0.990678</td>
          <td>0.113831</td>
        </tr>
        <tr>
          <th>11</th>
          <td>13.0</td>
          <td>786.0</td>
          <td>0.997471</td>
          <td>0.991530</td>
          <td>0.991408</td>
          <td>0.994888</td>
          <td>0.992384</td>
          <td>0.990678</td>
          <td>0.178088</td>
        </tr>
        <tr>
          <th>7</th>
          <td>9.0</td>
          <td>786.0</td>
          <td>0.997464</td>
          <td>0.992432</td>
          <td>0.992323</td>
          <td>0.995626</td>
          <td>0.992600</td>
          <td>0.992265</td>
          <td>0.093172</td>
        </tr>
        <tr>
          <th>8</th>
          <td>10.0</td>
          <td>786.0</td>
          <td>0.997464</td>
          <td>0.992432</td>
          <td>0.992323</td>
          <td>0.995626</td>
          <td>0.992600</td>
          <td>0.992265</td>
          <td>0.096574</td>
        </tr>
        <tr>
          <th>4</th>
          <td>6.0</td>
          <td>785.0</td>
          <td>0.996203</td>
          <td>0.988268</td>
          <td>0.988098</td>
          <td>0.992708</td>
          <td>0.989199</td>
          <td>0.987339</td>
          <td>0.146099</td>
        </tr>
        <tr>
          <th>5</th>
          <td>7.0</td>
          <td>785.0</td>
          <td>0.996203</td>
          <td>0.988268</td>
          <td>0.988098</td>
          <td>0.992708</td>
          <td>0.989199</td>
          <td>0.987339</td>
          <td>0.082990</td>
        </tr>
        <tr>
          <th>6</th>
          <td>8.0</td>
          <td>785.0</td>
          <td>0.996203</td>
          <td>0.988268</td>
          <td>0.988098</td>
          <td>0.992708</td>
          <td>0.989199</td>
          <td>0.987339</td>
          <td>0.155642</td>
        </tr>
        <tr>
          <th>9</th>
          <td>11.0</td>
          <td>784.0</td>
          <td>0.994958</td>
          <td>0.985137</td>
          <td>0.984922</td>
          <td>0.989801</td>
          <td>0.986816</td>
          <td>0.983464</td>
          <td>0.184558</td>
        </tr>
        <tr>
          <th>3</th>
          <td>5.0</td>
          <td>781.0</td>
          <td>0.992946</td>
          <td>0.981672</td>
          <td>0.981370</td>
          <td>0.989655</td>
          <td>0.986399</td>
          <td>0.976989</td>
          <td>0.070746</td>
        </tr>
        <tr>
          <th>2</th>
          <td>4.0</td>
          <td>654.0</td>
          <td>0.897127</td>
          <td>0.850063</td>
          <td>0.843625</td>
          <td>0.803326</td>
          <td>0.994903</td>
          <td>0.742036</td>
          <td>0.068205</td>
        </tr>
        <tr>
          <th>1</th>
          <td>3.0</td>
          <td>347.0</td>
          <td>0.597012</td>
          <td>0.629226</td>
          <td>0.601307</td>
          <td>0.301077</td>
          <td>0.940503</td>
          <td>0.472758</td>
          <td>0.057692</td>
        </tr>
        <tr>
          <th>0</th>
          <td>2.0</td>
          <td>81.0</td>
          <td>0.108348</td>
          <td>0.408280</td>
          <td>0.284173</td>
          <td>0.013082</td>
          <td>0.676628</td>
          <td>0.292339</td>
          <td>0.226802</td>
        </tr>
      </tbody>
    </table>
    </div>



