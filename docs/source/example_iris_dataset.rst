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

.. code:: ipython3

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



.. image:: output_3_1.png


.. parsed-literal::

    
     ==================================================================================================== 
    
    There are 13 outlier point(s) in black (noise of type-1) represent 9% of total points
    There are 8 weak point(s) in light grey (noise of type-2) represent 5% of total points
    DenMune detected 25 clusters 
    



.. image:: output_3_3.png


.. parsed-literal::

    k= 3 :Validity score is: 26 but best score is 26 at k= 3     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.0013012886047363281  seconds
    There are 5 outlier point(s) in black (noise of type-1) represent 3% of total points
    There are 10 weak point(s) in light grey (noise of type-2) represent 7% of total points
    DenMune detected 12 clusters 
    



.. image:: output_3_5.png


.. parsed-literal::

    k= 4 :Validity score is: 49 but best score is 49 at k= 4     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.0016024112701416016  seconds
    There are 2 outlier point(s) in black (noise of type-1) represent 1% of total points
    There are 9 weak point(s) in light grey (noise of type-2) represent 6% of total points
    DenMune detected 9 clusters 
    



.. image:: output_3_7.png


.. parsed-literal::

    k= 5 :Validity score is: 67 but best score is 67 at k= 5     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.0015184879302978516  seconds
    There are 2 outlier point(s) in black (noise of type-1) represent 1% of total points
    There are 5 weak point(s) in light grey (noise of type-2) represent 3% of total points
    DenMune detected 7 clusters 
    



.. image:: output_3_9.png


.. parsed-literal::

    k= 6 :Validity score is: 84 but best score is 84 at k= 6     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.0016658306121826172  seconds
    There are 2 outlier point(s) in black (noise of type-1) represent 1% of total points
    There are 2 weak point(s) in light grey (noise of type-2) represent 1% of total points
    DenMune detected 5 clusters 
    



.. image:: output_3_11.png


.. parsed-literal::

    k= 7 :Validity score is: 120 but best score is 120 at k= 7     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.0016393661499023438  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 2 weak point(s) in light grey (noise of type-2) represent 1% of total points
    DenMune detected 4 clusters 
    



.. image:: output_3_13.png


.. parsed-literal::

    k= 8 :Validity score is: 130 but best score is 130 at k= 8     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.0018620491027832031  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 2 weak point(s) in light grey (noise of type-2) represent 1% of total points
    DenMune detected 4 clusters 
    



.. image:: output_3_15.png


.. parsed-literal::

    k= 9 :Validity score is: 122 but best score is 130 at k= 8     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.0017979145050048828  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 2 weak point(s) in light grey (noise of type-2) represent 1% of total points
    DenMune detected 4 clusters 
    



.. image:: output_3_17.png


.. parsed-literal::

    k= 10 :Validity score is: 112 but best score is 130 at k= 8     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.00472259521484375  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 2 weak point(s) in light grey (noise of type-2) represent 1% of total points
    DenMune detected 3 clusters 
    



.. image:: output_3_19.png


.. parsed-literal::

    k= 11 :Validity score is: 133 but best score is 133 at k= 11     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.006127357482910156  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 3 clusters 
    



.. image:: output_3_21.png


.. parsed-literal::

    k= 12 :Validity score is: 134 but best score is 134 at k= 12     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.004782199859619141  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 3 clusters 
    



.. image:: output_3_23.png


.. parsed-literal::

    k= 13 :Validity score is: 134 but best score is 134 at k= 12     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.0021009445190429688  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 3 clusters 
    



.. image:: output_3_25.png


.. parsed-literal::

    k= 14 :Validity score is: 135 but best score is 135 at k= 14     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.0020799636840820312  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 0 weak point(s) in light grey (noise of type-2) represent 0% of total points
    DenMune detected 3 clusters 
    



.. image:: output_3_27.png


.. parsed-literal::

    k= 15 :Validity score is: 134 but best score is 135 at k= 14     
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
          <td>8.0</td>
          <td>133.0</td>
          <td>0.932131</td>
          <td>0.841868</td>
          <td>0.839116</td>
          <td>0.842942</td>
          <td>0.923649</td>
          <td>0.773391</td>
          <td>0.021382</td>
        </tr>
        <tr>
          <th>5</th>
          <td>10.0</td>
          <td>135.0</td>
          <td>0.897698</td>
          <td>0.797989</td>
          <td>0.795421</td>
          <td>0.745504</td>
          <td>0.786923</td>
          <td>0.809369</td>
          <td>0.017886</td>
        </tr>
        <tr>
          <th>6</th>
          <td>11.0</td>
          <td>135.0</td>
          <td>0.897698</td>
          <td>0.797989</td>
          <td>0.795421</td>
          <td>0.745504</td>
          <td>0.786923</td>
          <td>0.809369</td>
          <td>0.017965</td>
        </tr>
        <tr>
          <th>7</th>
          <td>12.0</td>
          <td>135.0</td>
          <td>0.897698</td>
          <td>0.797989</td>
          <td>0.795421</td>
          <td>0.745504</td>
          <td>0.786923</td>
          <td>0.809369</td>
          <td>0.018378</td>
        </tr>
        <tr>
          <th>8</th>
          <td>13.0</td>
          <td>135.0</td>
          <td>0.897698</td>
          <td>0.797989</td>
          <td>0.795421</td>
          <td>0.745504</td>
          <td>0.786923</td>
          <td>0.809369</td>
          <td>0.020036</td>
        </tr>
        <tr>
          <th>9</th>
          <td>14.0</td>
          <td>135.0</td>
          <td>0.897698</td>
          <td>0.797989</td>
          <td>0.795421</td>
          <td>0.745504</td>
          <td>0.786923</td>
          <td>0.809369</td>
          <td>0.019303</td>
        </tr>
        <tr>
          <th>10</th>
          <td>15.0</td>
          <td>135.0</td>
          <td>0.897698</td>
          <td>0.797989</td>
          <td>0.795421</td>
          <td>0.745504</td>
          <td>0.786923</td>
          <td>0.809369</td>
          <td>0.022101</td>
        </tr>
        <tr>
          <th>2</th>
          <td>7.0</td>
          <td>121.0</td>
          <td>0.869277</td>
          <td>0.784554</td>
          <td>0.778492</td>
          <td>0.775061</td>
          <td>0.922431</td>
          <td>0.682535</td>
          <td>0.086369</td>
        </tr>
        <tr>
          <th>4</th>
          <td>9.0</td>
          <td>102.0</td>
          <td>0.785836</td>
          <td>0.699696</td>
          <td>0.693196</td>
          <td>0.619015</td>
          <td>0.833488</td>
          <td>0.602915</td>
          <td>0.017289</td>
        </tr>
        <tr>
          <th>0</th>
          <td>5.0</td>
          <td>77.0</td>
          <td>0.676262</td>
          <td>0.633626</td>
          <td>0.617857</td>
          <td>0.446222</td>
          <td>0.928236</td>
          <td>0.480971</td>
          <td>0.076828</td>
        </tr>
        <tr>
          <th>1</th>
          <td>6.0</td>
          <td>76.0</td>
          <td>0.670988</td>
          <td>0.671427</td>
          <td>0.658472</td>
          <td>0.469120</td>
          <td>0.951927</td>
          <td>0.518611</td>
          <td>0.014116</td>
        </tr>
      </tbody>
    </table>
    </div>



