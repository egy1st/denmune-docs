Chameleon DS2 Dataset
=====================

.. code:: ipython3

    import time
    import os.path
    import requests

.. code:: ipython3

    # install DenMune clustering algorithm using pip command from the offecial Python repository, PyPi
    # from https://pypi.org/project/denmune/
    !pip install denmune
    
    # now import it
    from denmune import DenMune

.. code:: ipython3

    dataset = 'cham_02' # let us take Chameleon DS2 dataset as an example
    
    url = "https://zerobytes.one/denmune_data/"
    file_ext = ".txt"
    # ground_ext = "-gt" # no groundtruth for this dataset
    
    dataset_url = url + dataset + file_ext
    # groundtruth_url = url + dataset + ground_ext  + file_ext # no groundtruth for this dataset
    
    data_path = 'data/' # change it to whatever you put your data, set it to ''; so it will retrive from current folder
    if  not os.path.isfile(data_path + dataset + file_ext):
        req = requests.get(dataset_url)
        with open(data_path + dataset + file_ext, 'wb') as f:
            f.write(req.content)

.. code:: ipython3

    # Denmune's Paramaters
    # DenMune(dataset=dataset, k_nearest=n, data_path=data_path, verpose=verpose_mode, show_plot=show_plot, show_noise=show_noise)
    verpose_mode = True # view in-depth analysis of time complexity and outlier detection, num of clusters
    show_plot = True  # show plots on/off
    show_noise = True # show noise and outlier on/off
    
    # loop's parameters
    start = 5
    step = 5
    end=45
    
    for n in range(start, end+1, step):
        start_time = time.time()
        dm = DenMune(dataset=dataset, k_nearest=n, data_path=data_path, verpose=verpose_mode, show_noise=show_noise)
        labels_true, labels_pred = dm.output_Clusters()
        end_time = time.time()
        
       
        if show_plot:
            dm.plot_clusters(labels_pred, show_noise=show_noise)
        print ('k=' , n , end='     ')
                
        if not verpose_mode:
            print('\r', end='')
        else:
            print('\n', "=====" * 20 , '\n')


.. parsed-literal::

    using NGT, Proximity matrix has been calculated  in:  0.12062621116638184  seconds
    There are 98 outlier point(s) in black (noise of type-1) represent 1% of total points
    There are 413 weak point(s) in light grey (noise of type-2) represent 5% of total points
    DenMune detected 434 clusters 
    



.. image:: datasets/cham_02/output_3_1.png


.. parsed-literal::

    k= 5     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.28055691719055176  seconds
    There are 17 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 392 weak point(s) in light grey (noise of type-2) represent 5% of total points
    DenMune detected 61 clusters 
    



.. image:: datasets/cham_02/output_3_3.png


.. parsed-literal::

    k= 10     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.1849689483642578  seconds
    There are 6 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 417 weak point(s) in light grey (noise of type-2) represent 5% of total points
    DenMune detected 31 clusters 
    



.. image:: datasets/cham_02/output_3_5.png


.. parsed-literal::

    k= 15     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.2597064971923828  seconds
    There are 1 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 447 weak point(s) in light grey (noise of type-2) represent 6% of total points
    DenMune detected 15 clusters 
    



.. image:: datasets/cham_02/output_3_7.png


.. parsed-literal::

    k= 20     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.30834388732910156  seconds
    There are 1 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 459 weak point(s) in light grey (noise of type-2) represent 6% of total points
    DenMune detected 15 clusters 
    



.. image:: datasets/cham_02/output_3_9.png


.. parsed-literal::

    k= 25     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.48705267906188965  seconds
    There are 1 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 610 weak point(s) in light grey (noise of type-2) represent 8% of total points
    DenMune detected 12 clusters 
    



.. image:: datasets/cham_02/output_3_11.png


.. parsed-literal::

    k= 30     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.4156203269958496  seconds
    There are 1 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 662 weak point(s) in light grey (noise of type-2) represent 8% of total points
    DenMune detected 8 clusters 
    



.. image:: datasets/cham_02/output_3_13.png


.. parsed-literal::

    k= 35     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.39724206924438477  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 681 weak point(s) in light grey (noise of type-2) represent 9% of total points
    DenMune detected 6 clusters 
    



.. image:: datasets/cham_02/output_3_15.png


.. parsed-literal::

    k= 40     
     ==================================================================================================== 
    
    using NGT, Proximity matrix has been calculated  in:  0.38170456886291504  seconds
    There are 0 outlier point(s) in black (noise of type-1) represent 0% of total points
    There are 724 weak point(s) in light grey (noise of type-2) represent 9% of total points
    DenMune detected 6 clusters 
    



.. image:: datasets/cham_02/output_3_17.png


.. parsed-literal::

    k= 45     
     ==================================================================================================== 
    



.. parsed-literal::

    <Figure size 432x288 with 0 Axes>

