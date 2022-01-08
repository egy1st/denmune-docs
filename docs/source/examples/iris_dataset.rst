Iris Dataset
===============

.. code:: ipython3

    import time
    import os.path
    import requests
    from numpy import genfromtxt
    !mkdir data #let us create data folder to hold our data

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
    data_file = data_path + dataset + file_ext #  i.e. 'iris' + '.txt' ==> iris.txt
    
    data_path = 'data/' # change it to whatever you put your data, set it to ''; so it will retrive from current folder
    if  not os.path.isfile(data_path + dataset + file_ext):
        req = requests.get(dataset_url)
        with open(data_path + dataset + file_ext, 'wb') as f:
            f.write(req.content)
    data = genfromtxt(data_file , delimiter='\t') 
    
    if  not os.path.isfile(data_path + dataset + ground_ext + file_ext):
        req = requests.get(groundtruth_url)
        with open(data_path + dataset +  ground_ext + file_ext, 'wb') as f:
            f.write(req.content)    
    data_labels =  genfromtxt(groundtruth_url , delimiter='\t') #  i.e. 'iris' + + '-gt + '.txt' ==> iris-gt.txt 
    
    data2d_ext = '-2d'
    file_2d =  data_path + dataset + data2d_ext + file_ext # 'iris' + '-2d' + '.txt' ==> iris-2d.txt

.. code:: ipython3

    # Denmune's Paramaters
    verpose_mode = True # view in-depth analysis of time complexity and outlier detection, num of clusters
    show_groundtrugh = True  # show plots on/off
    show_noise = True # show noise and outlier on/off
    
    knn = 11
    dm = DenMune(data=data, file_2d=file_2d,  k_nearest=knn, verpose=verpose_mode, show_noise=show_noise, rgn_tsne=False)
    
    if show_groundtrugh:
        # Let us plot the groundtruth of this dataset which is reduced to 2-d using t-SNE
        print ("Dataset\'s Groundtruth")
        dm.plot_clusters(labels=data_labels, ground=True)
        print('\n', "=====" * 20 , '\n')       
    
    labels_pred = dm.fit_predict()
    validity = dm.validate_Clusters(labels_true=data_labels, labels_pred=labels_pred)
    
    dm.plot_clusters(labels=labels_pred, show_noise=show_noise)
            
    validity_key = "F1"
    # Acc=1, F1-score=2,  NMI=3, AMI=4, ARI=5,  Homogeneity=6, and Completeness=7       
    print ('k=' , knn, validity_key , 'score is:', round(validity[validity_key],3))


.. parsed-literal::

    Dataset's Groundtruth



.. image:: datasets/iris/output_3_1.png


.. parsed-literal::

    
     ==================================================================================================== 
    
    DenMune Analyzer
    ├── exec_time
    │   ├── DenMune: 0.023
    │   └── NGT: 0.002
    ├── n_clusters
    │   ├── actual: 3
    │   └── detected: 3
    ├── n_points
    │   ├── dim: 4
    │   ├── noise
    │   │   ├── type-1: 0
    │   │   └── type-2: 0
    │   ├── size: 150
    │   ├── strong: 84
    │   └── weak
    │       ├── all: 66
    │       ├── failed to merge: 0
    │       └── succeeded to merge: 66
    └── validity
        ├── ACC: 135
        ├── AMI: 0.795
        ├── ARI: 0.746
        ├── F1: 0.898
        ├── NMI: 0.798
        ├── completeness: 0.809
        └── homogeneity: 0.787
    



.. image:: datasets/iris/output_3_3.png


.. parsed-literal::

    k= 11 F1 score is: 0.898


