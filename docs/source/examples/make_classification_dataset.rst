Make Classification Dataset
=============================

.. code:: ipython3

    import time
    import os.path
    import requests
    from numpy import genfromtxt
    from sklearn.datasets import make_moons, make_circles, make_classification, make_gaussian_quantiles
    import pandas as pd
    !mkdir data #let us create data folder to hold our data


.. parsed-literal::

    mkdir: cannot create directory ‘data’: File exists


.. code:: ipython3

    # install DenMune clustering algorithm using pip command from the offecial Python repository, PyPi
    # from https://pypi.org/project/denmune/
    !pip install denmune
    
    # now import it
    from denmune import DenMune

.. code:: ipython3

    X, y = make_classification(n_samples=1000, n_features=5, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1, class_sep=1)
    pd.DataFrame(X).head()




.. raw:: html

    
      <div id="df-378c5510-586c-4bf7-bcb4-145141b55e29">
        <div class="colab-df-container">
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
          <th>0</th>
          <th>1</th>
          <th>2</th>
          <th>3</th>
          <th>4</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.686468</td>
          <td>-1.271874</td>
          <td>-0.254311</td>
          <td>1.544239</td>
          <td>1.202059</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.509081</td>
          <td>-0.216813</td>
          <td>-0.831148</td>
          <td>-0.457612</td>
          <td>0.333293</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.736607</td>
          <td>-1.376652</td>
          <td>-0.762185</td>
          <td>-1.054598</td>
          <td>1.070726</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.884990</td>
          <td>0.065417</td>
          <td>-0.397411</td>
          <td>-1.020738</td>
          <td>0.706335</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.771464</td>
          <td>-0.465186</td>
          <td>0.011318</td>
          <td>1.591621</td>
          <td>1.691010</td>
        </tr>
      </tbody>
    </table>
    </div>
          <button class="colab-df-convert" onclick="convertToInteractive('df-378c5510-586c-4bf7-bcb4-145141b55e29')"
                  title="Convert this dataframe to an interactive table."
                  style="display:none;">
    
      <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
           width="24px">
        <path d="M0 0h24v24H0V0z" fill="none"/>
        <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
      </svg>
          </button>
    
      <style>
        .colab-df-container {
          display:flex;
          flex-wrap:wrap;
          gap: 12px;
        }
    
        .colab-df-convert {
          background-color: #E8F0FE;
          border: none;
          border-radius: 50%;
          cursor: pointer;
          display: none;
          fill: #1967D2;
          height: 32px;
          padding: 0 0 0 0;
          width: 32px;
        }
    
        .colab-df-convert:hover {
          background-color: #E2EBFA;
          box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
          fill: #174EA6;
        }
    
        [theme=dark] .colab-df-convert {
          background-color: #3B4455;
          fill: #D2E3FC;
        }
    
        [theme=dark] .colab-df-convert:hover {
          background-color: #434B5C;
          box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
          filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
          fill: #FFFFFF;
        }
      </style>
    
          <script>
            const buttonEl =
              document.querySelector('#df-378c5510-586c-4bf7-bcb4-145141b55e29 button.colab-df-convert');
            buttonEl.style.display =
              google.colab.kernel.accessAllowed ? 'block' : 'none';
    
            async function convertToInteractive(key) {
              const element = document.querySelector('#df-378c5510-586c-4bf7-bcb4-145141b55e29');
              const dataTable =
                await google.colab.kernel.invokeFunction('convertToInteractive',
                                                         [key], {});
              if (!dataTable) return;
    
              const docLinkHtml = 'Like what you see? Visit the ' +
                '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
                + ' to learn more about interactive tables.';
              element.innerHTML = '';
              dataTable['output_type'] = 'display_data';
              await google.colab.output.renderOutput(dataTable, element);
              const docLink = document.createElement('div');
              docLink.innerHTML = docLinkHtml;
              element.appendChild(docLink);
            }
          </script>
        </div>
      </div>




.. code:: ipython3

    data = X
    data_labels = y
    file_2d = 'data/classification-2d.txt'

.. code:: ipython3

    # Denmune's Paramaters
    verpose_mode = True # view in-depth analysis of time complexity and outlier detection, num of clusters
    show_groundtrugh = True  # show plots on/off
    show_noise = True # show noise and outlier on/off
    
    knn = 12
    dm = DenMune(data=data,  file_2d=file_2d, k_nearest=knn, verpose=verpose_mode, show_noise=show_noise, rgn_tsne=True )
    labels_pred = dm.fit_predict()
    
    if show_groundtrugh:
        # Let us plot the groundtruth of this dataset
        print ("classification dataset", ": Groundtruht")
        dm.plot_clusters(labels=data_labels, ground=True)
        print('\n', "=====" * 20 , '\n')       
    
    # Let us plot the results produced using DenMune
    print ("classification dataset", ": DenMune Clustering")
    dm.plot_clusters(labels=labels_pred, show_noise=show_noise)
    
    validity = dm.validate_Clusters(labels_true=data_labels, labels_pred=labels_pred)
    validity_key = "F1" 
    # Acc=1, F1-score=2,  NMI=3, AMI=4, ARI=5,  Homogeneity=6, and Completeness=7       
    print ('k=' , knn, validity_key , 'score is:', round(validity[validity_key],3))


.. parsed-literal::

    /usr/local/lib/python3.7/dist-packages/sklearn/manifold/_t_sne.py:793: FutureWarning: The default learning rate in TSNE will change from 200.0 to 'auto' in 1.2.
      FutureWarning,


.. parsed-literal::

    classification dataset : Groundtruht



.. image:: datasets/make_classification/output_4_2.png


.. parsed-literal::

    
     ==================================================================================================== 
    
    classification dataset : DenMune Clustering



.. image:: datasets/make_classification/output_4_4.png


.. parsed-literal::

    DenMune Analyzer
    ├── exec_time
    │   ├── DenMune: 0.127
    │   ├── NGT: 0.016
    │   └── t_SNE: 7.32
    ├── n_clusters
    │   ├── actual: 2
    │   └── detected: 8
    ├── n_points
    │   ├── dim: 5
    │   ├── noise
    │   │   ├── type-1: 0
    │   │   └── type-2: 10
    │   ├── size: 1000
    │   ├── strong: 562
    │   └── weak
    │       ├── all: 438
    │       ├── failed to merge: 10
    │       └── succeeded to merge: 428
    └── validity
        ├── ACC: 869
        ├── AMI: 0.471
        ├── ARI: 0.544
        ├── F1: 0.868
        ├── NMI: 0.471
        ├── completeness: 0.476
        └── homogeneity: 0.466
    
    k= 12 F1 score is: 0.868

