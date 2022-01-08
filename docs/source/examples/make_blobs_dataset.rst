Make Blobs Dataset
=====================

.. code:: ipython3

    import time
    import os.path
    import requests
    from numpy import genfromtxt
    from sklearn.datasets import make_blobs
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

    blobs, labels = make_blobs(n_samples=500, centers=3, n_features=10, cluster_std=2, random_state=0)
    pd.DataFrame(blobs).head()




.. raw:: html

    
      <div id="df-e4746d45-62b1-40d1-9460-69ecc44d07a8">
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
          <th>5</th>
          <th>6</th>
          <th>7</th>
          <th>8</th>
          <th>9</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>8.464776</td>
          <td>-0.069017</td>
          <td>1.756548</td>
          <td>8.707434</td>
          <td>-5.776232</td>
          <td>-7.940546</td>
          <td>-11.879435</td>
          <td>4.030456</td>
          <td>2.497293</td>
          <td>3.976303</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.280346</td>
          <td>1.597010</td>
          <td>-0.010019</td>
          <td>0.024167</td>
          <td>-4.812835</td>
          <td>2.105739</td>
          <td>-2.318796</td>
          <td>7.886270</td>
          <td>11.581623</td>
          <td>-1.986161</td>
        </tr>
        <tr>
          <th>2</th>
          <td>7.101569</td>
          <td>-0.639732</td>
          <td>0.908646</td>
          <td>12.363544</td>
          <td>-4.675757</td>
          <td>-5.777533</td>
          <td>-7.718462</td>
          <td>4.613895</td>
          <td>6.588259</td>
          <td>6.682010</td>
        </tr>
        <tr>
          <th>3</th>
          <td>6.498527</td>
          <td>4.858662</td>
          <td>-3.969435</td>
          <td>7.259363</td>
          <td>-2.651539</td>
          <td>4.822131</td>
          <td>-7.695410</td>
          <td>8.926791</td>
          <td>2.744806</td>
          <td>-2.908591</td>
        </tr>
        <tr>
          <th>4</th>
          <td>7.716717</td>
          <td>9.796369</td>
          <td>-2.834763</td>
          <td>5.255861</td>
          <td>-10.935268</td>
          <td>-2.278602</td>
          <td>-5.112753</td>
          <td>9.064783</td>
          <td>-3.030756</td>
          <td>-4.987964</td>
        </tr>
      </tbody>
    </table>
    </div>
          <button class="colab-df-convert" onclick="convertToInteractive('df-e4746d45-62b1-40d1-9460-69ecc44d07a8')"
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
              document.querySelector('#df-e4746d45-62b1-40d1-9460-69ecc44d07a8 button.colab-df-convert');
            buttonEl.style.display =
              google.colab.kernel.accessAllowed ? 'block' : 'none';
    
            async function convertToInteractive(key) {
              const element = document.querySelector('#df-e4746d45-62b1-40d1-9460-69ecc44d07a8');
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

    data = blobs
    data_labels = labels
    file_2d = 'data/blobs-2d.txt'

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
        print ("blobs dataset", ": Groundtruht")
        dm.plot_clusters(labels=data_labels, ground=True)
        print('\n', "=====" * 20 , '\n')       
    
    # Let us plot the results produced using DenMune
    print ("blobs dataset", ": DenMune Clustering")
    dm.plot_clusters(labels=labels_pred, show_noise=show_noise)
    
    validity = dm.validate_Clusters(labels_true=data_labels, labels_pred=labels_pred)
    validity_key = "F1" 
    # Acc=1, F1-score=2,  NMI=3, AMI=4, ARI=5,  Homogeneity=6, and Completeness=7       
    print ('k=' , knn, validity_key , 'score is:', round(validity[validity_key],3))


.. parsed-literal::

    /usr/local/lib/python3.7/dist-packages/sklearn/manifold/_t_sne.py:793: FutureWarning: The default learning rate in TSNE will change from 200.0 to 'auto' in 1.2.
      FutureWarning,


.. parsed-literal::

    blobs dataset : Groundtruht



.. image:: datasets/make_blobs/output_4_2.png


.. parsed-literal::

    
     ==================================================================================================== 
    
    blobs dataset : DenMune Clustering



.. image:: datasets/make_blobs/output_4_4.png


.. parsed-literal::

    DenMune Analyzer
    ├── exec_time
    │   ├── DenMune: 0.092
    │   ├── NGT: 0.017
    │   └── t_SNE: 3.127
    ├── n_clusters
    │   ├── actual: 3
    │   └── detected: 3
    ├── n_points
    │   ├── dim: 10
    │   ├── noise
    │   │   ├── type-1: 0
    │   │   └── type-2: 2
    │   ├── size: 500
    │   ├── strong: 308
    │   └── weak
    │       ├── all: 192
    │       ├── failed to merge: 2
    │       └── succeeded to merge: 190
    └── validity
        ├── ACC: 497
        ├── AMI: 0.969
        ├── ARI: 0.982
        ├── F1: 0.994
        ├── NMI: 0.969
        ├── completeness: 0.969
        └── homogeneity: 0.969
    
    k= 12 F1 score is: 0.994

