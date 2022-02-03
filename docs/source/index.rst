Welcome to DenMune's documentation!
====================================

DenMune Clustering Algorithm
	A clustering algorithm that can find clusters of arbitrary size, shapes and densities in two-dimensions. Higher dimensions are first reduced to 2-D using the t-sne. The algorithm relies on a single parameter K (the number of nearest neighbors). The results show the superiority of DenMune. Enjoy the simplicty but the power of DenMune.

Check out the :doc:`installation` section for further information, including
how to :ref:`installation` the algorithm and use it.

.. note::

   This documentation associated with the paper "DenMune: Density peak based clustering using mutual nearest neighbors"
   
   DOI: https://doi.org/10.1016/j.patcog.2020.107589
   
   Source code is maintained at https://github.com/egy1st/denmune-clustering-algorithm
   
   

User Guide / Tutorials
------------------------

.. toctree::
   :maxdepth: 3
  
   
   README


Examples
----------------

.. toctree::
   :maxdepth: 2
   
   examples/iris_dataset
   examples/chameleon_datasets
   examples/2D_shapes_datasets
   examples/MNIST_dataset
   
  
Characteristics
----------------

.. toctree::
   :maxdepth: 2
   
   characteristics/noise_detection
   characteristics/clustering_propagation
   characteristics/clustering_propagation_snapshots
   characteristics/scalability_and_speed
   characteristics/stability_vs_knn
   characteristics/k_nearest_evolution
   
   
Participate in Competitions
----------------

.. toctree::
   :maxdepth: 2
   
   kaggle/training_MNIST
   kaggle/Get_97_by_training_MNIST_dataset
 
  
