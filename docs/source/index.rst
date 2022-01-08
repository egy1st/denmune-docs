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
   installation
   
  
Real Datasets
----------------

.. toctree::
   :maxdepth: 2
   
   examples/iris_dataset
   examples/seeds_dataset
   examples/mnist_dataset
   
     
   
Synthestic Datasets
-------------------
   
.. toctree::
   :maxdepth: 2
   
   examples/aggregation_dataset  
   examples/jain_dataset
   examples/flame_dataset
   examples/compound_dataset
   examples/vary_density_dataset
   examples/unbalance_dataset
   examples/spiral_dataset
   examples/pathbased_dataset
   examples/mouse_dataset
   
 
Intutive Datasets (not-labeled)
------------------------------------
   
.. toctree::
   :maxdepth: 2
   
   examples/chameleon_ds1_dataset
   examples/chameleon_ds2_dataset
   examples/chameleon_ds3_dataset
   examples/chameleon_ds4_dataset
   examples/clusterable_dataset
   
   
Parameteric Synthestic Datasets
----------------------------------
   
.. toctree::
   :maxdepth: 2
   
   examples/make_blobs_dataset
   examples/make_moons_dataset
   examples/make_circles_dataset
   examples/make_classification_dataset
   examples/make_gaussian_quantiles_dataset
   