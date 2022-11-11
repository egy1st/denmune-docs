Welcome to DenMune\'s documentation!
====================================

DenMune Clustering Algorithm

:   A clustering algorithm that can find clusters of arbitrary size,
    shapes and densities in two-dimensions. Higher dimensions are first
    reduced to 2-D using the t-sne. The algorithm relies on a single
    parameter K (the number of nearest neighbors). The results show the
    superiority of DenMune. Enjoy the simplicty but the power of
    DenMune.

::: {.note}
::: {.title}
Note
:::

This documentation associated with the paper \"DenMune: Density peak
based clustering using mutual nearest neighbors\"

DOI: <https://doi.org/10.1016/j.patcog.2020.107589>

Source code is maintained at
<https://github.com/scikit-learn-contrib/denmune-clustering-algorithm>
:::

User Guide / Tutorials
----------------------

::: {.toctree maxdepth="3"}
README
:::

Examples
--------

::: {.toctree maxdepth="2"}
examples/iris\_dataset examples/chameleon\_datasets
examples/2D\_shapes\_datasets examples/MNIST\_dataset
:::

Characteristics
---------------

::: {.toctree maxdepth="2"}
characteristics/noise\_detection characteristics/clustering\_propagation
characteristics/clustering\_propagation\_snapshots
characteristics/scalability\_and\_speed
characteristics/stability\_vs\_knn characteristics/k\_nearest\_evolution
:::

Participate in Competitions
---------------------------

::: {.toctree maxdepth="2"}
kaggle/validation kaggle/training\_MNIST
kaggle/Get\_97\_by\_training\_MNIST\_dataset
:::
