Dataloaders
============

The ``dataloaders`` are used to collate raw data of ``xFeatures``, ``yTrue`` and ``yPred`` and standardize them into input formats and methods
that can be called upon at various stages for further processing to serve specific tasks related to feature components. This is the module that 
user can call upon to load their data and trigger a series of auto-processing to generate the gap analysis report and spin up the ``dash`` web application.

CSVDataLoader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tenjin.data_loader.CSVDataLoader


DataframeLoader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tenjin.data_loader.DataframeLoader
