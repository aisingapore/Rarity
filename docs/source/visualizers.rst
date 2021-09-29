Visualizers
===========

Modules under ``Visualizers`` are mainly responsible for all interactive graphing works. It takes in direct inputs or post-processed inputs from interpreters 
and generate various plots using ``plotly`` frameworks. The types of graph generated depend on the feature component which is linked to specific task.


Viz - General Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: rarity.visualizers.general_metrics
    :members: plot_confusion_matrix, plot_classification_report, plot_roc_curve, plot_precisionRecall_curve, plot_prediction_vs_actual, plot_prediction_offset_overview, plot_std_error_metrics


Viz - Miss Predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: rarity.visualizers.miss_predictions
    :members: plot_probabilities_spread_pattern, plot_simple_probs_spread_overview, plot_prediction_offset_overview, 


Viz - Loss Clusters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: rarity.visualizers.loss_clusters
    :members: plot_offset_clusters, plot_logloss_clusters, plot_optimum_cluster_via_elbow_method


Viz - xFeature Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: rarity.visualizers.xfeature_distribution
    :members: plot_distribution_by_specific_feature, plot_distribution_by_kl_div_ranking


Viz - Shared Viz Component
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: rarity.visualizers.shared_viz_component
    :members: reponsive_table_to_filtered_datapoints, reponsive_table_to_filtered_datapoints_similaritiesCF