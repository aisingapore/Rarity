..
   Copyright 2021 AI Singapore. All rights reserved.

   Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
   the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
   an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
   specific language governing permissions and limitations under the License.


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