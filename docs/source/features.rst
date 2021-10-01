..
   Copyright 2021 AI Singapore. All rights reserved.

   Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
   the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
   an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
   specific language governing permissions and limitations under the License.


Features
=========

Modules under ``Features`` act as core integrator to link up inputs from ``interpreters`` and outputs interactive graphs via ``visualizers``. Major styled 
components built with ``dash`` are defined at this stage and customized accordingly in respective feature modules depending on the task it serves. All responsive 
parameters and callbacks managements are handled in this stage as well.


Feat - General Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rarity.features.GeneralMetrics

.. automodule:: rarity.features.feat_general_metrics
    :members: fig_confusion_matrix, fig_classification_report, fig_roc_curve, fig_precisionRecall_curve, fig_prediction_actual_comparison, fig_prediction_offset_overview, fig_standard_error_metrics


Feat - Miss Predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rarity.features.MissPredictions

.. automodule:: rarity.features.feat_miss_predictions
    :members: fig_plot_prediction_offset_overview, fig_probabilities_spread_pattern, table_with_relayout_datapoints, convert_relayout_data_to_df_reg, convert_relayout_data_to_df_cls, 


Feat - Loss Clusters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rarity.features.LossClusters

.. automodule:: rarity.features.feat_loss_clusters
    :members: fig_plot_offset_clusters_reg, fig_plot_logloss_clusters_cls, table_with_relayout_datapoints, convert_cluster_relayout_data_to_df_reg, convert_cluster_relayout_data_to_df_cls, 


Feat - xFeature Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rarity.features.FeatureDistribution

.. automodule:: rarity.features.feat_feature_distribution
    :members: fig_plot_distribution_by_kl_div_ranking, fig_plot_distribution_by_specific_feature


Feat - Similarities (+CounterFactuals)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rarity.features.SimilaritiesCF

.. automodule:: rarity.features.feat_similarities_counter_factuals
    :members: generate_similarities, generate_counterfactuals
