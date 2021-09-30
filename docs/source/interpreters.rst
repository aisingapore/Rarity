..
   Copyright 2021 AI Singapore. All rights reserved.

   Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
   the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
   an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
   specific language governing permissions and limitations under the License.


Interpreters
============

Modules under ``Interperters`` are used to tap-out all necesssry inputs transformation from ``dataloader``. 
The level of transformation depends on the feature it is used for. The outputs from interpreters can either be the direct inputs to ``visualizers`` 
or involves secondary processing before parsing to ``visualizer`` for graphing.

Int - General Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rarity.interpreters.structured_data.IntGeneralMetrics


Int - Miss Predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rarity.interpreters.structured_data.IntMissPredictions


Int - Loss Clusters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rarity.interpreters.structured_data.IntLossClusterer
    :members: extract_misspredictions, xform


Int - xFeature Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rarity.interpreters.structured_data.IntFeatureDistribution
    :members: _get_df_sliced, _get_single_feature_df_with_binning, _get_probabilities_by_bin_group, _get_df_feature_with_pred_state_cls, _get_probabilities_by_feature, _generate_kl_div_info_base, xform


Int - Similarities (+CounterFactuals)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rarity.interpreters.structured_data.IntSimilaritiesCounterFactuals
    :members: _get_categorical_features, _label_encode_categorical_features, _apply_standard_scale, _get_ranking_and_distance_metrics, xform
