..
   Copyright 2021 AI Singapore. All rights reserved.

   Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
   the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
   an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
   specific language governing permissions and limitations under the License.


Analyzer
========

This module serves as the main interface of a `html`-like page that collates all available features. Each feature component 
is presented as a `tab` which allow user to navigate to feature component of their choice. The content of each feature component differs 
from one component to another serving different task and displayed differently depending if the analysis is ``regression`` or ``classification``.


.. autoclass:: rarity.GapAnalyzer
    :members: _layout, run
