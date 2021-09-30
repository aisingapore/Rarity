..
   Copyright 2021 AI Singapore. All rights reserved.

   Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
   the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
   an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
   specific language governing permissions and limitations under the License.


Installation
=============

.. raw:: html

    <style> .red {color:#aa0060; font-weight:bold; font-size:16px} </style>

.. role:: red

**Rarity** has been tested on Python 3.8+. It is advisable to create a :red:`Virtual Environment` to use along with **Rarity**.

For details guide on how to create virtual environment, you can refer to this `user guide <https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/>`_.
After creation of the virtual environment, activate it and proceed with one of the following steps.


From PyPI
---------

.. code:: python

    pip install rarity


From Source
-----------

.. code:: python

    git clone https://github.com/aimakerspace/rarity.git
    cd rarity
    pip install -e .
