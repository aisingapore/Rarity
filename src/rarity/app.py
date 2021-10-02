# Copyright 2021 AI Singapore. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dash
import os


assets_path = os.path.join(os.getcwd(), 'assets')
app = dash.Dash(__name__, 
                meta_tags=[{'name': 'viewport', 
                            'content': 'width=device-width, initial-scale=1.0'}],
                suppress_callback_exceptions=True, assets_url_path=assets_path)

server = app.server
