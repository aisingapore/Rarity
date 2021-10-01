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

from dash.testing.application_runners import import_app
from rarity import GapAnalyzer


def test_GapAnalyzer_single_modal_reg(dash_duo, csv_loader_single_modal_reg):
    analyzer = GapAnalyzer(csv_loader_single_modal_reg, 'TestCase Analysis 1')
    app = import_app('rarity.app')
    app.layout = analyzer._layout()
    dash_duo.start_server(app)
    assert dash_duo.find_element("#tabs-feature-page")
    assert dash_duo.find_element("#feature-page")


def test_GapAnalyzer_bimodal_reg(dash_duo, csv_loader_bimodal_reg):
    analyzer = GapAnalyzer(csv_loader_bimodal_reg, 'TestCase Analysis 2')
    app = import_app('rarity.app')
    app.layout = analyzer._layout()
    dash_duo.start_server(app)
    assert dash_duo.find_element("#tabs-feature-page")
    assert dash_duo.find_element("#feature-page")


def test_GapAnalyzer_single_modal_cls(dash_duo, csv_loader_single_modal_cls):
    analyzer = GapAnalyzer(csv_loader_single_modal_cls, 'TestCase Analysis 3')
    app = import_app('rarity.app')
    app.layout = analyzer._layout()
    dash_duo.start_server(app)
    assert dash_duo.find_element("#tabs-feature-page")
    assert dash_duo.find_element("#feature-page")


def test_GapAnalyzer_bimodal_cls(dash_duo, csv_loader_bimodal_cls):
    analyzer = GapAnalyzer(csv_loader_bimodal_cls, 'TestCase Analysis 4')
    app = import_app('rarity.app')
    app.layout = analyzer._layout()
    dash_duo.start_server(app)
    assert dash_duo.find_element("#tabs-feature-page")
    assert dash_duo.find_element("#feature-page")
