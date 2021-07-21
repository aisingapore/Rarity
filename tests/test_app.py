from dash.testing.application_runners import import_app
from tenjin.data_loader import CSVDataLoader
from tenjin import GapAnalyzer


def test_GapAnalyzer_single_modal_reg(dash_duo, csv_data_loader_single_modal_reg):
    analyzer = GapAnalyzer(csv_data_loader_single_modal_reg, 'TestCase Analysis 1')
    app = import_app('src.tenjin.app')
    app.layout = analyzer._layout()
    dash_duo.start_server(app)
    assert dash_duo.find_element("#tabs-feature-page")
    assert dash_duo.find_element("#feature-page")
 
def test_GapAnalyzer_bimodal_reg(dash_duo, csv_data_loader_bimodal_reg):
    analyzer = GapAnalyzer(csv_data_loader_bimodal_reg, 'TestCase Analysis 2')
    app = import_app('src.tenjin.app')
    app.layout = analyzer._layout()
    dash_duo.start_server(app)
    assert dash_duo.find_element("#tabs-feature-page")
    assert dash_duo.find_element("#feature-page")

def test_GapAnalyzer_single_modal_cls(dash_duo, csv_data_loader_single_modal_cls):
    analyzer = GapAnalyzer(csv_data_loader_single_modal_cls, 'TestCase Analysis 3')
    app = import_app('src.tenjin.app')
    app.layout = analyzer._layout()
    dash_duo.start_server(app)
    assert dash_duo.find_element("#tabs-feature-page")
    assert dash_duo.find_element("#feature-page")
 
def test_GapAnalyzer_bimodal_cls(dash_duo, csv_data_loader_bimodal_cls):
    analyzer = GapAnalyzer(csv_data_loader_bimodal_cls, 'TestCase Analysis 4')
    app = import_app('src.tenjin.app')
    app.layout = analyzer._layout()
    dash_duo.start_server(app)
    assert dash_duo.find_element("#tabs-feature-page")
    assert dash_duo.find_element("#feature-page")