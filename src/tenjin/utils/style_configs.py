import dash_bootstrap_components as dbc
import dash_html_components as html


INSTRUCTION_TEXT_SHARED = 'Click and drag on the graph to select the range of data points to inspect feature values.'
INSTRUCTION_TEXT_REG = 'To reset back to default settings, hover over icons on the top right of the graph and click "Reset axes" icon.'
WARNING_TEXT = 'To inspect new range of datapoints in different graph, please first reset the earlier selection by clicking "Reset axes" icon ' \
                'at the top right corner of the graph.'

DEFAULT_HEADER_STYLE = {'fontWeight': 'bold', 'color': 'white', 'backgroundColor': '#7e746d', 'border': '1px solid white', 'height': '45px'}
DEFAULT_TITLE_STYLE = {'visibility': 'visible'}
DEFAULT_PLOT_NAME_STYLE = {'visibility': 'visible'}

DEFAULT_RANGE_SELECTION_TEXT_REG = "Enter range of data to compare distribution ( default slicing - last 20% of dataset ) :"
DEFAULT_RANGE_SELECTION_TEXT_CLS = "Enter range of data to compare distribution ( default slicing - full range ):"

DEFAULT_SLIDER_RANGE = {n: str(n) for n in range(1, 11)}


def default_header_style():
    DEFAULT_HEADER_STYLE['visibility'] = 'visible'
    return DEFAULT_HEADER_STYLE


def hidden_title_style():
    DEFAULT_TITLE_STYLE['visibility'] = 'hidden'
    return DEFAULT_TITLE_STYLE


def hidden_plot_name_style():
    DEFAULT_PLOT_NAME_STYLE['visibility'] = 'hidden'
    return DEFAULT_PLOT_NAME_STYLE


def collapse_header_style():
    DEFAULT_HEADER_STYLE['border'] = 'none'
    DEFAULT_HEADER_STYLE['visibility'] = 'collapse'
    return DEFAULT_HEADER_STYLE


def input_range_subnote(df):
    return f'Available dataset range : from 0 to max at {len(df)}'


def dummy_alert():
    alert_obj = dbc.Alert(color="light", style={'visibility': 'hidden'})
    return alert_obj


def activate_alert():
    alert_obj = dbc.Alert(INSTRUCTION_TEXT_REG, color="secondary", className='alert__note-reg')
    return alert_obj


def activate_cluster_error_alert(label_class):
    err_message = f'Miss prediction data points are not sufficient for auto-clustering in < Class {label_class} >. ' \
                    'Minimum number of datapoint for auto-clustering is 8 datapoints per class per model'
    alert_obj = dbc.Alert(err_message, color='warning', dismissable=True, is_open=True)
    return alert_obj


def activate_range_input_error_alert():
    err_message = 'Invalid data range detected. ' \
                    'Please input a valid data range format ' \
                    '(  example:   start_idx:stop_idx   =>   200:1000   or   25%:75%    with   start_idx  <  stop_idx  )'
    alert_obj = dbc.Alert(html.Pre(err_message, className='html_Pre__alert-message'), color='warning', dismissable=True, is_open=True)
    return alert_obj


def activate_invalid_limit_alert(df):
    err_message = f'Invalid data range detected. Allowable index range for slicing :   0 to {len(df)}   or   0% to 100%'
    alert_obj = dbc.Alert(html.Pre(err_message, className='html_Pre__alert-message'), color='warning', dismissable=True, is_open=True)
    return alert_obj


def activate_incomplete_range_entry_alert():
    err_message = 'Invalid data range detected. Please enter a complete range ' \
                    '(  example:   start_idx:stop_idx   =>   200:1000   or   25%:75%    with   start_idx  <  stop_idx  )'
    alert_obj = dbc.Alert(html.Pre(err_message, className='html_Pre__alert-message'), color='warning', dismissable=True, is_open=True)
    return alert_obj


def no_error_alert():
    alert_obj = dbc.Alert('', color='warning', dismissable=True, is_open=False)
    return alert_obj
