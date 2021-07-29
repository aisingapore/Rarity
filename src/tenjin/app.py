import dash

# app = dash.Dash(__name__, )
app = dash.Dash(__name__, 
                meta_tags=[{'name': 'viewport', 
                            'content': 'width=device-width, initial-scale=1.0'}], 
                suppress_callback_exceptions=True)
server = app.server
