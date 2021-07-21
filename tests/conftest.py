# add this in the conftest.py under tests folder
from selenium.webdriver.chrome.options import Options


def pytest_setup_options():
    options = Options()

    # added mainly for integration test in gitlab-ci to resolve 
    # (unknown error: DevToolsActivePort file doesn't exist)
    # (The process started from chrome location /usr/bin/google-chrome is no longer running, 
    # so ChromeDriver is assuming that Chrome has crashed.)
    # solution reference => https://github.com/plotly/dash/issues/1420
    options.add_argument('--no-sandbox')
    return options
