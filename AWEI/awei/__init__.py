__license__ = "Apache-2.0 License"
__version__ = "0.0.1"
__author__ = "Carolin Benjamins"


import datetime

name = "AWEI"
package_name = "awei"
author = __author__

author_email = "c.benjamins@ai.uni-hannover.de"
description = "Adaptive Weighted Expected Improvement"
url = "https://www.automl.org/"
project_urls = {
    # "Documentation": "https://carl.readthedocs.io/en/latest/",
    # "Source Code": "https://github.com/https://github.com/automl/CARL",
}
copyright = f"""
    Copyright {datetime.date.today().strftime('%Y')}, AutoML.org Freiburg-Hannover
"""
version = __version__

import awei.utils.config_setup
