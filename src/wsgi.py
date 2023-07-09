#
# Copyright 2021-2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Modifications copyright (C) 2023 Alexander Rose
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This is a simple wrapper for the Gunicorn server to locate your Flask application.
import os
import sys
import logging

import detector as myapp

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from version import __version__

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log current version number
logger.info('\n * Version \033[1;36m' + __version__ + '\033[0m')

# If you need to change the algorithm file, modify the "detector" value in the code above.
app = myapp.app
