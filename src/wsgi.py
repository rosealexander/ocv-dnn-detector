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
import detector as myapp

# If you need to change the algorithm file, modify the "myapp" value in the code above.
app = myapp.app
