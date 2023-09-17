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
# This file implements the scoring service shell. You don't necessarily need to modify it for various
# algorithms. It starts nginx and gunicorn with the correct configurations and then simply waits until
# gunicorn exits.
#
# The flask server is specified to be the app object in wsgi.py
#
# We set the following parameters:
#
# Parameter                         Environment Variable              Default Value
# ---------                         --------------------              -------------
# number of workers                 MODEL_SERVER_WORKERS              the number of CPU cores
# timeout                           MODEL_SERVER_TIMEOUT              60 seconds
# log level                         LOG_LEVEL                         WARNING
# enable enhanced logging           DEBUG                             None

from __future__ import print_function

import multiprocessing
import os
import signal
import subprocess
import sys
import logging
import psutil


# Read environment variables
LOG_LEVEL = os.getenv('LOG_LEVEL', 'WARNING').upper()
DEBUG = os.getenv('DEBUG', False)

# Configure logging
logger = logging.getLogger(__name__)

log_handler = logging.StreamHandler()
log_timestamp_format = "%Y-%m-%d %H:%M:%S %z"
log_formatter = logging.Formatter(fmt='[%(asctime)s] [%(process)d] [%(levelname)s] %(message)s', datefmt=log_timestamp_format)
log_formatter = log_handler.setFormatter(log_formatter)
logger.addHandler(log_handler)

if LOG_LEVEL == 'DEBUG' or DEBUG:
    LOG_LEVEL = 'DEBUG'
    logger.setLevel(level=logging.DEBUG)
elif LOG_LEVEL == 'INFO':
    logger.setLevel(level=logging.INFO)
elif LOG_LEVEL == 'ERROR':
    logger.setLevel(level=logging.ERROR)
elif LOG_LEVEL == 'CRITICAL':
    logger.setLevel(level=logging.CRITICAL)
else:
    logger.setLevel(level=logging.WARNING)

# set server configs
cpu_count = multiprocessing.cpu_count()
model_server_timeout = os.environ.get('MODEL_SERVER_TIMEOUT', 60)
model_server_workers = int(os.environ.get('MODEL_SERVER_WORKERS', cpu_count))


def sigterm_handler(nginx_pid, gunicorn_pid):
    try:
        os.kill(nginx_pid, signal.SIGQUIT)
    except OSError:
        pass
    try:
        os.kill(gunicorn_pid, signal.SIGTERM)
    except OSError:
        pass

    sys.exit(0)


def log_server_information():
    memory_info = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent()
    logger.debug("Memory Usage: {:.2f} GB used / {:.2f} GB total ({:.2f}% used)".format(memory_info.used / (1024 ** 3), memory_info.total / (1024 ** 3), memory_info.percent))
    logger.debug("CPU Usage: {:.2f}%".format(cpu_percent))


def start_server():
    if DEBUG:
        logger.info('Starting the server with debugging and {} workers.'.format(model_server_workers))
    else:
        logger.info('Starting the server with {} workers.'.format(model_server_workers))

    # link the log streams to stdout/err, so they will be logged to the container logs
    subprocess.check_call(['ln', '-sf', '/dev/stdout', '/var/log/nginx/access.log'])
    subprocess.check_call(['ln', '-sf', '/dev/stderr', '/var/log/nginx/error.log'])

    nginx = subprocess.Popen(['nginx', '-c', '/opt/ml/code/nginx.conf'])
    gunicorn = subprocess.Popen(['gunicorn',
                                 '--timeout', str(model_server_timeout),
                                 '-k', 'gevent',
                                 '-b', 'unix:/tmp/gunicorn.sock',
                                 '-w', str(model_server_workers),
                                 'wsgi:app'])

    signal.signal(signal.SIGTERM, lambda a, b: sigterm_handler(nginx.pid, gunicorn.pid))

    # If either subprocess exits, so do we.
    pids = set([nginx.pid, gunicorn.pid])
    while True:
        if DEBUG:
            log_server_information()
        pid, _ = os.wait()
        if pid in pids:
            break

    sigterm_handler(nginx.pid, gunicorn.pid)
    logger.info('Inference server exiting')


# The main routine just invokes the start function.
if __name__ == '__main__':
    start_server()
