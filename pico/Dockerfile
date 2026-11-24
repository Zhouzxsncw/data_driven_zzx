FROM aloha-models:cuda12.8
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy=
ENV NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
COPY requirements.txt /tmp/pico-requirements.txt
COPY wheels/ /tmp/pico-wheels/
RUN python3 -m pip --isolated install --no-index --find-links=/tmp/pico-wheels --target /opt/pico-packages --no-cache-dir --only-binary=:all: -r /tmp/pico-requirements.txt
ENV PYTHONPATH=/opt/pico-packages
