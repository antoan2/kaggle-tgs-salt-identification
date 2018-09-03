FROM floydhub/pytorch:0.4.0-gpu.cuda9cudnn7-py3.33

RUN mkdir /opt/learning-tool
WORKDIR /opt/learning-tool

COPY src/requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY src .
