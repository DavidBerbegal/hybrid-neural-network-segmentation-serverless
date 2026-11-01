FROM public.ecr.aws/lambda/python:3.11

RUN yum install -y git \
    && yum clean all \
    && rm -rf /var/cache/yum

COPY requirements.txt ./

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && PIP_ONLY_BINARY=:all: pip3 install --no-cache-dir -r requirements.txt

COPY app.py instrumentacion.py ./

CMD ["instrumentacion.handler"]