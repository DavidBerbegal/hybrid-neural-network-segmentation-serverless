FROM public.ecr.aws/lambda/python:3.11

# Necesario para instalar awsgi desde GitHub
RUN yum install -y git && yum clean all && rm -rf /var/cache/yum

COPY requirements.txt ./
# Forzar wheels precompilados (sin compilar nada)
RUN python3 -m pip install --upgrade pip setuptools wheel \
 && PIP_ONLY_BINARY=:all: pip3 install --no-cache-dir -r requirements.txt

COPY app.py ./
CMD ["app.handler"]
