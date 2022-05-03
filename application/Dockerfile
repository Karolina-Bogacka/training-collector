FROM python:3.8.3-slim-buster
WORKDIR /code
COPY requirements.txt /code/requirements.txt
RUN apt-get update && apt-get -y install iproute2
RUN ln -s /usr/lib/tc /lib/tc
RUN python -m pip install --upgrade pip
RUN pip3 install pillow
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . /code/application
ENV PYTHONPATH "${PYTHONPATH}:/code/application"
EXPOSE 8000
ENTRYPOINT ["python3", "./application/main.py"]
