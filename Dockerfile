FROM python:3.11.1

# Set the working directory
WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    && pip3 install --upgrade pip

RUN pip install -r requirements.txt

CMD ["python", "app.py"]