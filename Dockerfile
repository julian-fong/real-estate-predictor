FROM python:3.11.1

# Set the working directory
WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    bash \
    && pip3 install --upgrade pip

RUN pip install -e .

# CMD ["python", "real_estate_predictor/app.py"]

EXPOSE 8000

CMD ["uvicorn", "real_estate_predictor.app:app", "--host", "0.0.0.0", "--port", "8000"]


#commands
# docker build -t test_rep_1 .
# docker run -e REPLIERS_KEY=... -d -p 8000:8000 test_rep_1

# to access bash in the container
# docker exec -it <container_id_or_name> bash