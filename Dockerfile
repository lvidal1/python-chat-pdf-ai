FROM python:3.8-slim
# Or any preferred Python version.

RUN apt-get update && apt-get upgrade -y && apt-get install gcc -y
RUN pip install --upgrade pip
RUN mkdir /usr/src/app
WORKDIR /usr/src/app
COPY ./app/requirements.txt .
RUN pip install -r requirements.txt
ENV PYTHONUNBUFFERED 1
COPY ./app .

CMD [ "streamlit", "run", "app.py", "--server.port","8500" ]