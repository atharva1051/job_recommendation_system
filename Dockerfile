FROM python:3.11.2

WORKDIR /jrs

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install output

COPY . .

CMD [ "python", "app.py"]

EXPOSE 3333