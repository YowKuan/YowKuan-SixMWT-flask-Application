FROM python:3.7

WORKDIR /app

ADD . /app

RUN apt install libpq-dev

RUN pip install -r requirements_cpu.txt
Run flask db init
Run flask db migrate
Run flask db upgrade

CMD python wsgi.py