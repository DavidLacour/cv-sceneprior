FROM debian:11

RUN apt update && apt install -y python3 python3-pip python3-venv python3-opencv python3-psycopg2

RUN mkdir /var/www && chown www-data.www-data /var/www
USER www-data
WORKDIR /var/www

RUN python3 -m venv venv && . venv/bin/activate
ENV VIRTUAL_ENV /var/www/venv
ENV PATH /var/www/venv/bin:$PATH

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY marker marker
COPY gtm_hit gtm_hit
COPY home home
COPY manage.py ./
COPY gtmarker gtmarker
