conda install postgresql  
initdb -D invision   
chmod -R 700 invision
pg_ctl -D invision -l logfile start
createuser invision
createdb --owner=invision invision

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'invision',
        'USER': 'invision',
        'PASSWORD': '',
        'HOST': 'localhost',
        'PORT': '',
    }
}

python manage.py migrate
python manage.py runserver 0.0.0.0:4444