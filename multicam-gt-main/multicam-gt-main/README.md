

# EPFL MultiCam-GT v2
Updated version of EPFL MultiCam-GT Tool, a multiple view object annotation tool, now with:

- **Elevation model**: Improved ground plane representation through mesh-based elevation modeling, for object annotation in varying terrains.
- **Enhanced Object Transformation**: The new version includes object representation using dimensions and spatial orientation.
- **Tracklet Merging**: Functionality for merging tracklets, providing more streamlined object tracking.
- **Trajectory Visualization**: Features new tools for visualizing trajectories, enhancing understanding and analysis of object movement.
- **Database Integration**: All transformation and tracking data is now stored in the database, enabling more efficient data management and retrieval.

## Usage
Define the DATABASES variable in the settings.py file to point to your database. The application is configured to use a PostgreSQL database. You can use the following configuration:
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'invision',
        'USER': 'invision',
        'PASSWORD': '',
        'HOST': 'localhost',
        'PORT': '',
    }h
}
```
You can load the INVISION data into the database by running the following command:
```bash
python manage.py shell < invision_data_process.py
```
By default, the worker name will be 'INVISION'. You can change it by editing the invision_data_process.py file.
You can then run the following commands to run the application:
```bash
conda env create -n ENVNAME --file ENV.yml
pip install -r requirements.txt
initdb -D invision   
chmod -R 700 invision
pg_ctl -D invision -l logfile start
createuser invision
createdb --owner=invision invision

python manage.py migrate
python manage.py runserver 0.0.0.0:4444
```
You can now access the application at http://localhost:4444



### Acknowledgements
This project was based on the original [MultiCam-GT Tool](https://github.com/cvlab-epfl/multicam-gt) developed by the Computer Vision Laboratory at EPFL.
