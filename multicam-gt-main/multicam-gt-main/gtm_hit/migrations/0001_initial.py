# Generated by Django 4.1.7 on 2023-03-31 17:14

import django.contrib.postgres.fields
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="View",
            fields=[
                (
                    "view_id",
                    models.IntegerField(
                        primary_key=True, serialize=False, verbose_name="View ID"
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="Worker",
            fields=[
                (
                    "workerID",
                    models.TextField(max_length=40, primary_key=True, serialize=False),
                ),
                ("frameNB", models.IntegerField(default=-1)),
                ("frame_labeled", models.PositiveSmallIntegerField(default=0)),
                ("finished", models.BooleanField(default=False)),
                ("state", models.IntegerField(default=-1)),
                ("tuto", models.BooleanField(default=False)),
                ("time_list", models.TextField(default="")),
            ],
        ),
        migrations.CreateModel(
            name="ValidationCode",
            fields=[
                ("validationCode", models.TextField(primary_key=True, serialize=False)),
                (
                    "worker",
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.CASCADE, to="gtm_hit.worker"
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="SingleViewFrame",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("frame_id", models.IntegerField(verbose_name="SingleView ID")),
                ("timestamp", models.DateTimeField(default=django.utils.timezone.now)),
                (
                    "view",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="gtm_hit.view"
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="Person",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "person_id",
                    models.IntegerField(unique=True, verbose_name="PersonID"),
                ),
                ("annotation_complete", models.BooleanField(default=False)),
                (
                    "worker",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="gtm_hit.worker"
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="MultiViewFrame",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("frame_id", models.IntegerField(verbose_name="MultiView ID")),
                ("timestamp", models.DateTimeField(default=django.utils.timezone.now)),
                ("undistorted", models.BooleanField(default=False)),
                (
                    "worker",
                    models.ForeignKey(
                        default="IVAN",
                        on_delete=django.db.models.deletion.CASCADE,
                        to="gtm_hit.worker",
                    ),
                ),
            ],
            options={
                "unique_together": {("frame_id", "undistorted", "worker")},
            },
        ),
        migrations.CreateModel(
            name="Annotation",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("creation_method", models.TextField(default="existing_annotation")),
                ("validated", models.BooleanField(default=True)),
                ("rectangle_id", models.CharField(max_length=100)),
                ("rotation_theta", models.FloatField()),
                ("Xw", models.FloatField(verbose_name="X World Coordinate")),
                ("Yw", models.FloatField(verbose_name="Y World Coordinate")),
                ("Zw", models.FloatField(verbose_name="Z World Coordinate")),
                ("object_size_x", models.FloatField()),
                ("object_size_y", models.FloatField()),
                ("object_size_z", models.FloatField()),
                (
                    "frame",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="gtm_hit.multiviewframe",
                    ),
                ),
                (
                    "person",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="gtm_hit.person"
                    ),
                ),
            ],
            options={
                "unique_together": {("frame", "person")},
            },
        ),
        migrations.CreateModel(
            name="Annotation2DView",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("x1", models.FloatField(null=True)),
                ("y1", models.FloatField(null=True)),
                ("x2", models.FloatField(null=True)),
                ("y2", models.FloatField(null=True)),
                (
                    "cuboid_points",
                    django.contrib.postgres.fields.ArrayField(
                        base_field=models.FloatField(), null=True, size=20
                    ),
                ),
                (
                    "annotation",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="twod_views",
                        to="gtm_hit.annotation",
                    ),
                ),
                (
                    "view",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="gtm_hit.view"
                    ),
                ),
            ],
            options={
                "unique_together": {("view", "annotation")},
            },
        ),
    ]
