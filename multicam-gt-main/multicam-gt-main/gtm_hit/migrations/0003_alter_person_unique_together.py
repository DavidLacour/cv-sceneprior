# Generated by Django 4.1.7 on 2023-03-31 17:48

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("gtm_hit", "0002_alter_person_person_id"),
    ]

    operations = [
        migrations.AlterUniqueTogether(
            name="person",
            unique_together={("person_id", "worker")},
        ),
    ]
