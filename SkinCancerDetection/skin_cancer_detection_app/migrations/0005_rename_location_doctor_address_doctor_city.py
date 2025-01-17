# Generated by Django 5.1.3 on 2024-11-30 13:19

import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('skin_cancer_detection_app', '0004_remove_doctor_profile_picture'),
    ]

    operations = [
        migrations.RenameField(
            model_name='doctor',
            old_name='location',
            new_name='address',
        ),
        migrations.AddField(
            model_name='doctor',
            name='city',
            field=models.CharField(default=django.utils.timezone.now, max_length=100),
            preserve_default=False,
        ),
    ]
