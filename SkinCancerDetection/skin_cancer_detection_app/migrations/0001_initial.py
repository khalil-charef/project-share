# Generated by Django 5.1.3 on 2024-11-29 20:04

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Doctor',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('specialty', models.CharField(max_length=100)),
                ('location', models.CharField(max_length=200)),
                ('profile_picture', models.URLField(default='https://via.placeholder.com/100')),
                ('contact_url', models.URLField(blank=True, null=True)),
                ('appointment_url', models.URLField(blank=True, null=True)),
            ],
        ),
    ]
