# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2017-04-25 00:13
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='data',
            name='sick',
            field=models.BooleanField(),
        ),
    ]
