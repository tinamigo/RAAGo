# -*- coding: utf-8 -*-
# Generated by Django 1.9.6 on 2016-06-30 01:00
from __future__ import unicode_literals

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('events', '0002_event_description'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='event',
            options={'ordering': ('end_date', 'start_date', 'pk')},
        ),
    ]
