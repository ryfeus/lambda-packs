#!/usr/bin/env python
import numpy
from shapely.geometry import Point

def handler(event, context):
	patch = Point(0.0, 0.0).buffer(10.0)
	print(patch)
	print(patch.area)
	return patch.area