# -*- coding: utf-8 -*-
import numpy
import pandas


def handler(event, context):

	print(numpy.zeros(1))
	print(numpy.ones(1))
	s = pandas.Series(numpy.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
	print(s)
	result = {}
	
	return result
