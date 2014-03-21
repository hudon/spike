#! /bin/bash

#  This is a quick and dirty way to convert spike from 32 bit floats to 64 bit floats
sed -i -e 's/float32/float64/g' $(find . -name '*.py')
