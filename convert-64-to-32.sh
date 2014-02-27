#! /bin/bash

#  This is a quick and dirty way to convert spike from 64 bit floats to 32 bit floats
sed -i -e 's/float64/float32/g' $(find . -name '*.py')
