#!/bin/bash
# Animals with Attributes Dataset, http://attributes.kyb.tuebingen.mpg.de
# Train all attribute classifiers for fixed split and regularizer 
# (C) 2009 Christoph Lampert <chl@tuebingen.mpg.de>

SPLIT=0
C=10.

for A in `seq 1 85` ;
do
./attributes.py $A $SPLIT $C
done
