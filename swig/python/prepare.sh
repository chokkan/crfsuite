#!/bin/sh

ln -s ../crfsuite.cpp
ln -s ../export.i

if [ "$1" = "--swig" ];
then
    swig -c++ -python -I/home/users/okazaki/local/include -o export_wrap.cpp export.i
fi
