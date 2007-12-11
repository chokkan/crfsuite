#!/bin/bash

if [ $# -ne 1 ]; then
  echo "USAGE: $0 <tar-ball>"
  exit 1
fi

tar cvzf $1 \
    COPYING \
    include/cqdb.h \
    src/cqdb.c \
    src/lookup3.c \
    src/main.c
