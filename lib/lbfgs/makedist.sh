#!/bin/bash
# $Id$

if [ $# -ne 1 ]; then
  echo "USAGE: $0 <tar-ball>"
  exit 1
fi

tar cvzf $1 \
    README \
    COPYING \
    include/lbfgs.h \
    src/arithmetic_ansi.h \
    src/arithmetic_sse_double.h \
    src/arithmetic_sse_float.h \
    src/lbfgs.c \
    test/main.c \
    test/build.sh \
    test/lbfgs.vcproj \
    test/lbfgs.sln
