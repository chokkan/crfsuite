# $Id:$

lib_LTLIBRARIES = liblbfgs.la

liblbfgs_la_SOURCES = \
	src/arithmetic_ansi.h \
	src/arithmetic_sse_double.h \
	src/arithmetic_sse_float.h \
	include/lbfgs.h \
	src/lbfgs.c

liblbfgs_la_LDFLAGS = \
	-no-undefined \
	-release @VERSION@

#liblbfgs_la_LIBADD = 

AM_CFLAGS = @CFLAGS@
INCLUDES = @INCLUDES@
