# $Id:$

lib_LTLIBRARIES = libcrf.la

EXTRA_DIST = \
	crf.vcproj 

libcrfincludedir = $(includedir)
libcrfinclude_HEADERS = \
	include/crfsuite.h

libcrf_la_SOURCES = \
	src/dictionary.c \
	src/logging.c \
	src/logging.h \
	src/mt19937ar.c \
	src/mt19937ar.h \
	src/params.c \
	src/params.h \
	src/quark.c \
	src/quark.h \
	src/rumavl.c \
	src/rumavl.h \
	src/vecmath.h \
	src/crfsuite_internal.h \
	src/train.c \
	src/train_averaged_perceptron.c \
	src/train_lbfgs.c \
	src/crf1d.c \
	src/crf1d.h \
	src/crf1d_context.c \
	src/crf1d_feature.c \
	src/crf1d_learn.c \
	src/crf1d_model.c \
	src/crf1d_tag.c \
	src/crf.c

libcrf_la_CFLAGS = -I./include -I$(top_builddir)/lib/cqdb/include 

libcrf_la_LDFLAGS = \
	-no-undefined \
	-release @VERSION@

libcrf_la_LIBADD = \
	$(top_builddir)/lib/cqdb/libcqdb.la

AM_CFLAGS = @CFLAGS@
INCLUDES = @INCLUDES@
