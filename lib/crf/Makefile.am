# $Id:$

lib_LTLIBRARIES = libcrf.la

EXTRA_DIST = \
	crf.vcproj 

libcrf_la_SOURCES = \
	src/dictionary.c \
	src/logging.c \
	src/logging.h \
	src/params.c \
	src/params.h \
	src/quark.c \
	src/quark.h \
	src/rumavl.c \
	src/rumavl.h \
	src/mt19937ar.c \
	src/mt19937ar.h \
	src/crf1m.c \
	src/crf1m.h \
	src/crf1m_context.c \
	src/crf1m_feature.c \
	src/crf1m_learn.c \
	src/crf1m_learn_lbfgs.c \
	src/crf1m_learn_sgd.c \
	src/crf1m_model.c \
	src/crf1m_tag.c \
	include/crf.h \
	src/crf.c


libcrf_la_CFLAGS = -I./include -I$(top_builddir)/lib/cqdb/include 

libcrf_la_LDFLAGS = \
	-no-undefined \
	-release @VERSION@

libcrf_la_LIBADD = \
	$(top_builddir)/lib/cqdb/libcqdb.la

AM_CFLAGS = @CFLAGS@
INCLUDES = @INCLUDES@
