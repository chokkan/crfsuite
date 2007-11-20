# $Id:$

bin_PROGRAMS = crf
man_MANS = crf.1
EXTRA_DIST = ${man_MANS}

crf_SOURCES = \
	iwa.h \
	iwa.c \
	option.h \
	option.c \
	readdata.h \
	reader.c \
	learn.c \
	tag.c \
	dump.c \
	main.c

#crf_CPPFLAGS =

AM_CFLAGS = @CFLAGS@
INCLUDES = @INCLUDES@
AM_LDFLAGS = @LDFLAGS@

crf_CFLAGS = -I$(top_builddir)/lib/crf/include
crf_LDADD = $(top_builddir)/lib/crf/libcrf.la
