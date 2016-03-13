# $Id:$
lib_LTLIBRARIES = libcqdb.la

EXTRA_DIST = \
	cqdb.vcxproj \
	COPYING

libcqdb_la_SOURCES = \
	src/lookup3.c \
	include/cqdb.h \
	src/cqdb.c

libcqdb_la_LDFLAGS = \
	-no-undefined \
	-release @VERSION@

libcqdb_la_CFLAGS = -I./include

bin_PROGRAMS = cqdbtest
TESTS = cqdbtest
cqdbtest_SOURCES = src/cqdb_unittests.cc
cqdbtest_LDFLAGS = $(GTEST_LDFLAGS)
cqdbtest_LDADD = libcqdb.la $(GTEST_LIBS)
cqdbtest_CPPFLAGS = -I./include $(GTEST_CPPFLAGS)

AM_CFLAGS = @CFLAGS@
AM_CPPFLAGS = @INCLUDES@
