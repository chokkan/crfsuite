# $Id$

SUBDIRS = include lib/cqdb lib/crf frontend swig

docdir = $(prefix)/share/doc/@PACKAGE@
doc_DATA = README INSTALL COPYING AUTHORS ChangeLog

EXTRA_DIST = \
	crfsuite.sln \
	autogen.sh \
	win32/stdint.h \
	example/crfutils.py \
	example/template.py \
	example/pos.py \
	example/ner.py \
	example/chunking.py

AUTOMAKE_OPTIONS = foreign
ACLOCAL_AMFLAGS = -I m4

##############################
# googletest support
##############################
all-local: googletest/.built
check-recursive: googletest/.built

googletest/.built: googletest/Makefile
	cmake --build $(@D)
	touch $@
googletest/Makefile: googletest/CMakeLists.txt
	cd $(@D); cmake .
