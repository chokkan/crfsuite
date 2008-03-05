# $Id$

SUBDIRS = lib/cqdb lib/lbfgs lib/crf frontend

docdir = $(prefix)/share/doc/@PACKAGE@
doc_DATA = README INSTALL COPYING AUTHORS ChangeLog

EXTRA_DIST = \
	crfsuite.sln \
	autogen.sh \
	include/os.h \
	win32/stdint.h \
	example/CoNLL2000/to_crfsuite.py \
	example/CoNLL2000/crfsuite_to_flexcrf.py

AUTOMAKE_OPTIONS = foreign
ACLOCAL_AMFLAGS = -I m4
