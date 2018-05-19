CC=gcc
CFLAGS=-O1 -msse -msse2 -msse3 -msse4 -msse4.1 -fomit-frame-pointer -Wall -Wextra -DDSFMT_MEXP=2203
CLIBS=-I $(HOME)/include -L $(HOME)/lib -lutils -lm -lcmdl
EXEC=graph
INSTALL=install -m 111
BINDIR=$(HOME)/bin/

all: main.c dSFMT.c
	$(CC) $(CFLAGS) main.c dSFMT.c -o $(EXEC) $(CLIBS)
	ctags *.c

install:
	$(INSTALL) $(EXEC) $(BINDIR)
