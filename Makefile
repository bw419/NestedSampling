CC = g++ --std=c++11
CCSW = -O3 -Wno-deprecated-declarations
PLATFORM = `uname`

all:	nestedsampling

nestedsampling: NestedSampling.o Globals.o MCMC.o Distributions.o CircBuffer.o
	$(CC) -o ../nestedsampling NestedSampling.o Globals.o MCMC.o Distributions.o CircBuffer.o ${CCSW} -lGL -lGLU -lglut; \

# dependencies....

.cpp.o:
	$(CC) ${CCSW} -c $<

clean:
	echo cleaning up; /bin/rm -f core *.o ../nestedsampling

