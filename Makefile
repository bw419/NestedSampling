CC = g++ --std=c++11

all:	nestedsampling

nestedsampling: NestedSampling.o MCMC.o Distributions.o CircBuffer.o Globals.o
	$(CC) -o nestedsampling NestedSampling.o MCMC.o Distributions.o CircBuffer.o Globals.o;  \

# dependencies....

.cpp.o:
	$(CC) -c $<

clean:
	echo cleaning up; /bin/rm -f core *.o nestedsampling
