all:	nestedsampling

nestedsampling: NestedSampling.o MCMC.o Distributions.o CircBuffer.o Globals.o
	g++ --std=c++11 -o cpp_out_temp NestedSampling.o MCMC.o Distributions.o CircBuffer.o Globals.o;  \

# dependencies....
