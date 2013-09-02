FLAGS = -fPIC -D_GNU_SOURCE -fno-omit-frame-pointer -pthread -Wall -fopenmp -lgomp

all:
	g++ -c ${FLAGS} shotgun_api.cpp -o shotgun_api.o
	g++ -shared ${FLAGS} -W1,-lgomp,-soname,shotgun_api.so -o shotgun_api.so shotgun_api.o
