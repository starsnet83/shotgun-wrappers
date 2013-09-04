FLAGS = -D_GNU_SOURCE -fno-omit-frame-pointer -pthread -Wall -fopenmp -lgomp
CXX = g++

all: shotgun_api.o lasso.o
	$(CXX) -shared $(FLAGS) -o shotgun_api.so shotgun_api.o lasso.o

shotgun_api.o: shotgun_api.cpp
	$(CXX) -fPIC $(FLAGS) -c shotgun_api.cpp

lasso.o: lasso.cpp
	$(CXX) -fPIC $(FLAGS) -c lasso.cpp

clean:
	rm -f *.o *.so
