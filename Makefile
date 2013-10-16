FLAGS = -I/homes/grail/tbjohns/usr/local/include/python2.7 -g -D_GNU_SOURCE -fno-omit-frame-pointer -pthread -Wall -fopenmp -lgomp
CXX = g++

all: shotgun_api.o lasso.o logreg.o shared.o
	$(CXX) -shared $(FLAGS) -o shotgun_api.so shotgun_api.o lasso.o logreg.o shared.o

shotgun_api.o: shotgun_api.cpp common.h
	$(CXX) -fPIC $(FLAGS) -c shotgun_api.cpp

lasso.o: lasso.cpp common.h
	$(CXX) -fPIC $(FLAGS) -c lasso.cpp

logreg.o: logreg.cpp common.h
	$(CXX) -fPIC $(FLAGS) -c logreg.cpp

shared.o: shared.cpp common.h
	$(CXX) -fPIC $(FLAGS) -c shared.cpp

clean:
	rm -f *.o *.so
