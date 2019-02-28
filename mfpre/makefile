CC     = g++
CCFLAG = -march=native -O3 -pipe -Wall -Wextra -Wunused-parameter \
                        -Wunused-variable
INC    = -I$(METIS_DIR)/include
LDFLAGS = -lm -lmetis -L$(METIS_DIR)/lib
LDFLAGS += -Wl,-rpath=$(METIS_DIR)/lib

graphpart: graphPart.o
	$(CC) -o graphpart graphPart.o $(LDFLAGS)
	rm *.o

graphPart.o: graphPart.cpp
	$(CC) $(CCFLAG) $(INC) -c graphPart.cpp 

plot:
	python plot.py

clean:
	rm -rf *.o graphpart partGrid*
