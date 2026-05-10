CC = gcc

CFLAGS = -O2 -Wall -Wextra -std=c11

OBJ = main.o utils.o dataset.o tree.o forest.o

all: random_forest

random_forest: $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o random_forest

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

utils.o: utils.c
	$(CC) $(CFLAGS) -c utils.c

dataset.o: dataset.c
	$(CC) $(CFLAGS) -c dataset.c

tree.o: tree.c
	$(CC) $(CFLAGS) -c tree.c

forest.o: forest.c
	$(CC) $(CFLAGS) -c forest.c

clean:
	del /Q *.o random_forest.exe