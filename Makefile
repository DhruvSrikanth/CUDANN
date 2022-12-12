# Compiler
CC = g++-12

# Compiler flags
CC_FLAGS = -O3

# Compile
compile: main.cpp
	$(CC) $(CC_FLAGS) main.cpp -o main

# Run
run: main
	./main