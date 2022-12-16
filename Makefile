# Compiler
CC = g++-12

# Compiler flags
CC_FLAGS = -O3 -target x86_64-apple-darwin

# Compile
compile: cudann/serial/test.cpp
	$(CC) cudann/serial/test.cpp cudann/serial/initialize.cpp cudann/serial/linear.cpp -o cudann/serial/test

# Run
run: main
	./main

# Compile and test
test:
	make compile
	cudann/serial/test
