# Compiler
CC = g++-12

# Compiler flags
CC_FLAGS = -O3 -target x86_64-apple-darwin

# Compile
compile: cudann/test/test.cpp
	$(CC) cudann/test/test.cpp cudann/serial/utils/tensor.cpp -o cudann/test/test

# Run
run: main
	./main

# Compile and test
test:
	make compile
	cudann/test/test
