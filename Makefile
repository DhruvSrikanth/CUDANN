# Compiler
CC = g++-12
CPP_FILES := $(wildcard cudann/*/*/*.cpp)


# Compiler flags
CC_FLAGS = -O3 -target x86_64-apple-darwin

# Compile
compile: cudann/test/test.cpp
	$(CC) -o cudann/test/test cudann/test/test.cpp $(CPP_FILES)

clean:
	rm cudann/test/test

# Run
run: main
	./main

# Compile and test
test:
	make compile
	cudann/test/test
	make clean
