# Compiler
CC = g++-12
CPP_FILES := $(wildcard cudann/*/*/*.cpp)

# Compiler flags
CC_FLAGS = -O3

# Compile random
compile_random: cudann/test/test_random.cpp
	$(CC) -o cudann/test/test_random cudann/test/test_random.cpp $(CPP_FILES)

# Compile and test random
test_random:
	make compile_random
	cudann/test/test_random
	rm cudann/test/test_random

# Compile mnist
compile_mnist: cudann/test/test_mnist.cpp
	$(CC) -o cudann/test/test_mnist cudann/test/test_mnist.cpp $(CPP_FILES)

# Compile and test mnist
test_mnist:
	make compile_mnist
	cudann/test/test_mnist
	rm cudann/test/test_mnist


