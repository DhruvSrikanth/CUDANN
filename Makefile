# Compiler
CC = g++-12
CPP_FILES := $(wildcard cudann/*/*/*.cpp)

# Compiler flags
CC_FLAGS = -O3


# Compile layers test
compile_layers: cudann/test/test_layers.cpp
	$(CC) -o cudann/test/test_layers cudann/test/test_layers.cpp $(CPP_FILES)

# Compile and test layers
test_layers:
	make compile_layers
	cudann/test/test_layers
	rm cudann/test/test_layers

# Compile criterion test
compile_criterion: cudann/test/test_criterion.cpp
	$(CC) -o cudann/test/test_criterion cudann/test/test_criterion.cpp $(CPP_FILES)

# Compile and test criterion
test_criterion:
	make compile_criterion
	cudann/test/test_criterion
	rm cudann/test/test_criterion

# Compile model test
compile_model: cudann/test/test_model.cpp
	$(CC) -o cudann/test/test_model cudann/test/test_model.cpp $(CPP_FILES)

# Compile and test model
test_model:
	make compile_model
	cudann/test/test_model
	rm cudann/test/test_model

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


