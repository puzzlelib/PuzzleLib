OUTPUT_NAME := Driver.so

SRC_DIR := .
OBJ_DIR := .

SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRC_FILES))

PY_VERSION := python3.6


CC := g++
CPPFLAGS := -Wall -std=c++11 -O3 -I/usr/local/include/$(PY_VERSION) -I/usr/include/$(PY_VERSION) $\
    -I$(HOME)/.local/include/$(PY_VERSION)m -I/opt/rocm/opencl/include -fPIC

LDFLAGS := -L/opt/rocm/opencl/lib/x86_64 -lOpenCL


$(OUTPUT_NAME): $(OBJ_FILES)
	$(CC) -shared -o $@ $^ $(LDFLAGS)


$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CPPFLAGS) -c -o $@ $<


clean:
	rm -f $(OBJ_DIR)/*.o $(OUTPUT_NAME)
