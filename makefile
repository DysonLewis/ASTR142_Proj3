# Makefile for C++ extension modules

PYTHON := python3
PYTHON_CONFIG := python3-config
NUMPY_INCLUDE := $(shell $(PYTHON) -c "import numpy; print(numpy.get_include())")
PYTHON_INCLUDE := $(shell $(PYTHON_CONFIG) --includes)
PYTHON_LDFLAGS := $(shell $(PYTHON_CONFIG) --ldflags --embed 2>/dev/null || $(PYTHON_CONFIG) --ldflags)

CXX := g++
CXXFLAGS := -std=c++23 -O3 -fPIC -Wall -Wextra -pedantic -fopenmp
INCLUDES := $(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE)
LDFLAGS_COMMON := -fopenmp

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    EXT_SUFFIX := $(shell $(PYTHON_CONFIG) --extension-suffix)
    LDFLAGS := -shared $(LDFLAGS_COMMON)
endif
ifeq ($(UNAME_S),Darwin)
    EXT_SUFFIX := $(shell $(PYTHON_CONFIG) --extension-suffix)
    LDFLAGS := -bundle -undefined dynamic_lookup $(LDFLAGS_COMMON)
endif

TARGET_ACCEL := accel$(EXT_SUFFIX)
TARGET_SIM := simulator$(EXT_SUFFIX)
SOURCE_ACCEL := accel.cpp
SOURCE_SIM := simulator.cpp

.PHONY: all clean test

all: $(TARGET_ACCEL) $(TARGET_SIM)

$(TARGET_ACCEL): $(SOURCE_ACCEL)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $<

$(TARGET_SIM): $(SOURCE_SIM)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $<

clean:
	rm -f $(TARGET_ACCEL) $(TARGET_SIM)
	rm -rf __pycache__
	rm -f *.pyc

test: $(TARGET_ACCEL) $(TARGET_SIM)
	$(PYTHON) test_accel.py
	$(PYTHON) proj1.py