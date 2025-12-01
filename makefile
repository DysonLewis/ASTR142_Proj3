# Makefile for accel C++ extension module

PYTHON := python3
PYTHON_CONFIG := python3-config
NUMPY_INCLUDE := $(shell $(PYTHON) -c "import numpy; print(numpy.get_include())")
PYTHON_INCLUDE := $(shell $(PYTHON_CONFIG) --includes)
PYTHON_LDFLAGS := $(shell $(PYTHON_CONFIG) --ldflags --embed 2>/dev/null || $(PYTHON_CONFIG) --ldflags)

CXX := g++
CXXFLAGS := -std=c++23 -O1 -fPIC -Wall -Wextra -pedantic
INCLUDES := $(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE)

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    EXT_SUFFIX := $(shell $(PYTHON_CONFIG) --extension-suffix)
    LDFLAGS := -shared
endif
ifeq ($(UNAME_S),Darwin)
    EXT_SUFFIX := $(shell $(PYTHON_CONFIG) --extension-suffix)
    LDFLAGS := -bundle -undefined dynamic_lookup
endif

TARGET := accel$(EXT_SUFFIX)
SOURCE := accel.cpp

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
	rm -rf __pycache__
	rm -f *.pyc

test: $(TARGET)
	$(PYTHON) test_accel.py