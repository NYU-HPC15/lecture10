EXECUTABLES = reduction full_reduction
# CC=gcc49

all: $(EXECUTABLES)

LDFLAGS += $(foreach librarydir,$(subst :, ,$(LD_LIBRARY_PATH)),-L$(librarydir))

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
  LDFLAGS += -lrt -lOpenCL -lm
  CFLAGS += -Wall -std=gnu99 -g -O2
endif
ifeq ($(UNAME_S),Darwin)
  LDFLAGS +=  -framework OpenCL -lm
  CFLAGS += -Wall -std=c99 -g -O2
endif

ifdef OPENCL_INC
  CPPFLAGS = -I$(OPENCL_INC)
endif

ifdef OPENCL_LIB
  LDFLAGS = -L$(OPENCL_LIB)
endif

reduction.o: reduction.c cl-helper.h timing.h
full_reduction.o: full_reduction.c cl-helper.h timing.h
cl-helper.o: cl-helper.c cl-helper.h timing.h

reduction: reduction.o cl-helper.o
full_reduction: full_reduction.o cl-helper.o

clean:
	rm -f $(EXECUTABLES) *.o
