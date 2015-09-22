# makefile for generic c++ project
# generated with `makeproject` on Tue Sep 22 08:07:33 PDT 2015
# Author: Dan Guest <dguest@cern.ch>

# _______________________________________________________________
# Basic Setup

# --- set dirs
BUILD        := build
SRC          := src
INC          := include
DICT         := dict
OUTPUT       := bin
LIB          := lib

#  set search path
vpath %.cxx  $(SRC)
vpath %.hh   $(INC)
vpath %.h    $(INC)
vpath %Dict.h $(DICT)
vpath %Dict.cxx $(DICT)

# --- set compiler and flags (roll c options and include paths together)
CXX          ?= g++
CXXFLAGS     := -O2 -Wall -fPIC -I$(INC) -g -std=c++11
LIBS         := # blank, more will be added below
LDFLAGS      := # blank, more will be added below
# add eigen
CXXFLAGS     += $(shell pkg-config eigen3 --cflags)

# ---- define objects from files in the SRC directory
GEN_OBJ_SRC   := $(wildcard $(SRC)/*.cxx)
GEN_OBJ       := $(notdir $(GEN_OBJ_SRC:%.cxx=%.o))

# this list may be manipulated in other segments further down
GEN_OBJ_PATHS := $(GEN_OBJ:%=$(BUILD)/%)

# --- all top level (added further down)
ALL_TOP_LEVEL := # blank, more will be added below

# _______________________________________________________________
# Add Top Level Objects

# --- python top level objects
PY_OBJ       := lwtag.o
PY_OBJ_PATH  := $(PY_OBJ:%=$(BUILD)/%)
PY_SRC_PATH  := $(PY_OBJ:%.o=$(SRC)/%.cxx)
PY_SO        := $(LIB)/lwtag.so

# filter out the general objects
GEN_OBJ_PATHS := $(filter-out $(BUILD)/lwtag.o,$(GEN_OBJ_PATHS))

# add to all top level
ALL_TOP_LEVEL += $(PY_SO)

# --- stuff used for the c++ executable
# everything with this prefix will be built as an executable
EXE_PREFIX   := lwtag-

ALL_EXE_SRC   := $(wildcard $(SRC)/$(EXE_PREFIX)*.cxx)
ALL_EXE       := $(notdir $(ALL_EXE_SRC:%.cxx=%))
ALL_EXE_PATHS := $(ALL_EXE:%=$(OUTPUT)/%)

# filter out the general objects
GEN_OBJ_PATHS := $(filter-out $(BUILD)/$(EXE_PREFIX)%.o,$(GEN_OBJ_PATHS))

# add to all top level
ALL_TOP_LEVEL += $(ALL_EXE_PATHS)

# _______________________________________________________________
# Add Libraries

# --- python config
PY_CONFIG := python3-config

PY_FLAGS  :=   $(shell $(PY_CONFIG) --includes)
PY_LIBS   := -L$(shell $(PY_CONFIG) --prefix)/lib
PY_LIBS   +=   $(shell $(PY_CONFIG) --libs)

# define these last because they inherit other LDFLAGS
PY_LDFLAGS := $(LDFLAGS)
PY_LDFLAGS += $(PY_LIBS)
PY_LDFLAGS += -shared

# --- first call here
all: $(ALL_TOP_LEVEL)

# _______________________________________________________________
# Add Build Rules

# python object compile
$(PY_OBJ_PATH): $(PY_SRC_PATH)
	@echo compiling python object $@
	@mkdir -p $(BUILD)
	@$(CXX) -c $(CXXFLAGS) $(PY_FLAGS) $< -o $@

# python linking
$(PY_SO): $(GEN_OBJ_PATHS) $(PY_OBJ_PATH)
	@mkdir -p $(LIB)
	@echo "linking $^ --> $@"
	@$(CXX) -o $@ $^ $(LIBS) $(PY_LDFLAGS)

# build exe
$(OUTPUT)/$(EXE_PREFIX)%: $(GEN_OBJ_PATHS) $(BUILD)/$(EXE_PREFIX)%.o
	@mkdir -p $(OUTPUT)
	@echo "linking $^ --> $@"
	@$(CXX) -o $@ $^ $(LIBS) $(LDFLAGS)

# compile rule
$(BUILD)/%.o: %.cxx
	@echo compiling $<
	@mkdir -p $(BUILD)
	@$(CXX) -c $(CXXFLAGS) $< -o $@

# use auto dependency generation
ALLOBJ       := $(GEN_OBJ)
DEP          := $(BUILD)

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(MAKECMDGOALS),rmdep)
include  $(ALLOBJ:%.o=$(DEP)/%.d)
endif
endif

DEPTARGSTR = -MT $(BUILD)/$*.o -MT $(DEP)/$*.d
$(DEP)/%.d: %.cxx
	@echo making dependencies for $<
	@mkdir -p $(DEP)
	@$(CXX) -MM -MP $(DEPTARGSTR) $(CXXFLAGS) $(PY_FLAGS) $< -o $@

# clean
.PHONY : clean rmdep all
CLEANLIST     = *~ *.o *.o~ *.d core
clean:
	rm -fr $(CLEANLIST) $(CLEANLIST:%=$(BUILD)/%) $(CLEANLIST:%=$(DEP)/%)
	rm -fr $(BUILD) $(DICT) $(OUTPUT)

rmdep:
	rm -f $(DEP)/*.d
