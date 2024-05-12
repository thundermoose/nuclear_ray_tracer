nvidia_compiler := nvcc -c -std=c++17 -ccbin /usr/local/bin/g++
compiler := g++ -std=c++17 -c
linker := g++ -std=c++17

nvidia_external_dependencies := cuda
external_dependencies := $(nvidia_external_dependencies)

source_path := ./sources
release_objects_path := .release_objects
test_objects_path := .test_objects
debug_objects_path := .debug_objects
release_dependencies_path := .release_dependencies
test_dependencies_path := .test_dependencies
debug_dependencies_path := .debug_dependencies

nvidia_compiler_flags := -I$(source_path) -I/usr/local/include/blixttest/ $(shell pkg-config -cflags $(nvidia_external_dependencies))
compiler_flags := -I$(source_path) -I/usr/local/include/blixttest/ $(shell pkg-config -cflags $(external_dependencies))
linker_flags := $(shell pkg-config -libs $(external_dependencies)) -lstdc++
test_linker_flags := -L/usr/local/lib/ -lblixttest

executeable := nuclear_raytracer 
cuda_sources := $(shell find $(source_path) -regex [^\#]*\\.cu$)
sources := $(shell find $(source_path) -regex [^\#]*\\.cpp$)
release_objects := $(cuda_sources:$(source_path)/%.cu=$(release_objects_path)/%.o) $(sources:$(source_path)/%.cpp=$(release_objects_path)/%.o)
test_objects := $(cuda_sources:$(source_path)/%.cu=$(test_objects_path)/%.o) $(sources:$(source_path)/%.cpp=$(test_objects_path)/%.o)
debug_objects := $(cuda_sources:$(source_path)/%.cu=$(debug_objects_path)/%.o) $(sources:$(source_path)/%.cpp=$(debug_objects_path)/%.o)

.PRECIOUS: $(release_objects) $(test_objects) $(debug_objects)

all: release/$(executeable)

test: test/$(executeable)

debug: debug/$(executeable)

release/$(executeable): nvidia_compiler_flags+= -DNDEBUG -O3
release/$(executeable): $(release_objects)
	mkdir -p $(@D)
	$(linker) -o $@ $^ $(linker_flags)

test/%: nvidia_compiler_flags+= -g -G -DTEST
test/%: $(test_objects)
	mkdir -p $(@D)
	$(linker) -o $@ $^ $(linker_flags) $(test_linker_flags)

debug/%: nvidia_compiler_flags+= -g -G
debug/%: $(debug_objects)
	mkdir -p $(@D)
	$(linker) -o $@ $^ $(linker_flags)

clean:
	rm -rf $(release_objects_path) $(release_dependencies_path)
	rm -rf $(test_objects_path) $(test_dependencies_path)
	rm -rf $(debug_objects_path) $(debug_dependencies_path)

$(release_objects_path)/%.o: $(source_path)/%.cu
	mkdir -p $(@D)
	$(nvidia_compiler) -o $@ $< $(nvidia_compiler_flags)

$(release_objects_path)/%.o: $(source_path)/%.cpp
	mkdir -p $(@D)
	$(compiler) -o $@ $< $(compiler_flags)

$(test_objects_path)/%.o: $(source_path)/%.cu
	mkdir -p $(@D)
	$(nvidia_compiler) -o $@ $< $(nvidia_compiler_flags)

$(test_objects_path)/%.o: $(source_path)/%.cpp
	mkdir -p $(@D)
	$(compiler) -o $@ $< $(compiler_flags)

$(debug_objects_path)/%.o: $(source_path)/%.cu
	mkdir -p $(@D)
	$(nvidia_compiler) -o $@ $< $(nvidia_compiler_flags)

$(debug_objects_path)/%.o: $(source_path)/%.cpp
	mkdir -p $(@D)
	$(compiler) -o $@ $< $(compiler_flags)
