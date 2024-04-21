compiler := nvcc -c
linker := nvcc
external_dependencies := cuda
source_path := ./sources
release_objects_path := .release_objects
test_objects_path := .test_objects
debug_objects_path := .debug_objects
release_dependencies_path := .release_dependencies
test_dependencies_path := .test_dependencies
debug_dependencies_path := .debug_dependencies

compiler_flags := -I$(source_path) -I/usr/local/include/blixttest/ $(shell pkg-config -cflags $(external_dependencies))
linker_flags := $(shell pkg-config -libs $(external_dependencies))
test_linker_flags := -L/usr/local/lib/ -lblixttest

executeable := nuclear_raytracer 
sources := $(shell find $(source_path) -regex [^\#]*\\.cu$)
release_objects := $(sources:$(source_path)/%.cu=$(release_objects_path)/%.o)
test_objects := $(sources:$(source_path)/%.cu=$(test_objects_path)/%.o)
debug_objects := $(sources:$(source_path)/%.cu=$(debug_objects_path)/%.o)

.PRECIOUS: $(release_objects) $(test_objects) $(debug_objects)

all: release/$(executeable)

test: test/$(executeable)

debug: debug/$(executeable)

release/$(executeable): compiler_flags+= -DNDEBUG -O3
release/$(executeable): $(release_objects)
	mkdir -p $(@D)
	$(linker) -o $@ $^ $(linker_flags)

test/%: compiler_flags+= -g -G -DTEST
test/%: $(test_objects)
	mkdir -p $(@D)
	$(linker) -o $@ $^ $(linker_flags) $(test_linker_flags)

debug/%: compiler_flags+= -g -G
debug/%: $(debug_objects)
	mkdir -p $(@D)
	$(linker) -o $@ $^ $(linker_flags)

clean:
	rm -rf $(release_objects_path) $(release_dependencies_path)
	rm -rf $(test_objects_path) $(test_dependencies_path)
	rm -rf $(debug_objects_path) $(debug_dependencies_path)

$(release_objects_path)/%.o: $(source_path)/%.cu
	mkdir -p $(@D)
	$(compiler) -o $@ $< $(compiler_flags)

$(test_objects_path)/%.o: $(source_path)/%.cu
	mkdir -p $(@D)
	$(compiler) -o $@ $< $(compiler_flags)

$(debug_objects_path)/%.o: $(source_path)/%.cu
	mkdir -p $(@D)
	$(compiler) -o $@ $< $(compiler_flags)

