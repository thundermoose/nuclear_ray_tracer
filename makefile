compiler := nvcc -c
linker := nvcc

source_path := ./sources
release_objects_path := .release_objects
test_objects_path := .test_objects
debug_objects_path := .debug_objects
release_dependencies_path := .release_dependencies
test_dependencies_path := .test_dependencies
debug_dependencies_path := .debug_dependencies

compiler_flags := -I$(source_path) -I/usr/local/include/blixttest/
linker_flags := 
test_linker_flags := -L/usr/local/lib/ -lblixttest

executeable := nuclear_raytracer 
sources := $(shell find $(source_path) -regex [^\#]*\\.cu$)
release_objects := $(sources:$(source_path)/%.cu=$(release_objects_path)/%.o)
test_objects := $(sources:$(source_path)/%.cu=$(test_objects_path)/%.o)
debug_objects := $(sources:$(source_path)/%.cu=$(debug_objects_path)/%.o)

release_dependencies := $(sources:$(source_path)/%.cu=$(release_dependencies_path)/%.d)
test_dependencies := $(sources:$(source_path)/%.cu=$(test_dependencies_path)/%.d)
debug_dependencies := $(sources:$(source_path)/%.cu=$(debug_dependencies_path)/%.d)

.PRECIOUS: $(release_objects) $(test_objects) $(debug_objects) $(release_dependencies) $(test_dependencies) $(debug_dependencies)

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

include $(release_dependencies)
$(release_objects_path)/%.o: $(source_path)/%.cu
	mkdir -p $(@D)
	$(compiler) -o $@ $< $(compiler_flags)

include $(test_dependencies)
$(test_objects_path)/%.o: $(source_path)/%.cu
	mkdir -p $(@D)
	$(compiler) -o $@ $< $(compiler_flags)

include $(debug_dependencies)
$(debug_objects_path)/%.o: $(source_path)/%.cu
	mkdir -p $(@D)
	$(compiler) -o $@ $< $(compiler_flags)

$(release_dependencies_path)/%.d: $(source_path)/%.cu
	mkdir -p $(@D)
	$(compiler) -o /dev/null -M -MF $@ -MT $(@:$(release_dependencies_path)/%.d=$(release_objects_path)/%.o) $< $(compiler_flags)

$(test_dependencies_path)/%.d: $(source_path)/%.cu
	mkdir -p $(@D)
	$(compiler) -o /dev/null -M -MF $@ -MT $(@:$(test_dependencies_path)/%.d=$(test_objects_path)/%.o) $< $(compiler_flags)

$(debug_dependencies_path)/%.d: $(source_path)/%.cu
	mkdir -p $(@D)
	$(compiler) -o /dev/null -M -MF $@ -MT $(@:$(debug_dependencies_path)/%.d=$(debug_objects_path)/%.o) $< $(compiler_flags)
