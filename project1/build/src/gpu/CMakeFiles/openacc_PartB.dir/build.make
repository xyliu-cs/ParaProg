# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /nfsmnt/120040051/CSC4005-2023Fall/project1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /nfsmnt/120040051/CSC4005-2023Fall/project1/build

# Include any dependencies generated for this target.
include src/gpu/CMakeFiles/openacc_PartB.dir/depend.make

# Include the progress variables for this target.
include src/gpu/CMakeFiles/openacc_PartB.dir/progress.make

# Include the compile flags for this target's objects.
include src/gpu/CMakeFiles/openacc_PartB.dir/flags.make

src/gpu/CMakeFiles/openacc_PartB.dir/openacc_PartB.cpp.o: src/gpu/CMakeFiles/openacc_PartB.dir/flags.make
src/gpu/CMakeFiles/openacc_PartB.dir/openacc_PartB.cpp.o: ../src/gpu/openacc_PartB.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/nfsmnt/120040051/CSC4005-2023Fall/project1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/gpu/CMakeFiles/openacc_PartB.dir/openacc_PartB.cpp.o"
	cd /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/gpu && pgc++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/openacc_PartB.dir/openacc_PartB.cpp.o -c /nfsmnt/120040051/CSC4005-2023Fall/project1/src/gpu/openacc_PartB.cpp

src/gpu/CMakeFiles/openacc_PartB.dir/openacc_PartB.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/openacc_PartB.dir/openacc_PartB.cpp.i"
	cd /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/gpu && pgc++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /nfsmnt/120040051/CSC4005-2023Fall/project1/src/gpu/openacc_PartB.cpp > CMakeFiles/openacc_PartB.dir/openacc_PartB.cpp.i

src/gpu/CMakeFiles/openacc_PartB.dir/openacc_PartB.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/openacc_PartB.dir/openacc_PartB.cpp.s"
	cd /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/gpu && pgc++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /nfsmnt/120040051/CSC4005-2023Fall/project1/src/gpu/openacc_PartB.cpp -o CMakeFiles/openacc_PartB.dir/openacc_PartB.cpp.s

src/gpu/CMakeFiles/openacc_PartB.dir/__/utils.cpp.o: src/gpu/CMakeFiles/openacc_PartB.dir/flags.make
src/gpu/CMakeFiles/openacc_PartB.dir/__/utils.cpp.o: ../src/utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/nfsmnt/120040051/CSC4005-2023Fall/project1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/gpu/CMakeFiles/openacc_PartB.dir/__/utils.cpp.o"
	cd /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/gpu && pgc++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/openacc_PartB.dir/__/utils.cpp.o -c /nfsmnt/120040051/CSC4005-2023Fall/project1/src/utils.cpp

src/gpu/CMakeFiles/openacc_PartB.dir/__/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/openacc_PartB.dir/__/utils.cpp.i"
	cd /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/gpu && pgc++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /nfsmnt/120040051/CSC4005-2023Fall/project1/src/utils.cpp > CMakeFiles/openacc_PartB.dir/__/utils.cpp.i

src/gpu/CMakeFiles/openacc_PartB.dir/__/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/openacc_PartB.dir/__/utils.cpp.s"
	cd /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/gpu && pgc++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /nfsmnt/120040051/CSC4005-2023Fall/project1/src/utils.cpp -o CMakeFiles/openacc_PartB.dir/__/utils.cpp.s

# Object files for target openacc_PartB
openacc_PartB_OBJECTS = \
"CMakeFiles/openacc_PartB.dir/openacc_PartB.cpp.o" \
"CMakeFiles/openacc_PartB.dir/__/utils.cpp.o"

# External object files for target openacc_PartB
openacc_PartB_EXTERNAL_OBJECTS =

src/gpu/openacc_PartB: src/gpu/CMakeFiles/openacc_PartB.dir/openacc_PartB.cpp.o
src/gpu/openacc_PartB: src/gpu/CMakeFiles/openacc_PartB.dir/__/utils.cpp.o
src/gpu/openacc_PartB: src/gpu/CMakeFiles/openacc_PartB.dir/build.make
src/gpu/openacc_PartB: src/gpu/CMakeFiles/openacc_PartB.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/nfsmnt/120040051/CSC4005-2023Fall/project1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable openacc_PartB"
	cd /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/gpu && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/openacc_PartB.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/gpu/CMakeFiles/openacc_PartB.dir/build: src/gpu/openacc_PartB

.PHONY : src/gpu/CMakeFiles/openacc_PartB.dir/build

src/gpu/CMakeFiles/openacc_PartB.dir/clean:
	cd /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/gpu && $(CMAKE_COMMAND) -P CMakeFiles/openacc_PartB.dir/cmake_clean.cmake
.PHONY : src/gpu/CMakeFiles/openacc_PartB.dir/clean

src/gpu/CMakeFiles/openacc_PartB.dir/depend:
	cd /nfsmnt/120040051/CSC4005-2023Fall/project1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /nfsmnt/120040051/CSC4005-2023Fall/project1 /nfsmnt/120040051/CSC4005-2023Fall/project1/src/gpu /nfsmnt/120040051/CSC4005-2023Fall/project1/build /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/gpu /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/gpu/CMakeFiles/openacc_PartB.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/gpu/CMakeFiles/openacc_PartB.dir/depend

