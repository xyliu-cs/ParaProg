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
include src/cpu/CMakeFiles/simd_PartA.dir/depend.make

# Include the progress variables for this target.
include src/cpu/CMakeFiles/simd_PartA.dir/progress.make

# Include the compile flags for this target's objects.
include src/cpu/CMakeFiles/simd_PartA.dir/flags.make

src/cpu/CMakeFiles/simd_PartA.dir/simd_PartA.cpp.o: src/cpu/CMakeFiles/simd_PartA.dir/flags.make
src/cpu/CMakeFiles/simd_PartA.dir/simd_PartA.cpp.o: ../src/cpu/simd_PartA.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/nfsmnt/120040051/CSC4005-2023Fall/project1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/cpu/CMakeFiles/simd_PartA.dir/simd_PartA.cpp.o"
	cd /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/cpu && /opt/rh/devtoolset-7/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/simd_PartA.dir/simd_PartA.cpp.o -c /nfsmnt/120040051/CSC4005-2023Fall/project1/src/cpu/simd_PartA.cpp

src/cpu/CMakeFiles/simd_PartA.dir/simd_PartA.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/simd_PartA.dir/simd_PartA.cpp.i"
	cd /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/cpu && /opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /nfsmnt/120040051/CSC4005-2023Fall/project1/src/cpu/simd_PartA.cpp > CMakeFiles/simd_PartA.dir/simd_PartA.cpp.i

src/cpu/CMakeFiles/simd_PartA.dir/simd_PartA.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/simd_PartA.dir/simd_PartA.cpp.s"
	cd /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/cpu && /opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /nfsmnt/120040051/CSC4005-2023Fall/project1/src/cpu/simd_PartA.cpp -o CMakeFiles/simd_PartA.dir/simd_PartA.cpp.s

src/cpu/CMakeFiles/simd_PartA.dir/__/utils.cpp.o: src/cpu/CMakeFiles/simd_PartA.dir/flags.make
src/cpu/CMakeFiles/simd_PartA.dir/__/utils.cpp.o: ../src/utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/nfsmnt/120040051/CSC4005-2023Fall/project1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/cpu/CMakeFiles/simd_PartA.dir/__/utils.cpp.o"
	cd /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/cpu && /opt/rh/devtoolset-7/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/simd_PartA.dir/__/utils.cpp.o -c /nfsmnt/120040051/CSC4005-2023Fall/project1/src/utils.cpp

src/cpu/CMakeFiles/simd_PartA.dir/__/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/simd_PartA.dir/__/utils.cpp.i"
	cd /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/cpu && /opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /nfsmnt/120040051/CSC4005-2023Fall/project1/src/utils.cpp > CMakeFiles/simd_PartA.dir/__/utils.cpp.i

src/cpu/CMakeFiles/simd_PartA.dir/__/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/simd_PartA.dir/__/utils.cpp.s"
	cd /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/cpu && /opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /nfsmnt/120040051/CSC4005-2023Fall/project1/src/utils.cpp -o CMakeFiles/simd_PartA.dir/__/utils.cpp.s

# Object files for target simd_PartA
simd_PartA_OBJECTS = \
"CMakeFiles/simd_PartA.dir/simd_PartA.cpp.o" \
"CMakeFiles/simd_PartA.dir/__/utils.cpp.o"

# External object files for target simd_PartA
simd_PartA_EXTERNAL_OBJECTS =

src/cpu/simd_PartA: src/cpu/CMakeFiles/simd_PartA.dir/simd_PartA.cpp.o
src/cpu/simd_PartA: src/cpu/CMakeFiles/simd_PartA.dir/__/utils.cpp.o
src/cpu/simd_PartA: src/cpu/CMakeFiles/simd_PartA.dir/build.make
src/cpu/simd_PartA: src/cpu/CMakeFiles/simd_PartA.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/nfsmnt/120040051/CSC4005-2023Fall/project1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable simd_PartA"
	cd /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/cpu && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/simd_PartA.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/cpu/CMakeFiles/simd_PartA.dir/build: src/cpu/simd_PartA

.PHONY : src/cpu/CMakeFiles/simd_PartA.dir/build

src/cpu/CMakeFiles/simd_PartA.dir/clean:
	cd /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/cpu && $(CMAKE_COMMAND) -P CMakeFiles/simd_PartA.dir/cmake_clean.cmake
.PHONY : src/cpu/CMakeFiles/simd_PartA.dir/clean

src/cpu/CMakeFiles/simd_PartA.dir/depend:
	cd /nfsmnt/120040051/CSC4005-2023Fall/project1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /nfsmnt/120040051/CSC4005-2023Fall/project1 /nfsmnt/120040051/CSC4005-2023Fall/project1/src/cpu /nfsmnt/120040051/CSC4005-2023Fall/project1/build /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/cpu /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/cpu/CMakeFiles/simd_PartA.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/cpu/CMakeFiles/simd_PartA.dir/depend

