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
include src/cpu/CMakeFiles/pthread_PartB.dir/depend.make

# Include the progress variables for this target.
include src/cpu/CMakeFiles/pthread_PartB.dir/progress.make

# Include the compile flags for this target's objects.
include src/cpu/CMakeFiles/pthread_PartB.dir/flags.make

src/cpu/CMakeFiles/pthread_PartB.dir/pthread_PartB.cpp.o: src/cpu/CMakeFiles/pthread_PartB.dir/flags.make
src/cpu/CMakeFiles/pthread_PartB.dir/pthread_PartB.cpp.o: ../src/cpu/pthread_PartB.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/nfsmnt/120040051/CSC4005-2023Fall/project1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/cpu/CMakeFiles/pthread_PartB.dir/pthread_PartB.cpp.o"
	cd /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/cpu && /opt/rh/devtoolset-7/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pthread_PartB.dir/pthread_PartB.cpp.o -c /nfsmnt/120040051/CSC4005-2023Fall/project1/src/cpu/pthread_PartB.cpp

src/cpu/CMakeFiles/pthread_PartB.dir/pthread_PartB.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pthread_PartB.dir/pthread_PartB.cpp.i"
	cd /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/cpu && /opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /nfsmnt/120040051/CSC4005-2023Fall/project1/src/cpu/pthread_PartB.cpp > CMakeFiles/pthread_PartB.dir/pthread_PartB.cpp.i

src/cpu/CMakeFiles/pthread_PartB.dir/pthread_PartB.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pthread_PartB.dir/pthread_PartB.cpp.s"
	cd /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/cpu && /opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /nfsmnt/120040051/CSC4005-2023Fall/project1/src/cpu/pthread_PartB.cpp -o CMakeFiles/pthread_PartB.dir/pthread_PartB.cpp.s

src/cpu/CMakeFiles/pthread_PartB.dir/__/utils.cpp.o: src/cpu/CMakeFiles/pthread_PartB.dir/flags.make
src/cpu/CMakeFiles/pthread_PartB.dir/__/utils.cpp.o: ../src/utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/nfsmnt/120040051/CSC4005-2023Fall/project1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/cpu/CMakeFiles/pthread_PartB.dir/__/utils.cpp.o"
	cd /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/cpu && /opt/rh/devtoolset-7/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pthread_PartB.dir/__/utils.cpp.o -c /nfsmnt/120040051/CSC4005-2023Fall/project1/src/utils.cpp

src/cpu/CMakeFiles/pthread_PartB.dir/__/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pthread_PartB.dir/__/utils.cpp.i"
	cd /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/cpu && /opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /nfsmnt/120040051/CSC4005-2023Fall/project1/src/utils.cpp > CMakeFiles/pthread_PartB.dir/__/utils.cpp.i

src/cpu/CMakeFiles/pthread_PartB.dir/__/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pthread_PartB.dir/__/utils.cpp.s"
	cd /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/cpu && /opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /nfsmnt/120040051/CSC4005-2023Fall/project1/src/utils.cpp -o CMakeFiles/pthread_PartB.dir/__/utils.cpp.s

# Object files for target pthread_PartB
pthread_PartB_OBJECTS = \
"CMakeFiles/pthread_PartB.dir/pthread_PartB.cpp.o" \
"CMakeFiles/pthread_PartB.dir/__/utils.cpp.o"

# External object files for target pthread_PartB
pthread_PartB_EXTERNAL_OBJECTS =

src/cpu/pthread_PartB: src/cpu/CMakeFiles/pthread_PartB.dir/pthread_PartB.cpp.o
src/cpu/pthread_PartB: src/cpu/CMakeFiles/pthread_PartB.dir/__/utils.cpp.o
src/cpu/pthread_PartB: src/cpu/CMakeFiles/pthread_PartB.dir/build.make
src/cpu/pthread_PartB: src/cpu/CMakeFiles/pthread_PartB.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/nfsmnt/120040051/CSC4005-2023Fall/project1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable pthread_PartB"
	cd /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/cpu && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pthread_PartB.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/cpu/CMakeFiles/pthread_PartB.dir/build: src/cpu/pthread_PartB

.PHONY : src/cpu/CMakeFiles/pthread_PartB.dir/build

src/cpu/CMakeFiles/pthread_PartB.dir/clean:
	cd /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/cpu && $(CMAKE_COMMAND) -P CMakeFiles/pthread_PartB.dir/cmake_clean.cmake
.PHONY : src/cpu/CMakeFiles/pthread_PartB.dir/clean

src/cpu/CMakeFiles/pthread_PartB.dir/depend:
	cd /nfsmnt/120040051/CSC4005-2023Fall/project1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /nfsmnt/120040051/CSC4005-2023Fall/project1 /nfsmnt/120040051/CSC4005-2023Fall/project1/src/cpu /nfsmnt/120040051/CSC4005-2023Fall/project1/build /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/cpu /nfsmnt/120040051/CSC4005-2023Fall/project1/build/src/cpu/CMakeFiles/pthread_PartB.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/cpu/CMakeFiles/pthread_PartB.dir/depend

