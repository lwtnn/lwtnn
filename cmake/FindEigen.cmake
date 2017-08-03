#
# Simple module for finding the Eigen headers on the build system.
#

# Look for the "Eigen/Eigen" header file.
find_path( Eigen_INCLUDE_DIR NAMES Eigen/Eigen
   PATH_SUFFIXES include include/eigen3
   HINTS ${EIGEN_ROOT_DIR} )

# Set the Eigen_INCLUDE_DIRS variable as well, as that's the one usually
# expected by clients.
set( Eigen_INCLUDE_DIRS ${Eigen_INCLUDE_DIR}
   CACHE PATH "Eigen include directories to use" )

# Handle the standard find_package arguments.
include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( Eigen
   FOUND_VAR Eigen_FOUND
   REQUIRED_VARS Eigen_INCLUDE_DIR )
mark_as_advanced( Eigen_FOUND Eigen_INCLUDE_DIR Eigen_INCLUDE_DIRS )
