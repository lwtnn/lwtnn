#
# Configuration file allowing clients to build code against the lwtnn
# library/libraries.
#

# Figure out the root directory of the installation.
get_filename_component( _thisdir "${CMAKE_CURRENT_LIST_FILE}" PATH )
get_filename_component( _basedir "${_thisdir}" PATH )
get_filename_component( lwtnn_INSTALL_DIR "${_basedir}" ABSOLUTE CACHE )

# Tell the user what happened.
if( NOT lwtnn_FIND_QUIETLY )
   message( STATUS
      "Found lwtnn: ${lwtnn_INSTALL_DIR} (version: ${lwtnn_VERSION})" )
endif()

# Include the "targets file".
include( ${_thisdir}/lwtnnConfig-targets.cmake )

# Set some "old style" variables for using the lwtnn installation.
set( lwtnn_INCLUDE_DIRS "${lwtnn_INSTALL_DIR}/include" )
set( lwtnn_LIBRARIES lwtnn::lwtnn )

# Clean up.
unset( _thisdir )
unset( _basedir )
