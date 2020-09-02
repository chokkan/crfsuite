cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0048 NEW)

set(CRFSUITE_MAJOR_VERSION 1)
set(CRFSUITE_MINOR_VERSION 0)
set(CRFSUITE_PATCH_VERSION 0)
set(CRFSUITE_VERSION ${CRFSUITE_MAJOR_VERSION}.${CRFSUITE_MINOR_VERSION}.${CRFSUITE_PATCH_VERSION})

project(crfsuite VERSION ${CRFSUITE_VERSION} LANGUAGES C)

find_package(lbfgs REQUIRED)

add_library(cqdb STATIC lib/cqdb/src/lookup3.c lib/cqdb/src/cqdb.c)
set(HEADER_FILES
  lib/cqdb/include/cqdb.h
)
target_include_directories(cqdb PRIVATE lib/cqdb/include)
set_target_properties(cqdb
  PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  PUBLIC_HEADER "${HEADER_FILES}"
  )

add_library(crf STATIC lib/crf/src/dictionary.c lib/crf/src/logging.c lib/crf/src/params.c lib/crf/src/quark.c lib/crf/src/rumavl.c lib/crf/src/dataset.c lib/crf/src/holdout.c lib/crf/src/train_arow.c lib/crf/src/train_averaged_perceptron.c lib/crf/src/train_l2sgd.c lib/crf/src/train_lbfgs.c lib/crf/src/train_passive_aggressive.c lib/crf/src/crf1d_context.c lib/crf/src/crf1d_model.c lib/crf/src/crf1d_feature.c lib/crf/src/crf1d_encode.c lib/crf/src/crf1d_tag.c lib/crf/src/crfsuite_train.c lib/crf/src/crfsuite.c)
set(HEADER_FILES
  lib/crf/src/logging.h lib/crf/src/params.h lib/crf/src/quark.h lib/crf/src/rumavl.h lib/crf/src/vecmath.h lib/crf/src/crfsuite_internal.h lib/crf/src/crf1d.h
)
target_include_directories(crf PRIVATE lib/crf/src include lib/cqdb/include ${LBFGS_INCLUDE_DIRS})
set_target_properties(crf
  PROPERTIES
  OUTPUT_NAME crfsuite
  POSITION_INDEPENDENT_CODE ON
  PUBLIC_HEADER "${HEADER_FILES}"
  )


add_executable(crfsuite
  frontend/dump.c frontend/iwa.c frontend/learn.c frontend/main.c frontend/option.c frontend/reader.c frontend/tag.c
  frontend/iwa.h frontend/option.h frontend/readdata.h
  )
target_include_directories(crfsuite PRIVATE lib/crf/src include lib/cqdb/include)
target_link_libraries(crfsuite crf cqdb m ${LBFGS_LIBRARIES})

install(TARGETS cqdb crf crfsuite EXPORT crfsuite-targets
  RUNTIME DESTINATION bin COMPONENT bin
  ARCHIVE DESTINATION lib COMPONENT dev
  LIBRARY DESTINATION lib COMPONENT shlib
  PUBLIC_HEADER DESTINATION include COMPONENT dev)
export(TARGETS cqdb crf FILE "${PROJECT_BINARY_DIR}/crfsuite-targets.cmake")
export(PACKAGE crfsuite)


# Create the Config.cmake and ConfigVersion.cmake files
get_filename_component(CMAKE_INSTALL_INCLUDEDIR_ABS include REALPATH BASE_DIR "${CMAKE_INSTALL_PREFIX}")
set(CONF_INCLUDE_DIRS ${CMAKE_INSTALL_INCLUDEDIR_ABS})
configure_file(crfsuiteConfig.cmake.in "cmake/crfsuiteConfig.cmake" @ONLY)
configure_file(crfsuiteConfigVersion.cmake.in "cmake/crfsuiteConfigVersion.cmake" @ONLY)
install(FILES "${PROJECT_BINARY_DIR}/cmake/crfsuiteConfig.cmake" "${PROJECT_BINARY_DIR}/cmake/crfsuiteConfigVersion.cmake" DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/crfsuite" COMPONENT dev)
set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include" "${PROJECT_BINARY_DIR}")
configure_file(crfsuiteConfig.cmake.in "crfsuiteConfig.cmake" @ONLY)
configure_file(crfsuiteConfigVersion.cmake.in "${PROJECT_BINARY_DIR}/crfsuiteConfigVersion.cmake" @ONLY)

# add_executable(sample sample/sample.c)
# target_link_libraries(sample PRIVATE crfsuite m)
# target_include_directories(sample PRIVATE ${CONF_INCLUDE_DIRS})
