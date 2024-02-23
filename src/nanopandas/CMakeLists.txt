nanobind_add_module(nanopandaslib NOMINSIZE arrays.cpp
  algorithms/string_algorithms.cpp
  algorithms/generic_algorithms.cpp
)
target_link_libraries(nanopandaslib
  PRIVATE nanoarrow
  PRIVATE utf8proc
)
set_target_properties(nanoarrow
                      PROPERTIES POSITION_INDEPENDENT_CODE
                      ON)

install(TARGETS nanopandaslib
  LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME}/)

set(SRC_FILES
  __init__.py
)

if (NOT (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR) OR MSVC)
  if (MSVC)
    set(OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>)
  else()
    set(OUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
  endif()

  foreach(SRC_FILE IN LISTS SRC_FILES)
    set(IN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/${SRC_FILE})
    set(OUT_FILE ${OUT_DIR}/${SRC_FILE})
    set(SRC_FILES_OUT ${SRC_FILES_OUT} ${OUT_FILE})
    add_custom_command(
      DEPENDS ${IN_FILE} TARGET OUTPUT ${OUT_FILE}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${IN_FILE} ${OUT_DIR})
  endforeach()

  add_custom_target(copy-python-src ALL DEPENDS ${SRC_FILES_OUT})
endif()
