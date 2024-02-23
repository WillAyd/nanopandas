#include "generic.hpp"

template <> const char *dtype([[maybe_unused]] const StringArray &self) {
  return "string[arrow]";
}

template <> const char *dtype([[maybe_unused]] const BoolArray &self) {
  return "boolean[arrow]";
}

template <> const char *dtype([[maybe_unused]] const Int64Array &self) {
  return "int64[arrow]";
}
