#include "algorithms.hpp"
#include "array_types.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

// try to match all pandas methods
// https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html#method-summary
NB_MODULE(nanopandaslib, m) {
  nb::class_<ExtensionArray>(m, "ExtensionArray");

  nb::class_<BoolArray, ExtensionArray>(m, "BoolArray")
      .def(nb::init<std::vector<std::optional<bool>>>())
      .def("__len__", &__len__<BoolArray>)
      .def("dtype", &dtype<BoolArray>)
      .def("nbytes", &nbytes<BoolArray>)
      .def("size", &size<BoolArray>)
      .def("any", &any<BoolArray>)
      .def("all", &all<BoolArray>);

  nb::class_<Int64Array, ExtensionArray>(m, "Int64Array")
      .def(nb::init<std::vector<std::optional<int64_t>>>())
      .def("__len__", &__len__<Int64Array>)
      .def("dtype", &dtype<Int64Array>)
      .def("nbytes", &nbytes<Int64Array>)
      .def("size", &size<Int64Array>)
      .def("any", &any<Int64Array>)
      .def("all", &all<Int64Array>);

  nb::class_<StringArray, ExtensionArray>(m, "StringArray")
      .def(nb::init<std::vector<std::optional<std::string_view>>>())
      .def("dtype", &dtype<StringArray>)
      .def("nbytes", &nbytes<StringArray>)
      .def("size", &size<StringArray>)
      .def("any", &any<StringArray>)
      .def("all", &all<StringArray>);
}
