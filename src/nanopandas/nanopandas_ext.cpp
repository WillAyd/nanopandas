#include "algorithms.hpp"
#include "array_types.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

// try to match all pandas methods
// https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html#method-summary
NB_MODULE(nanopandas_ext, m) {
  nb::class_<ExtensionArray>(m, "ExtensionArray");

  nb::class_<BoolArray, ExtensionArray>(m, "BoolArray")
      .def(nb::init<std::vector<std::optional<bool>>>())
      .def("__len__", &__len__<BoolArray>)
      .def_prop_ro("dtype", &dtype<BoolArray>)
      .def_prop_ro("nbytes", &nbytes<BoolArray>)
      .def("size", &size<BoolArray>)
      .def("any", &any<BoolArray>)
      .def("all", &all<BoolArray>)
      .def("__repr__", &__repr__<BoolArray>)
      .def("__getitem__", &__getitem__<BoolArray>)
      .def("__eq__", &__eq__<BoolArray>)
      .def("isna", &isna<BoolArray>)
      .def("take", &take<BoolArray>)
      .def("copy", &copy<BoolArray>)
      .def("fillna", &fillna<BoolArray>)
      .def("dropna", &dropna<BoolArray>)
      .def("_from_sequence", &FromSequence<BoolArray>)
      .def("_from_factorized", &FromFactorized<BoolArray>)
      .def("to_pylist", &to_pylist<BoolArray>);

  nb::class_<Int64Array, ExtensionArray>(m, "Int64Array")
      .def(nb::init<std::vector<std::optional<int64_t>>>())
      .def("__len__", &__len__<Int64Array>)
      .def_prop_ro("dtype", &dtype<Int64Array>)
      .def_prop_ro("nbytes", &nbytes<Int64Array>)
      .def("size", &size<Int64Array>)
      .def("any", &any<Int64Array>)
      .def("all", &all<Int64Array>)
      .def("__repr__", &__repr__<Int64Array>)
      .def("__getitem__", &__getitem__<Int64Array>)
      .def("__eq__", &__eq__<Int64Array>)
      .def("isna", &isna<Int64Array>)
      .def("take", &take<Int64Array>)
      .def("copy", &copy<Int64Array>)
      .def("fillna", &fillna<Int64Array>)
      .def("dropna", &dropna<Int64Array>)
      .def("_from_sequence", &FromSequence<Int64Array>)
      .def("_from_factorized", &FromFactorized<Int64Array>)
      .def("to_pylist", &to_pylist<Int64Array>)

      // integral-specific algorithms
      .def("sum", &Sum<Int64Array>)
      .def("min", &Min<Int64Array>)
      .def("max", &Max<Int64Array>);

  nb::class_<StringArray, ExtensionArray>(m, "StringArray")
      .def(nb::init<std::vector<std::optional<std::string_view>>>())
      .def_prop_ro("dtype", &dtype<StringArray>)
      .def_prop_ro("nbytes", &nbytes<StringArray>)
      .def("size", &size<StringArray>)
      .def("any", &any<StringArray>)
      .def("all", &all<StringArray>)
      .def("__repr__", &__repr__<StringArray>)
      .def("__getitem__", &__getitem__<StringArray>)
      .def("__eq__", &__eq__<StringArray>)
      .def("isna", &isna<StringArray>)
      .def("take", &take<StringArray>)
      .def("copy", &copy<StringArray>)
      .def("fillna", &fillna<StringArray>)
      .def("dropna", &dropna<StringArray>)
      .def("_from_sequence", &FromSequence<StringArray>)
      .def("_from_factorized", &FromFactorized<StringArray>)
      .def("to_pylist", &to_pylist<StringArray>)

      // string-specific algorithms
      .def("len", &len<StringArray>)
      .def("lower", &Lower)
      .def("upper", &Upper)
      .def("capitalize", &Capitalize)
      .def("isalnum", &IsAlnum)
      .def("isalpha", &IsAlpha)
      .def("isdigit", &IsDigit)
      .def("isspace", &IsSpace)
      .def("islower", &IsLower)
      .def("isupper", &IsUpper);
}
