#include "algorithms.hpp"
#include "array_types.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

// try to match all pandas methods
// https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html#method-summary
NB_MODULE(nanopandas_ext, m) {
  nb::class_<ExtensionArray>(m, "ExtensionArray");

  nb::class_<BoolArray, ExtensionArray>(m, "BoolArray")
      .def(nb::init<std::vector<std::optional<bool>>>())
      .def("__len__", &LenDunder<BoolArray>)
      .def_prop_ro("dtype", &Dtype<BoolArray>)
      .def_prop_ro("nbytes", &Nbytes<BoolArray>)
      .def_prop_ro("size", &Size<BoolArray>)
      .def("any", &Any<BoolArray>)
      .def("all", &All<BoolArray>)
      .def("__repr__", &ReprDunder<BoolArray>)
      .def("__getitem__", &GetItemDunder<BoolArray>)
      .def("__eq__", &EqDunder<BoolArray>)
      .def("isna", &IsNA<BoolArray>)
      .def("take", &Take<BoolArray>)
      .def("copy", &Copy<BoolArray>)
      .def("fillna", &FillNA<BoolArray>)
      .def("dropna", &DropNA<BoolArray>)
      .def("interpolate", &Interpolate<BoolArray>)
      .def("unique", &Unique<BoolArray>)
      .def("factorize", &Factorize<BoolArray>)
      .def("_pad_or_backfill", &PadOrBackfill<BoolArray>)
      .def("_from_sequence", &FromSequence<BoolArray>)
      .def("_from_factorized", &FromFactorized<BoolArray>)
      .def("to_pylist", &ToPyList<BoolArray>)
      .def("_concat_same_type", &ConcatSameType<BoolArray>);

  nb::class_<Int64Array, ExtensionArray>(m, "Int64Array")
      .def(nb::init<std::vector<std::optional<int64_t>>>())
      .def("__len__", &LenDunder<Int64Array>)
      .def_prop_ro("dtype", &Dtype<Int64Array>)
      .def_prop_ro("nbytes", &Nbytes<Int64Array>)
      .def_prop_ro("size", &Size<Int64Array>)
      .def("any", &Any<Int64Array>)
      .def("all", &All<Int64Array>)
      .def("__repr__", &ReprDunder<Int64Array>)
      .def("__getitem__", &GetItemDunder<Int64Array>)
      .def("__eq__", &EqDunder<Int64Array>)
      .def("isna", &IsNA<Int64Array>)
      .def("take", &Take<Int64Array>)
      .def("copy", &Copy<Int64Array>)
      .def("fillna", &FillNA<Int64Array>)
      .def("dropna", &DropNA<Int64Array>)
      .def("interpolate", &Interpolate<Int64Array>)
      .def("unique", &Unique<Int64Array>)
      .def("factorize", &Factorize<Int64Array>)
      .def("_pad_or_backfill", &PadOrBackfill<Int64Array>)
      .def("_from_sequence", &FromSequence<Int64Array>)
      .def("_from_factorized", &FromFactorized<Int64Array>)
      .def("to_pylist", &ToPyList<Int64Array>)
      .def("_concat_same_type", &ConcatSameType<Int64Array>)

      // integral-specific algorithms
      .def("sum", &Sum<Int64Array>)
      .def("min", &Min<Int64Array>)
      .def("max", &Max<Int64Array>);

  nb::class_<StringArray, ExtensionArray>(m, "StringArray")
      .def(nb::init<std::vector<std::optional<std::string_view>>>())
      .def("__len__", &LenDunder<StringArray>)
      .def_prop_ro("dtype", &Dtype<StringArray>)
      .def_prop_ro("nbytes", &Nbytes<StringArray>)
      .def_prop_ro("size", &Size<StringArray>)
      .def("any", &Any<StringArray>)
      .def("all", &All<StringArray>)
      .def("__repr__", &ReprDunder<StringArray>)
      .def("__getitem__", &GetItemDunder<StringArray>)
      .def("__eq__", &EqDunder<StringArray>)
      .def("isna", &IsNA<StringArray>)
      .def("take", &Take<StringArray>)
      .def("copy", &Copy<StringArray>)
      .def("fillna", &FillNA<StringArray>)
      .def("dropna", &DropNA<StringArray>)
      .def("interpolate", &Interpolate<StringArray>)
      .def("unique", &Unique<StringArray>)
      .def("factorize", &Factorize<StringArray>)
      .def("_pad_or_backfill", &PadOrBackfill<StringArray>)
      .def("_from_sequence", &FromSequence<StringArray>)
      .def("_from_factorized", &FromFactorized<StringArray>)
      .def("to_pylist", &ToPyList<StringArray>)
      .def("_concat_same_type", &ConcatSameType<StringArray>)

      // string-specific algorithms
      .def("len", &Len<StringArray>)
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
