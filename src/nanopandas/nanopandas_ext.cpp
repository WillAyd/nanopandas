#include "algorithms.hpp"
#include "array_types.hpp"
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

// try to match all pandas methods
// https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html#method-summary
NB_MODULE(nanopandas_ext, m) {
  nb::class_<ExtensionArray>(m, "ExtensionArray");

  nb::class_<BoolArray, ExtensionArray>(m, "BoolArray")
      .def(nb::init<std::vector<std::optional<bool>>>())
      .def("__len__", &LenDunder<BoolArray>)
      .def_prop_ro("dtype", &Dtype<BoolArray>)
      .def_prop_ro("nbytes", &Nbytes<BoolArray>)
      .def_prop_ro("shape", &Shape<BoolArray>)
      .def_prop_ro("size", &Size<BoolArray>)
      .def_prop_ro("ndim", &NDim<BoolArray>)
      .def("any", &Any<BoolArray>)
      .def("all", &All<BoolArray>)
      .def("__repr__", &ReprDunder<BoolArray>)
      .def("__getitem__", &GetItemDunder<BoolArray>)
      .def("__eq__", &EqDunder<BoolArray>)
      .def("isna", &IsNA<BoolArray>)
      .def("take", &Take<BoolArray>, "indices"_a, "allow_fill"_a = false,
           "fill_value"_a = nb::none())
      .def("copy", &Copy<BoolArray>)
      .def("fillna", &FillNA<BoolArray>)
      .def("dropna", &DropNA<BoolArray>)
      .def("interpolate", &Interpolate<BoolArray>)
      .def("unique", &Unique<BoolArray>)
      .def("factorize", &Factorize<BoolArray>)
      .def("reshape", &Reshape<BoolArray>)
      .def("_pad_or_backfill", &PadOrBackfill<BoolArray>)
      .def("_from_sequence", &FromSequence<BoolArray>)
      .def("_from_factorized", &FromFactorized<BoolArray>)
      .def("tolist", &ToPyList<BoolArray>)
      .def("_concat_same_type", &ConcatSameType<BoolArray>);

  nb::class_<ExtensionDtype<BoolArray>>(m, "StringDtype")
      .def("__str__", &ExtensionDtype<BoolArray>::Str)
      .def_prop_ro("na_value", &ExtensionDtype<BoolArray>::NaValue)
      .def_prop_ro("kind", &ExtensionDtype<BoolArray>::Kind)
      .def_prop_ro("name", &ExtensionDtype<BoolArray>::Name)
      .def_prop_ro("is_numeric", &ExtensionDtype<BoolArray>::IsNumeric)
      .def_prop_ro("is_boolean", &ExtensionDtype<BoolArray>::IsBoolean)
      .def_prop_ro("_can_hold_na", &ExtensionDtype<BoolArray>::CanHoldNA)
      .def_prop_ro("_is_immutable", &ExtensionDtype<BoolArray>::IsImmutable);

  nb::class_<Int64Array, ExtensionArray>(m, "Int64Array")
      .def(nb::init<std::vector<std::optional<int64_t>>>())
      .def("__len__", &LenDunder<Int64Array>)
      .def_prop_ro("dtype", &Dtype<Int64Array>)
      .def_prop_ro("nbytes", &Nbytes<Int64Array>)
      .def_prop_ro("shape", &Shape<Int64Array>)
      .def_prop_ro("size", &Size<Int64Array>)
      .def_prop_ro("ndim", &NDim<Int64Array>)
      .def("any", &Any<Int64Array>)
      .def("all", &All<Int64Array>)
      .def("__repr__", &ReprDunder<Int64Array>)
      .def("__getitem__", &GetItemDunder<Int64Array>)
      .def("__eq__", &EqDunder<Int64Array>)
      .def("isna", &IsNA<Int64Array>)
      .def("take", &Take<Int64Array>, "indices"_a, "allow_fill"_a = false,
           "fill_value"_a = nb::none())
      .def("copy", &Copy<Int64Array>)
      .def("fillna", &FillNA<Int64Array>)
      .def("dropna", &DropNA<Int64Array>)
      .def("interpolate", &Interpolate<Int64Array>)
      .def("unique", &Unique<Int64Array>)
      .def("factorize", &Factorize<Int64Array>)
      .def("reshape", &Reshape<Int64Array>)
      .def("_pad_or_backfill", &PadOrBackfill<Int64Array>)
      .def("_from_sequence", &FromSequence<Int64Array>)
      .def("_from_factorized", &FromFactorized<Int64Array>)
      .def("tolist", &ToPyList<Int64Array>)
      .def("_concat_same_type", &ConcatSameType<Int64Array>)

      // integral-specific algorithms
      .def("sum", &Sum<Int64Array>)
      .def("min", &Min<Int64Array>)
      .def("max", &Max<Int64Array>);

  nb::class_<ExtensionDtype<Int64Array>>(m, "StringDtype")
      .def("__str__", &ExtensionDtype<Int64Array>::Str)
      .def_prop_ro("na_value", &ExtensionDtype<Int64Array>::NaValue)
      .def_prop_ro("kind", &ExtensionDtype<Int64Array>::Kind)
      .def_prop_ro("name", &ExtensionDtype<Int64Array>::Name)
      .def_prop_ro("is_numeric", &ExtensionDtype<Int64Array>::IsNumeric)
      .def_prop_ro("is_boolean", &ExtensionDtype<Int64Array>::IsBoolean)
      .def_prop_ro("_can_hold_na", &ExtensionDtype<Int64Array>::CanHoldNA)
      .def_prop_ro("_is_immutable", &ExtensionDtype<Int64Array>::IsImmutable);

  nb::class_<StringArray, ExtensionArray>(m, "StringArray")
      .def(nb::init<std::vector<std::optional<std::string_view>>>())
      .def("__len__", &LenDunder<StringArray>)
      .def_prop_ro("dtype", &Dtype<StringArray>)
      .def_prop_ro("nbytes", &Nbytes<StringArray>)
      .def_prop_ro("shape", &Shape<StringArray>)
      .def_prop_ro("size", &Size<StringArray>)
      .def_prop_ro("ndim", &NDim<StringArray>)
      .def("any", &Any<StringArray>)
      .def("all", &All<StringArray>)
      .def("__repr__", &ReprDunder<StringArray>)
      .def("__getitem__", &GetItemDunder<StringArray>)
      .def("__eq__", &EqDunder<StringArray>)
      .def("isna", &IsNA<StringArray>)
      .def("take", &Take<StringArray>, "indices"_a, "allow_fill"_a = false,
           "fill_value"_a = nb::none())
      .def("copy", &Copy<StringArray>)
      .def("fillna", &FillNA<StringArray>)
      .def("dropna", &DropNA<StringArray>)
      .def("interpolate", &Interpolate<StringArray>)
      .def("unique", &Unique<StringArray>)
      .def("factorize", &Factorize<StringArray>)
      .def("reshape", &Reshape<StringArray>)
      .def("_pad_or_backfill", &PadOrBackfill<StringArray>)
      .def("_from_sequence", &FromSequence<StringArray>)
      .def("_from_factorized", &FromFactorized<StringArray>)
      .def("tolist", &ToPyList<StringArray>)
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
      .def("isupper", &IsUpper)

      // these are all assorted functions that do nothing but are required
      // for structural subtyping with a pandas extension array
      // See https://github.com/pandas-dev/pandas/pull/57634
      .def_prop_ro(
          "_typ",
          []([[maybe_unused]] const StringArray &arr) { return "extension"; })
      .def_prop_ro("__pandas_priority__", &NoOp)
      .def_prop_ro("__hash__", &NoOp)
      .def("_from_scalars", &NoOp)
      .def("_from_sequence_of_strings", &NoOp)
      .def("__setitem__", &NoOp)
      .def("__iter__",
           [](const StringArray &arr) {
             return nb::make_iterator(nb::type<StringArray>(), "iterator",
                                      arr.begin(), arr.end());
           })
      .def("__contains__", &NoOp)
      .def("to_numpy", &NoOp)
      .def("astype", &NoOp)
      .def("_hasna", &NoOp)
      .def("_values_for_argsort", &NoOp)
      .def("argsort", &NoOp)
      .def("argmin", &NoOp)
      .def("argmax", &NoOp)
      .def("duplicated", &NoOp)
      .def("shift", &NoOp)
      .def("duplicated", &NoOp)
      .def("searchsorted", &NoOp)
      .def("equals", &NoOp)
      .def("isin", &NoOp)
      .def("_values_for_factorize", &NoOp)
      .def("repeat", &NoOp)
      .def("view", &NoOp)
      .def("_get_repr_footer", &NoOp)
      .def("_repr_2d", &NoOp)
      .def("_formatter", &NoOp)
      .def("transpose", &NoOp)
      .def_prop_ro("T", &NoOp)
      .def("ravel", &NoOp)
      .def_prop_ro("_can_hold_na", &NoOp)
      .def("_accumulate", &NoOp)
      .def("_reduce", &NoOp)
      .def("_values_for_json", &NoOp)
      .def("_hash_pandas_object", &NoOp)
      .def("_explode", &NoOp)
      .def("delete", &NoOp)
      .def("insert", &NoOp)
      .def("_putmask", &NoOp)
      .def("_where", &NoOp)
      .def("_fill_mask_inplace", &NoOp)
      .def("_rank", &NoOp)
      .def("_empty", &NoOp)
      .def("_quantile", &NoOp)
      .def("_mode", &NoOp)
      .def("__array_ufunc__", &NoOp)
      .def("map", &NoOp)
      .def("_groupby_op", &NoOp);

  nb::class_<ExtensionDtype<StringArray>>(m, "StringDtype")
      .def("__str__", &ExtensionDtype<StringArray>::Str)
      .def_prop_ro("na_value", &ExtensionDtype<StringArray>::NaValue)
      .def_prop_ro("kind", &ExtensionDtype<StringArray>::Kind)
      .def_prop_ro("name", &ExtensionDtype<StringArray>::Name)
      .def_prop_ro("names", &ExtensionDtype<StringArray>::Names)
      .def_prop_ro("_is_numeric", &ExtensionDtype<StringArray>::IsNumeric)
      .def_prop_ro("_is_boolean", &ExtensionDtype<StringArray>::IsBoolean)
      .def_prop_ro("_can_hold_na", &ExtensionDtype<StringArray>::CanHoldNA)
      .def_prop_ro("_is_immutable", &ExtensionDtype<StringArray>::IsImmutable)

      // hacks for extensiondtype protocol
      .def_prop_ro(
          "type",
          [](const ExtensionDtype<StringArray> &) { return nb::none(); })
      .def("construct_array_type",
           [](const ExtensionDtype<StringArray> &) { return; })
      .def("empty", [](const ExtensionDtype<StringArray> &) { return; })
      .def("construct_from_string",
           [](const ExtensionDtype<StringArray> &) { return; })
      .def("construct_from_string",
           [](const ExtensionDtype<StringArray> &) { return; })
      .def("is_dtype", [](const ExtensionDtype<StringArray> &) { return; })
      .def("_get_common_dtype",
           [](const ExtensionDtype<StringArray> &) { return; })
      .def_prop_ro("index_class",
                   [](const ExtensionDtype<StringArray> &) { return; })
      .def_prop_ro("_supports_2d",
                   [](const ExtensionDtype<StringArray> &) { return false; })
      .def_prop_ro("_can_fast_transpose",
                   [](const ExtensionDtype<StringArray> &) { return false; });
}
