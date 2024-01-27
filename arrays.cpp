#include <algorithm>
#include <set>

#include <nanoarrow/nanoarrow.h>
#include <nanoarrow/nanoarrow_types.h>
#include <utf8proc.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <nanoarrow/nanoarrow.hpp>


namespace nb = nanobind;

class StringArray {
public:
  template <typename C> StringArray(const C &strings) {
    static_assert(std::is_same<typename C::value_type,
                               std::optional<std::string>>::value ||
                  std::is_same<typename C::value_type,
                               std::optional<std::string_view>>::value);
    if (ArrowArrayInitFromType(array_.get(), NANOARROW_TYPE_LARGE_STRING)) {
      throw std::runtime_error("Unable to init StringArray!");
    };

    if (ArrowArrayStartAppending(array_.get())) {
      throw std::runtime_error("Could not append to StringArray!");
    }

    for (const auto &opt_str : strings) {
      if (const auto &str = opt_str) {
        struct ArrowStringView sv = {str->data(),
                                     static_cast<int64_t>(str->size())};
        if (ArrowArrayAppendString(array_.get(), sv)) {
          throw std::invalid_argument("Could not append string: " +
                                      std::string(*str));
        }
      } else {
        if (ArrowArrayAppendNull(array_.get(), 1)) {
          throw std::invalid_argument("Failed to append null!");
        }
      }
    }

    if (ArrowArrayFinishBuildingDefault(array_.get(), nullptr)) {
      throw std::runtime_error("Failed to finish building array!");
    }

    ArrowArrayViewInitFromType(array_view_.get(), NANOARROW_TYPE_LARGE_STRING);
    if (ArrowArrayViewSetArray(array_view_.get(), array_.get(), nullptr)) {
      throw std::runtime_error("Failed to set array view!");
    }
  }

  int64_t size() { return array_->length; }

  int64_t nbytes() {
    struct ArrowBuffer *data_buffer = ArrowArrayBuffer(array_.get(), 1);
    return data_buffer->size_bytes;
  }

  std::string dtype() { return std::string("large_string[nanoarrow]"); }

  bool any() { return array_->length > array_->null_count; }
  bool all() { return array_->null_count == 0; }

  StringArray unique() {
    // TODO: this should never be optional in unique; simply required by current
    // constructor but there is probably a smarter way to template that
    std::set<std::optional<std::string>> result;
    const auto n = array_->length;

    for (int64_t i = 0; i < n; i++) {
      if (ArrowArrayViewIsNull(array_view_.get(), i)) {
        continue;
      }

      const auto sv = ArrowArrayViewGetStringUnsafe(array_view_.get(), i);
      const std::string value{sv.data, static_cast<size_t>(sv.size_bytes)};
      result.insert(value);
    }

    return StringArray{result};
  }

  StringArray upper() {
    std::vector<std::optional<std::string>> result;
    const auto n = array_->length;

    result.reserve(n);
    for (int64_t i = 0; i < n; i++) {
      if (ArrowArrayViewIsNull(array_view_.get(), i)) {
        result.push_back(std::nullopt);
      } else {
        const auto sv = ArrowArrayViewGetStringUnsafe(array_view_.get(), i);
        std::string value{sv.data, static_cast<size_t>(sv.size_bytes)};
        std::transform(value.begin(), value.end(), value.begin(), utf8proc_toupper);
        result.push_back(value);
      }
    }

    return StringArray{result};
  }

  std::vector<std::optional<std::string_view>> to_pylist() {
    const auto n = array_->length;

    std::vector<std::optional<std::string_view>> result;
    result.reserve(n);
    for (int64_t i = 0; i < n; i++) {
      if (ArrowArrayViewIsNull(array_view_.get(), i)) {
        result.push_back(std::nullopt);
      } else {
        const auto sv = ArrowArrayViewGetStringUnsafe(array_view_.get(), i);
        const std::string_view value{sv.data,
                                     static_cast<size_t>(sv.size_bytes)};
        result.push_back(value);
      }
    }

    return result;
  }

private:
  nanoarrow::UniqueArray array_;
  nanoarrow::UniqueArrayView array_view_;
};

NB_MODULE(nanopandas, m) {
  nb::class_<StringArray>(m, "StringArray")
      .def(nb::init<std::vector<std::optional<std::string_view>>>())
      .def_prop_ro("size", &StringArray::size)
      .def_prop_ro("nbytes", &StringArray::nbytes)
      .def_prop_ro("dtype", &StringArray::dtype)
      .def("any", &StringArray::any)
      .def("all", &StringArray::all)
      .def("unique", &StringArray::unique)
      .def("upper", &StringArray::upper)
      .def("to_pylist", &StringArray::to_pylist);
}
