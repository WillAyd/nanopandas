#include <algorithm>
#include <set>

#include <nanoarrow/nanoarrow_types.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>
#include <utf8proc.h>

#include <nanoarrow/nanoarrow.hpp>

static_assert(std::is_same_v<std::uint8_t, char> ||
                  std::is_same_v<std::uint8_t, unsigned char>,
              "uint8_t must be a typedef for char or unsigned char");

namespace nb = nanobind;

static utf8proc_int32_t utf8proc_toupper_wrapper(utf8proc_int32_t codepoint,
                                                 void *data) {
  return utf8proc_toupper(codepoint);
}

static utf8proc_int32_t utf8proc_tolower_wrapper(utf8proc_int32_t codepoint,
                                                 void *data) {
  return utf8proc_tolower(codepoint);
}

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

  StringArray cat() { throw std::runtime_error("not implemented"); }
  std::vector<std::vector<std::string>> split() {
    throw std::runtime_error("not implemented");
  }
  std::vector<std::vector<std::string>> rsplit() {
    throw std::runtime_error("not implemented");
  }
  std::vector<char> get(size_t position) {
    throw std::runtime_error("not implemented");
  }
  std::vector<std::vector<std::string>> join() {
    throw std::runtime_error("not implemented");
  }
  StringArray get_dummies() { throw std::runtime_error("not implemented"); }
  std::vector<bool> contains() { throw std::runtime_error("not implemented"); }
  StringArray replace() { throw std::runtime_error("not implemented"); }
  StringArray removeprefix(const std::string &prefix) {
    throw std::runtime_error("not implemented");
  }
  StringArray removesuffix(const std::string &suffix) {
    throw std::runtime_error("not implemented");
  }
  StringArray repeat(size_t repeats) {
    throw std::runtime_error("not implemented");
  }
  StringArray pad(size_t width, const std::string &side, char fillchar) {
    throw std::runtime_error("not implemented");
  }
  StringArray center(size_t width, char fillchar) {
    throw std::runtime_error("not implemented");
  }
  StringArray ljust(size_t width, char fillchar) {
    throw std::runtime_error("not implemented");
  }
  StringArray rjust(size_t width, char fillchar) {
    throw std::runtime_error("not implemented");
  }
  StringArray zfill(size_t width) {
    throw std::runtime_error("not implemented");
  }
  StringArray wrap(size_t width) {
    throw std::runtime_error("not implemented");
  }
  StringArray slice(ssize_t start, ssize_t stop, ssize_t step) {
    throw std::runtime_error("not implemented");
  }
  StringArray slice_replace(ssize_t start, ssize_t stop, char repl) {
    throw std::runtime_error("not implemented");
  }
  std::vector<size_t> count() { throw std::runtime_error("not implemented"); }
  std::vector<bool> startswith() {
    throw std::runtime_error("not implemented");
  }
  std::vector<bool> endswith() { throw std::runtime_error("not implemented"); }
  std::vector<std::vector<std::string>> findall() {
    throw std::runtime_error("not implemented");
  }
  std::vector<std::vector<std::string>> match() {
    throw std::runtime_error("not implemented");
  }
  std::vector<std::vector<std::string>> extract() {
    throw std::runtime_error("not implemented");
  }
  std::vector<std::vector<std::string>> extractall() {
    throw std::runtime_error("not implemented");
  }
  std::vector<size_t> len() { throw std::runtime_error("not implemented"); }
  StringArray strip() { throw std::runtime_error("not implemented"); }
  StringArray rstrip() { throw std::runtime_error("not implemented"); }
  StringArray lstrip() { throw std::runtime_error("not implemented"); }
  std::vector<std::vector<std::string>> partition() {
    throw std::runtime_error("not implemented");
  }
  std::vector<std::vector<std::string>> rpartition() {
    throw std::runtime_error("not implemented");
  }

  StringArray lower() {
    std::vector<std::optional<std::string>> result;
    const auto n = array_->length;

    result.reserve(n);
    for (int64_t i = 0; i < n; i++) {
      if (ArrowArrayViewIsNull(array_view_.get(), i)) {
        result.push_back(std::nullopt);
      } else {
        const auto sv = ArrowArrayViewGetStringUnsafe(array_view_.get(), i);
        unsigned char *dst;

        const ssize_t nbytes = utf8proc_map_custom(
            reinterpret_cast<const uint8_t *>(sv.data), sv.size_bytes, &dst,
            UTF8PROC_STABLE, utf8proc_tolower_wrapper, NULL);
        if (nbytes < 0) {
          throw std::runtime_error("error occurred converting tolower!");
        }

        // utf8proc and std::string but malloc on the heap; maybe we could avoid
        // a second malloc with a StringArray constructor for raw C pointers
        std::string converted{};
        converted.resize(nbytes);
        memcpy(&converted[0], dst, nbytes);
        free(dst);

        result.push_back(std::move(converted));
      }
    }

    return StringArray{result};
  }

  StringArray casefold() { throw std::runtime_error("not implemented"); }

  StringArray upper() {
    std::vector<std::optional<std::string>> result;
    const auto n = array_->length;

    result.reserve(n);
    for (int64_t i = 0; i < n; i++) {
      if (ArrowArrayViewIsNull(array_view_.get(), i)) {
        result.push_back(std::nullopt);
      } else {
        const auto sv = ArrowArrayViewGetStringUnsafe(array_view_.get(), i);
        unsigned char *dst;

        const ssize_t nbytes = utf8proc_map_custom(
            reinterpret_cast<const uint8_t *>(sv.data), sv.size_bytes, &dst,
            UTF8PROC_STABLE, utf8proc_toupper_wrapper, NULL);
        if (nbytes < 0) {
          throw std::runtime_error("error occurred converting toupper!");
        }

        // utf8proc and std::string but malloc on the heap; maybe we could avoid
        // a second malloc with a StringArray constructor for raw C pointers
        std::string converted{};
        converted.resize(nbytes);
        memcpy(&converted[0], dst, nbytes);
        free(dst);

        result.push_back(std::move(converted));
      }
    }

    return StringArray{result};
  }

  std::vector<size_t> find(const std::string &string, size_t start,
                           size_t end) {
    throw std::runtime_error("not implemented");
  }
  std::vector<size_t> rfind(const std::string &string, size_t start,
                            size_t end) {
    throw std::runtime_error("not implemented");
  }

  std::vector<size_t> index(const std::string &string, size_t start,
                            size_t end) {
    throw std::runtime_error("not implemented");
  }
  std::vector<size_t> rindex(const std::string &string, size_t start,
                             size_t end) {
    throw std::runtime_error("not implemented");
  }

  StringArray capitalize() {
    std::vector<std::optional<std::string>> result;
    const auto n = array_->length;

    result.reserve(n);
    for (int64_t i = 0; i < n; i++) {
      if (ArrowArrayViewIsNull(array_view_.get(), i)) {
        result.push_back(std::nullopt);
      } else {
        const auto sv = ArrowArrayViewGetStringUnsafe(array_view_.get(), i);
        std::vector<utf8proc_uint8_t> dst;
        dst.reserve(static_cast<size_t>(sv.size_bytes));

        size_t bytes_read = 0;
        size_t bytes_rem;
        while ((bytes_rem = static_cast<size_t>(sv.size_bytes) - bytes_read) >
               0) {
          utf8proc_int32_t codepoint;
          size_t codepoint_bytes = utf8proc_iterate(
              reinterpret_cast<const utf8proc_uint8_t *>(sv.data + bytes_read),
              bytes_rem, &codepoint);

          if (bytes_read ==
              0) { // maybe not the best impl, but easier to template with upper
            codepoint = utf8proc_toupper(codepoint);
          }
          bytes_read += codepoint_bytes;

          std::array<utf8proc_uint8_t, 4> encoded;
          utf8proc_encode_char(codepoint, encoded.data());
          dst.insert(dst.end(), encoded.begin(),
                     encoded.begin() + codepoint_bytes);
        }
        result.push_back(std::string{dst.begin(), dst.end()});
      }
    }

    return StringArray{result};
  }

  StringArray swapcase() { throw std::runtime_error("not implemented"); }
  StringArray normalize() { throw std::runtime_error("not implemented"); }
  StringArray translate() { throw std::runtime_error("not implemented"); }
  std::vector<bool> isalnum() { throw std::runtime_error("not implemented"); }
  std::vector<bool> isalpha() { throw std::runtime_error("not implemented"); }
  std::vector<bool> isdigit() { throw std::runtime_error("not implemented"); }
  std::vector<bool> isspace() { throw std::runtime_error("not implemented"); }
  std::vector<bool> islower() { throw std::runtime_error("not implemented"); }
  std::vector<bool> isupper() { throw std::runtime_error("not implemented"); }
  std::vector<bool> istitle() { throw std::runtime_error("not implemented"); }
  std::vector<bool> isnumeric() { throw std::runtime_error("not implemented"); }
  std::vector<bool> isdecimal() { throw std::runtime_error("not implemented"); }

  // Misc extras
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

// try to match all pandas methods
// https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html#method-summary
NB_MODULE(nanopandas, m) {
  nb::class_<StringArray>(m, "StringArray")
      .def(nb::init<std::vector<std::optional<std::string_view>>>())
      .def("cat", &StringArray::cat)
      .def("split", &StringArray::split)
      .def("rsplit", &StringArray::rsplit)
      .def("get", &StringArray::get)
      .def("join", &StringArray::join)
      .def("get_dummies", &StringArray::get_dummies)
      .def("contains", &StringArray::contains)
      .def("replace", &StringArray::replace)
      .def("removeprefix", &StringArray::removeprefix)
      .def("removesuffix", &StringArray::removesuffix)
      .def("repeat", &StringArray::repeat)
      .def("pad", &StringArray::pad)
      .def("center", &StringArray::center)
      .def("ljust", &StringArray::ljust)
      .def("rjust", &StringArray::rjust)
      .def("zfill", &StringArray::zfill)
      .def("wrap", &StringArray::wrap)
      .def("slice", &StringArray::slice)
      .def("slice_replace", &StringArray::slice_replace)
      .def("count", &StringArray::count)
      .def("startswith", &StringArray::startswith)
      .def("endswith", &StringArray::endswith)
      .def("findall", &StringArray::findall)
      .def("match", &StringArray::match)
      .def("extract", &StringArray::extract)
      .def("extractall", &StringArray::extractall)
      .def("len", &StringArray::len)
      .def("strip", &StringArray::strip)
      .def("rstrip", &StringArray::rstrip)
      .def("lstrip", &StringArray::lstrip)
      .def("partition", &StringArray::partition)
      .def("rpartition", &StringArray::rpartition)
      .def("lower", &StringArray::lower)
      .def("casefold", &StringArray::casefold)
      .def("upper", &StringArray::upper)
      .def("find", &StringArray::find)
      .def("rfind", &StringArray::rfind)
      .def("index", &StringArray::rindex)
      .def("capitalize", &StringArray::capitalize)
      .def("swapcase", &StringArray::swapcase)
      .def("normalize", &StringArray::normalize)
      .def("translate", &StringArray::translate)
      .def("isalnum", &StringArray::isalnum)
      .def("isalpha", &StringArray::isalpha)
      .def("isdigit", &StringArray::isdigit)
      .def("isspace", &StringArray::isspace)
      .def("islower", &StringArray::islower)
      .def("isupper", &StringArray::isupper)
      .def("istitle", &StringArray::istitle)
      .def("isnumeric", &StringArray::isnumeric)
      .def("isdecimal", &StringArray::isdecimal)

      // some extras that may be useful
      .def_prop_ro("size", &StringArray::size)
      .def_prop_ro("nbytes", &StringArray::nbytes)
      .def_prop_ro("dtype", &StringArray::dtype)
      .def("any", &StringArray::any)
      .def("all", &StringArray::all)
      .def("unique", &StringArray::unique)
      .def("to_pylist", &StringArray::to_pylist);
}
