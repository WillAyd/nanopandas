#pragma once

#include <nanoarrow/nanoarrow.hpp>
#include <nanobind/nanobind.h>
#include <optional>
#include <string>
#include <string_view>

class ExtensionArray {
public: // TODO: can we make these private / protected?
  nanoarrow::UniqueArrayView array_view_;

protected:
  nanoarrow::UniqueArray array_;
};

class BoolArray : public ExtensionArray {
public:
  using ScalarT = bool;
  static constexpr enum ArrowType ArrowT = NANOARROW_TYPE_BOOL;
  static constexpr const char Name[20] = "BoolArray";

  template <typename C> explicit BoolArray(const C &booleans) {
    // TODO: assert we only get bool or std::optional<bool>
    // static_assert(std::is_integral<typename C::value_type>::value ||
    //              std::is_same<typename C::value_type,
    //                           std::optional<bool>>::value);

    if (ArrowArrayInitFromType(array_.get(), NANOARROW_TYPE_BOOL)) {
      throw std::runtime_error("Unable to init BoolArray!");
    }

    if (ArrowArrayStartAppending(array_.get())) {
      throw std::runtime_error("Could not append to BoolArray!");
    }

    for (const auto &opt_boolean : booleans) {
      if (const auto &boolean = opt_boolean) {
        if (ArrowArrayAppendInt(array_.get(), *boolean)) {
          throw std::invalid_argument("Could not append integer: " +
                                      std::to_string(*boolean));
        }
      } else {
        if (ArrowArrayAppendNull(array_.get(), 1)) {
          throw std::invalid_argument("Failed to append null!");
        }
      }
    }

    struct ArrowError error;
    if (ArrowArrayFinishBuildingDefault(array_.get(), &error)) {
      throw std::runtime_error("Failed to finish building array!" +
                               std::string(error.message));
    }

    ArrowArrayViewInitFromType(array_view_.get(), NANOARROW_TYPE_INT64);
    if (ArrowArrayViewSetArray(array_view_.get(), array_.get(), &error)) {
      throw std::runtime_error("Failed to set array view!" +
                               std::string(error.message));
    }
  }

  BoolArray(nanoarrow::UniqueArray &&array) {
    array_ = std::move(array);
    ArrowArrayViewInitFromType(array_view_.get(), NANOARROW_TYPE_BOOL);
    struct ArrowError error;
    if (ArrowArrayViewSetArray(array_view_.get(), array_.get(), &error)) {
      throw std::runtime_error("Failed to set array view:" +
                               std::string(error.message));
    }
  }
};

class Int64Array : public ExtensionArray {
public:
  using ScalarT = int64_t;
  static constexpr enum ArrowType ArrowT = NANOARROW_TYPE_INT64;
  static constexpr const char Name[20] = "Int64Array";

  template <typename C> explicit Int64Array(const C &integers) {
    // TODO: assert we only get integral or std::optional<integral>
    // static_assert(std::is_integral<typename C::value_type>::value ||
    //              std::is_same<typename C::value_type,
    //                           std::optional<std::string_view>>::value);

    if (ArrowArrayInitFromType(array_.get(), NANOARROW_TYPE_INT64)) {
      throw std::runtime_error("Unable to init Int64Array!");
    };

    if (ArrowArrayStartAppending(array_.get())) {
      throw std::runtime_error("Could not append to Int64Array!");
    }

    for (const auto &opt_integer : integers) {
      if (const auto &integer = opt_integer) {
        if (ArrowArrayAppendInt(array_.get(), *integer)) {
          throw std::invalid_argument("Could not append integer: " +
                                      std::to_string(*integer));
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

    ArrowArrayViewInitFromType(array_view_.get(), NANOARROW_TYPE_INT64);
    if (ArrowArrayViewSetArray(array_view_.get(), array_.get(), nullptr)) {
      throw std::runtime_error("Failed to set array view!");
    }
  }

  Int64Array(nanoarrow::UniqueArray &&array) {
    array_ = std::move(array);
    ArrowArrayViewInitFromType(array_view_.get(), NANOARROW_TYPE_INT64);
    struct ArrowError error;
    if (ArrowArrayViewSetArray(array_view_.get(), array_.get(), &error)) {
      throw std::runtime_error("Failed to set array view:" +
                               std::string(error.message));
    }
  }
};

class StringArray : public ExtensionArray {
public:
  using ScalarT = std::string_view;
  static constexpr enum ArrowType ArrowT = NANOARROW_TYPE_LARGE_STRING;
  static constexpr const char Name[20] = "StringArray";

  template <typename C> explicit StringArray(const C &strings) {
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

    if (ArrowArrayReserve(array_.get(), strings.size())) {
      throw std::runtime_error("Unable to reserve array!");
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

  StringArray(nanoarrow::UniqueArray &&array) {
    array_ = std::move(array);
    ArrowArrayViewInitFromType(array_view_.get(), NANOARROW_TYPE_LARGE_STRING);
    struct ArrowError error;
    if (ArrowArrayViewSetArray(array_view_.get(), array_.get(), &error)) {
      throw std::runtime_error("Failed to set array view:" +
                               std::string(error.message));
    }
  }
};
