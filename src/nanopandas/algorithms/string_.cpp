#include "string_.hpp"

#include <array>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

StringArray Lower(const StringArray &self) {
  std::vector<std::optional<std::string>> result;
  const auto n = self.array_view_->length;

  result.reserve(n);
  for (int64_t i = 0; i < n; i++) {
    if (ArrowArrayViewIsNull(self.array_view_.get(), i)) {
      result.push_back(std::nullopt);
    } else {
      const auto sv = ArrowArrayViewGetStringUnsafe(self.array_view_.get(), i);
      unsigned char *dst;

      constexpr auto lambda = [](utf8proc_int32_t codepoint,
                                 [[maybe_unused]] void *) -> utf8proc_int32_t {
        return utf8proc_tolower(codepoint);
      };

      const utf8proc_ssize_t nbytes = utf8proc_map_custom(
          reinterpret_cast<const uint8_t *>(sv.data), sv.size_bytes, &dst,
          UTF8PROC_STABLE, lambda, NULL);
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

StringArray Upper(const StringArray &self) {
  nanoarrow::UniqueArray result;
  if (ArrowArrayInitFromType(result.get(), NANOARROW_TYPE_LARGE_STRING)) {
    throw std::runtime_error("Unable to init large string array!");
  }
  const auto n = self.array_view_->length;

  if (ArrowArrayStartAppending(result.get())) {
    throw std::runtime_error("Could not start appending");
  }

  if (ArrowArrayReserve(result.get(), n)) {
    throw std::runtime_error("Unable to reserve array!");
  }

  for (int64_t i = 0; i < n; i++) {
    if (ArrowArrayViewIsNull(self.array_view_.get(), i)) {
      if (ArrowArrayAppendNull(result.get(), 1)) {
        throw std::runtime_error("failed to append null!");
      }
    } else {
      const auto sv = ArrowArrayViewGetStringUnsafe(self.array_view_.get(), i);
      unsigned char *dst;

      constexpr auto lambda = [](utf8proc_int32_t codepoint,
                                 [[maybe_unused]] void *) -> utf8proc_int32_t {
        return utf8proc_toupper(codepoint);
      };

      const utf8proc_ssize_t nbytes = utf8proc_map_custom(
          reinterpret_cast<const uint8_t *>(sv.data), sv.size_bytes, &dst,
          UTF8PROC_STABLE, lambda, NULL);
      if (nbytes < 0) {
        throw std::runtime_error("error occurred converting toupper!");
      }

      struct ArrowStringView dest_sv = {reinterpret_cast<const char *>(dst),
                                        nbytes};

      if (ArrowArrayAppendString(result.get(), dest_sv)) {
        throw std::runtime_error("failed to append string");
      }
      free(dst);
    }
  }

  struct ArrowError error;
  if (ArrowArrayFinishBuildingDefault(result.get(), &error)) {
    throw std::runtime_error("Failed to finish building: " +
                             std::string(error.message));
  }

  return StringArray{std::move(result)};
}

StringArray Capitalize(const StringArray &self) {
  std::vector<std::optional<std::string>> result;
  const auto n = self.array_view_->length;

  result.reserve(n);
  for (int64_t i = 0; i < n; i++) {
    if (ArrowArrayViewIsNull(self.array_view_.get(), i)) {
      result.push_back(std::nullopt);
    } else {
      const auto sv = ArrowArrayViewGetStringUnsafe(self.array_view_.get(), i);
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

  return StringArray{std::move(result)};
}

/*
 * utf8proc applications
 */
static BoolArray
ApplyUtf8ProcFunction(const struct ArrowArrayView *array_view,
                      const std::function<bool(utf8proc_int32_t)> &func) {
  nanoarrow::UniqueArray result;
  if (ArrowArrayInitFromType(result.get(), NANOARROW_TYPE_BOOL)) {
    throw std::runtime_error("Unable to init bool array!");
  }
  const auto n = array_view->length;

  if (ArrowArrayStartAppending(result.get())) {
    throw std::runtime_error("Could not start appending");
  }

  if (ArrowArrayReserve(result.get(), n)) {
    throw std::runtime_error("Unable to reserve array!");
  }

  for (int64_t i = 0; i < n; i++) {
    if (ArrowArrayViewIsNull(array_view, i)) {
      if (ArrowArrayAppendNull(result.get(), i)) {
        throw std::runtime_error("failed to append null!");
      }
    } else {
      const auto sv = ArrowArrayViewGetStringUnsafe(array_view, i);

      size_t bytes_read = 0;
      size_t bytes_rem;

      bool is_true = true;
      while ((bytes_rem = static_cast<size_t>(sv.size_bytes) - bytes_read) >
             0) {
        utf8proc_int32_t codepoint;
        size_t codepoint_bytes = utf8proc_iterate(
            reinterpret_cast<const utf8proc_uint8_t *>(sv.data + bytes_read),
            bytes_rem, &codepoint);

        is_true = func(codepoint);
        if (!is_true) {
          break;
        }

        bytes_read += codepoint_bytes;
      }

      if (ArrowArrayAppendInt(result.get(), is_true)) {
        throw std::runtime_error("failed to append bool!");
      }
    }
  }

  struct ArrowError error;
  if (ArrowArrayFinishBuildingDefault(result.get(), &error)) {
    throw std::runtime_error("Failed to finish building: " +
                             std::string(error.message));
  }

  return result;
}

BoolArray IsAlnum(const StringArray &self) {
  constexpr auto lambda = [](utf8proc_int32_t codepoint) {
    const auto category = utf8proc_category(codepoint);
    switch (category) {
    case UTF8PROC_CATEGORY_LU:
    case UTF8PROC_CATEGORY_LL:
    case UTF8PROC_CATEGORY_LT:
    case UTF8PROC_CATEGORY_LM:
    case UTF8PROC_CATEGORY_LO:
    case UTF8PROC_CATEGORY_ND:
    case UTF8PROC_CATEGORY_NL:
    case UTF8PROC_CATEGORY_NO:
      return true;
    default:
      return false;
    }
  };

  return ApplyUtf8ProcFunction(self.array_view_.get(), lambda);
}

BoolArray IsAlpha(const StringArray &self) {
  constexpr auto lambda = [](utf8proc_int32_t codepoint) {
    const auto category = utf8proc_category(codepoint);
    switch (category) {
    case UTF8PROC_CATEGORY_LU:
    case UTF8PROC_CATEGORY_LL:
    case UTF8PROC_CATEGORY_LT:
    case UTF8PROC_CATEGORY_LM:
    case UTF8PROC_CATEGORY_LO:
      return true;
    default:
      return false;
    }
  };

  return ApplyUtf8ProcFunction(self.array_view_.get(), lambda);
}

BoolArray IsDigit(const StringArray &self) {
  constexpr auto lambda = [](utf8proc_int32_t codepoint) {
    const auto category = utf8proc_category(codepoint);
    switch (category) {
    case UTF8PROC_CATEGORY_ND:
    case UTF8PROC_CATEGORY_NL:
    case UTF8PROC_CATEGORY_NO:
      return true;
    default:
      return false;
    }
  };

  return ApplyUtf8ProcFunction(self.array_view_.get(), lambda);
}

BoolArray IsSpace(const StringArray &self) {
  constexpr auto lambda = [](utf8proc_int32_t codepoint) {
    const auto category = utf8proc_category(codepoint);
    switch (category) {
    case UTF8PROC_CATEGORY_ZS:
    case UTF8PROC_CATEGORY_ZL:
    case UTF8PROC_CATEGORY_ZP:
      return true;
    default:
      return false;
    }
  };

  return ApplyUtf8ProcFunction(self.array_view_.get(), lambda);
}

BoolArray IsLower(const StringArray &self) {
  constexpr auto lambda = [](utf8proc_int32_t codepoint) {
    return utf8proc_islower(codepoint);
  };

  return ApplyUtf8ProcFunction(self.array_view_.get(), lambda);
}

BoolArray IsUpper(const StringArray &self) {
  constexpr auto lambda = [](utf8proc_int32_t codepoint) {
    return utf8proc_isupper(codepoint);
  };

  return ApplyUtf8ProcFunction(self.array_view_.get(), lambda);
}
