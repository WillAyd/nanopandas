#include <limits>
#include <optional>
#include <stdint.h>

template <typename T> std::optional<typename T::ScalarT> Sum(const T &self) {
  const auto n = self.array_view_->length;
  if ((n == 0) || (self.array_view_->null_count == n)) {
    return std::nullopt;
  }

  typename T::ScalarT result = 0;
  for (int64_t i = 0; i < n; i++) {
    if (ArrowArrayViewIsNull(self.array_view_.get(), i)) {
      continue;
    } else {
      result += ArrowArrayViewGetIntUnsafe(self.array_view_.get(), i);
    }
  }

  return result;
}

template <typename T> std::optional<typename T::ScalarT> Min(const T &self) {
  const auto n = self.array_view_->length;
  if ((n == 0) || (self.array_view_->null_count == n)) {
    return std::nullopt;
  }

  typename T::ScalarT result = std::numeric_limits<typename T::ScalarT>::max();
  for (int64_t i = 0; i < n; i++) {
    if (ArrowArrayViewIsNull(self.array_view_.get(), i)) {
      continue;
    } else {
      const int64_t value =
          ArrowArrayViewGetIntUnsafe(self.array_view_.get(), i);
      if (value < result) {
        result = value;
      }
    }
  }

  return result;
}

template <typename T> std::optional<typename T::ScalarT> Max(const T &self) {
  const auto n = self.array_view_->length;
  if ((n == 0) || (self.array_view_->null_count == n)) {
    return std::nullopt;
  }

  typename T::ScalarT result = std::numeric_limits<typename T::ScalarT>::min();
  for (int64_t i = 0; i < n; i++) {
    if (ArrowArrayViewIsNull(self.array_view_.get(), i)) {
      continue;
    } else {
      const int64_t value =
          ArrowArrayViewGetIntUnsafe(self.array_view_.get(), i);
      if (value > result) {
        result = value;
      }
    }
  }

  return result;
}
