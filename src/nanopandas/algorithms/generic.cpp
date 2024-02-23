#include "generic.hpp"

template <> const char *Dtype([[maybe_unused]] const StringArray &self) {
  return "string[arrow]";
}

template <> const char *Dtype([[maybe_unused]] const BoolArray &self) {
  return "boolean[arrow]";
}

template <> const char *Dtype([[maybe_unused]] const Int64Array &self) {
  return "int64[arrow]";
}

template <>
BoolArray ConcatSameType(const BoolArray &self, const BoolArray &other) {
  nanoarrow::UniqueArray result;
  if (ArrowArrayInitFromType(result.get(), NANOARROW_TYPE_BOOL)) {
    throw std::runtime_error("Unable to init boolean array!");
  }

  // TODO: check for overflow
  const auto n = self.array_view_->length + other.array_view_->length;

  if (ArrowArrayStartAppending(result.get())) {
    throw std::runtime_error("Could not start appending");
  }

  if (ArrowArrayReserve(result.get(), n)) {
    throw std::runtime_error("Unable to reserve array!");
  }

  for (int64_t idx = 0; idx < self.array_view_.get()->length; idx++) {
    if (ArrowArrayViewIsNull(self.array_view_.get(), idx)) {
      if (ArrowArrayAppendNull(result.get(), 1)) {
        throw std::runtime_error("failed to append null!");
      }
    } else {
      const auto value =
          ArrowArrayViewGetIntUnsafe(self.array_view_.get(), idx);
      if (ArrowArrayAppendInt(result.get(), value)) {
        throw std::runtime_error("failed to append value!");
      }
    }
  }

  for (int64_t idx = 0; idx < other.array_view_->length; idx++) {
    if (ArrowArrayViewIsNull(other.array_view_.get(), idx)) {
      if (ArrowArrayAppendNull(result.get(), 1)) {
        throw std::runtime_error("failed to append null!");
      }
    } else {
      const auto value =
          ArrowArrayViewGetIntUnsafe(other.array_view_.get(), idx);
      if (ArrowArrayAppendInt(result.get(), value)) {
        throw std::runtime_error("failed to append value!");
      }
    }
  }

  struct ArrowError error;
  if (ArrowArrayFinishBuildingDefault(result.get(), &error)) {
    throw std::runtime_error("Failed to finish building: " +
                             std::string(error.message));
  }

  return BoolArray(std::move(result));
}

template <>
StringArray ConcatSameType(const StringArray &self, const StringArray &other) {
  nanoarrow::UniqueArray result;
  if (ArrowArrayInitFromType(result.get(), NANOARROW_TYPE_LARGE_STRING)) {
    throw std::runtime_error("Unable to init large string array!");
  }

  // TODO: check for overflow
  const auto n = self.array_view_->length + other.array_view_->length;

  if (ArrowArrayStartAppending(result.get())) {
    throw std::runtime_error("Could not start appending");
  }

  if (ArrowArrayReserve(result.get(), n)) {
    throw std::runtime_error("Unable to reserve array!");
  }

  for (int64_t idx = 0; idx < self.array_view_.get()->length; idx++) {
    if (ArrowArrayViewIsNull(self.array_view_.get(), idx)) {
      if (ArrowArrayAppendNull(result.get(), 1)) {
        throw std::runtime_error("failed to append null!");
      }
    } else {
      const auto sv =
          ArrowArrayViewGetStringUnsafe(self.array_view_.get(), idx);
      if (ArrowArrayAppendString(result.get(), sv)) {
        throw std::runtime_error("failed to append string!");
      }
    }
  }

  for (int64_t idx = 0; idx < other.array_view_->length; idx++) {
    if (ArrowArrayViewIsNull(other.array_view_.get(), idx)) {
      if (ArrowArrayAppendNull(result.get(), 1)) {
        throw std::runtime_error("failed to append null!");
      }
    } else {
      const auto sv =
          ArrowArrayViewGetStringUnsafe(other.array_view_.get(), idx);
      if (ArrowArrayAppendString(result.get(), sv)) {
        throw std::runtime_error("failed to append string!");
      }
    }
  }

  struct ArrowError error;
  if (ArrowArrayFinishBuildingDefault(result.get(), &error)) {
    throw std::runtime_error("Failed to finish building: " +
                             std::string(error.message));
  }

  return StringArray(std::move(result));
}
