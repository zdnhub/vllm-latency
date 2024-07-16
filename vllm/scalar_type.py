from enum import Enum

from ._custom_classes import ScalarType


# Mirrors enum in `core/scalar_type.hpp`
class NanRepr(Enum):
    IEEE_754 = 0  # nans are: Exp all 1s, mantissa not all 0s
    NONE = 1  # nans are not supported
    EXTD_RANGE_MAX_MIN = 2  # nans are: Exp all 1s, mantissa all 1s


class scalar_types:
    s4 = ScalarType.s(4, None)
    u4 = ScalarType.u(4, None)
    s8 = ScalarType.s(8, None)
    u8 = ScalarType.u(8, None)
    fE4M3fn = ScalarType.fn(4, 3, True, NanRepr.EXTD_RANGE_MAX_MIN.value)
    fE5M2 = ScalarType.f(5, 2)
    fE8M7 = ScalarType.f(8, 7)
    fE5M10 = ScalarType.f(5, 10)

    # "gptq" types
    u4b8 = ScalarType.u(4, 8)
    u8b128 = ScalarType.u(8, 128)

    # colloquial names
    bfloat16 = fE8M7
    float16 = fE5M10
