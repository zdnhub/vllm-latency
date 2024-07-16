from typing import Optional, Union
from .scalar_type import NanRepr

class ScalarType:
    """
    ScalarType can represent a wide range of floating point and integer types,
    in particular it can be used to represent sub-byte data types (something
    that torch.dtype currently does not support).
    """

    def __init__(self, exponent: int, mantissa: int, bias: int,
                 signed: bool) -> None:
        ...

    @classmethod
    def s(cls, size_bits: int, zero_point: Optional[int]) -> ScalarType:
        "Create a signed integer scalar type (size_bits includes the sign-bit)."
        ...

    @classmethod
    def u(cls, size_bits: int, zero_point: Optional[int]) -> ScalarType:
        """Create a signed integer scalar type."""
        ...

    @classmethod
    def f(cls, exponent: int, mantissa: int) -> ScalarType:
        ...
        
    @classmethod
    def fn(cls, exponent: int, mantissa: int, finite_values_only: bool, 
           nan_repr: int) -> ScalarType:
        ...

    @property
    def mantissa(self) -> int:
        ...

    @property
    def exponent(self) -> int:
        ...

    @property
    def bias(self) -> int:
        ...

    @property
    def size_bits(self) -> int:
        ...
        
    @property
    def nan_repr(self) -> NanRepr:
        ...

    def max(self) -> Union[int, float]:
        ...

    def min(self) -> Union[int, float]:
        ...

    def is_signed(self) -> bool:
        ...

    def is_integer(self) -> bool:
        ...

    def is_floating_point(self) -> bool:
        ...
        
    def is_ieee_754(self) -> bool:
        ...

    def has_nans(self) -> bool:
        ...
    
    def has_infs(self) -> bool:
        ...

    def has_zero_points(self) -> bool:
        ...

    def __eq__(self, value: object) -> bool:
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...
