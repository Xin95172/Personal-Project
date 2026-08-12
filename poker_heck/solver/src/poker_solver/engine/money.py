"""以 BB 輸入、以整數 units 結算的金額工具。"""

from decimal import Decimal, InvalidOperation


UNITS_PER_BB = 100


def bb_to_units(value: int | float | str | Decimal) -> int:
    """將 BB 數值轉成精確整數 units；最小精度為 0.01 BB。"""
    try:
        units = Decimal(str(value)) * UNITS_PER_BB
    except InvalidOperation as error:
        raise ValueError(f"invalid BB amount: {value!r}") from error
    if units != units.to_integral_value():
        raise ValueError("BB amounts must be multiples of 0.01")
    return int(units)


def format_bb(units: int) -> str:
    """將整數 units 顯示為精簡的 BB 文字。"""
    return f"{Decimal(units) / UNITS_PER_BB:g} BB"
