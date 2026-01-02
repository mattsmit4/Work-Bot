"""Configuration for ST-Bot."""

from config.synonyms import SYNONYMS, expand_synonyms
from config.patterns import (
    LENGTH_PATTERN,
    SKU_PATTERN,
    CONNECTOR_DETECT,
    GREETING_PATTERNS,
    FAREWELL_PATTERNS,
    INSTALL_PATTERNS,
    DAISY_CHAIN_PATTERNS,
    extract_lengths,
    has_pattern,
)

__all__ = [
    "SYNONYMS",
    "expand_synonyms",
    "LENGTH_PATTERN",
    "SKU_PATTERN",
    "CONNECTOR_DETECT",
    "GREETING_PATTERNS",
    "FAREWELL_PATTERNS",
    "INSTALL_PATTERNS",
    "DAISY_CHAIN_PATTERNS",
    "extract_lengths",
    "has_pattern",
]
