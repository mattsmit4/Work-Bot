"""
Synonym and abbreviation mappings for query understanding.

These mappings help normalize user queries before processing.
"""

# Connector abbreviations and variations
CONNECTOR_SYNONYMS = {
    "dp": "displayport",
    "tb": "thunderbolt",
    "tb3": "thunderbolt 3",
    "tb4": "thunderbolt 4",
    "usb c": "usb-c",
    "usbc": "usb-c",
    "type c": "usb-c",
    "type-c": "usb-c",
    "usb a": "usb-a",
    "usba": "usb-a",
    "type a": "usb-a",
    "type-a": "usb-a",
}

# Common typos and misspellings
TYPO_CORRECTIONS = {
    # HDMI typos
    "hdim": "hdmi",
    "hmdi": "hdmi",
    "hdm": "hdmi",
    "hmi": "hdmi",
    # DisplayPort typos
    "displayprot": "displayport",
    "dispayport": "displayport",
    "dislpayport": "displayport",
    # USB-C typos
    "usc-c": "usb-c",
    "usb-v": "usb-c",
    "usbcc": "usb-c",
    # Thunderbolt typos
    "thunderbot": "thunderbolt",
    "thuderbolt": "thunderbolt",
    # Ethernet typos
    "ethernnet": "ethernet",
    "ehternet": "ethernet",
    "enthernet": "ethernet",
    # Cable typos
    "cabel": "cable",
    "calbe": "cable",
    "caple": "cable",
    # Adapter typos
    "adaptor": "adapter",
    "addapter": "adapter",
    "adpater": "adapter",
}

# Cable type abbreviations
CABLE_TYPE_SYNONYMS = {
    "cat5": "category 5 ethernet",
    "cat5e": "category 5e ethernet",
    "cat6": "category 6 ethernet",
    "cat6a": "category 6a ethernet",
    "cat7": "category 7 ethernet",
}

# Video standards and resolutions
VIDEO_SYNONYMS = {
    "4k": "3840x2160",
    "4k60": "4k 60hz",
    "1080p": "1920x1080",
    "720p": "1280x720",
    "uhd": "4k",
    "full hd": "1080p",
}

# Common product phrases
PRODUCT_SYNONYMS = {
    "charger": "charging cable",
    "power cord": "power cable",
    "monitor cable": "display cable",
    "laptop dock": "docking station",
}

# Material variations
MATERIAL_SYNONYMS = {
    "braided": "nylon braided",
    "aluminum": "aluminium",
}

# Feature abbreviations
FEATURE_SYNONYMS = {
    "poe": "power over ethernet",
    "4k support": "4k display support",
}

# Combined synonyms dictionary
SYNONYMS = {
    **CONNECTOR_SYNONYMS,
    **TYPO_CORRECTIONS,
    **CABLE_TYPE_SYNONYMS,
    **VIDEO_SYNONYMS,
    **PRODUCT_SYNONYMS,
    **MATERIAL_SYNONYMS,
    **FEATURE_SYNONYMS,
}


def expand_synonyms(text: str) -> str:
    """
    Expand common abbreviations and synonyms in user queries.
    
    Args:
        text: User query text
        
    Returns:
        Text with synonyms expanded
        
    Example:
        >>> expand_synonyms("I need a 6ft DP cable")
        "i need a 6ft displayport cable"
    """
    text_lower = text.lower()
    expanded = text_lower
    
    # Sort by length (longest first) to handle multi-word synonyms
    for abbr, full in sorted(SYNONYMS.items(), key=lambda x: -len(x[0])):
        # Use word boundaries to avoid partial matches
        import re
        pattern = r'\b' + re.escape(abbr) + r'\b'
        expanded = re.sub(pattern, full, expanded)
    
    return expanded
