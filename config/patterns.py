"""
Regex patterns for structured data extraction.

Contains only essential patterns for extracting structured data
like lengths, SKUs, and explicit connectors.
"""

import re

# === Length/Distance Patterns ===

# Length units
LENGTH_UNIT = r'(?:ft|feet|foot|in(?:ch(?:es)?)?|cm|centimeter(?:s)?|centimetre(?:s)?|m|meter(?:s)?|metre(?:s)?)'

# Number + unit pattern (with word boundaries)
NUM_WITH_UNIT = rf'\b\d+(?:\.\d+)?\s*{LENGTH_UNIT}\b'

# Number + unit pattern (without boundaries for internal use)
NUM_WITH_UNIT_NOB = rf'\d+(?:\.\d+)?\s*(?:ft|feet|foot|in(?:ch(?:es)?)?|cm|centimeter(?:s)?|centimetre(?:s)?|m|meter(?:s)?|metre(?:s)?)'


# === SKU Patterns ===

# StarTech product numbers (alphanumeric with optional hyphens, 3+ chars)
PRODUCT_NUMBER_PATTERN = r'[A-Z0-9-]{3,}'


# === Connector Patterns ===

# Connector types (for detection, not extraction)
CONNECTOR_PATTERN = r'\b(usb[\s\-]?c|usb[\s\-]?a|usb|hdmi|displayport|display\s*port|dp|dvi|vga|thunderbolt)\b'

# Specific connector-to-connector patterns
CONNECTOR_TO_PATTERNS = {
    'usb-c_to_hdmi': r'\busb[\s\-]?c\s+to\s+hdmi\b',
    'usb-c_to_dp': r'\busb[\s\-]?c\s+to\s+(?:displayport|display\s*port|dp)\b',
    'hdmi_to_usb-c': r'\bhdmi\s+to\s+usb[\s\-]?c\b',
    'vga_to_hdmi': r'\bvga\s+to\s+hdmi\b',
    'hdmi_to_vga': r'\bhdmi\s+to\s+vga\b',
    'dp_to_hdmi': r'\b(?:displayport|display\s*port|dp)\s+to\s+hdmi\b',
    'hdmi_to_dp': r'\bhdmi\s+to\s+(?:displayport|display\s*port|dp)\b',
    'dvi_to_hdmi': r'\bdvi\s+to\s+hdmi\b',
    'hdmi_to_dvi': r'\bhdmi\s+to\s+dvi\b',
}

# Single connector cable patterns (e.g., "HDMI cable", "USB-C cable")
SINGLE_CONNECTOR_PATTERNS = {
    'hdmi': r'\bhdmi\s+cables?\b',
    'displayport': r'\b(?:displayport|display\s*port|dp)\s+cables?\b',
    'vga': r'\bvga\s+cables?\b',
    'dvi': r'\bdvi\s+cables?\b',
    'usb-c': r'\busb[\s\-]?c\s+cables?\b',
    'usb-a': r'\busb[\s\-]?a\s+cables?\b',
}


# === Number Patterns ===

# Numeric words for parsing
NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "twenty": 20,
}


# === Intent Detection Patterns ===

# Greeting patterns
GREETING_PATTERNS = [
    r'\bhello\b',
    r'\bhi\b',
    r'\bhey\b',
    r'\bgood\s+morning\b',
    r'\bgood\s+afternoon\b',
    r'\bgood\s+evening\b',
]

# Farewell patterns
FAREWELL_PATTERNS = [
    r'\bthank\s*you\b',
    r'\bthanks\b',
    r'\bbye\b',
    r'\bgoodbye\b',
    r'\bsee\s+you\b',
    r'\bappreciate\s+it\b',
    r'\bcheers\b',
]

# Installation/setup request patterns (blocked)
# Note: "setup" alone is intentionally NOT included here because "multi-monitor setup"
# and "dual monitor setup" should trigger SETUP_GUIDANCE, not INSTALL_HELP.
# INSTALL_HELP is for actual installation/configuration instructions.
#
# IMPORTANT: "mount" as a noun is a product category ("monitor mount", "wall mount")
# Only match "mount" when used as a VERB (installation action):
# - "how do I mount this?" → INSTALL_HELP (blocked)
# - "mounting the bracket" → INSTALL_HELP (blocked)
# - "I need a monitor mount" → NEW_SEARCH (product search)
INSTALL_PATTERNS = [
    r'\binstall\b',
    r'\binstallation\b',
    # "set up" patterns - be careful not to match "set up dual monitors" (product search)
    r'\bhow\s+(?:do\s+i\s+|to\s+)?set\s*up\b',  # "how do I set up", "how to set up"
    r'\bset\s*up\s+(?:help|guide|instructions?|steps?)\b',  # "set up help", "set up instructions"
    r'\bsetup\s+(?:instructions?|guide|help|steps?)\b',  # "setup instructions", "setup help"
    r'\bhow\s+(?:do\s+i\s+|to\s+)?setup\b',  # "how do I setup", "how to setup"
    r'\bconfigure\b',
    r'\bconfiguration\b',
    r'\bwiring\b',
    # "mount" as VERB only - not as noun (product category)
    r'\bhow\s+(?:do\s+i\s+|to\s+)?mount\b',  # "how do I mount", "how to mount"
    r'\bmount(?:ing)?\s+(?:it|this|the|my|a)\b',  # "mount it", "mounting this", "mount the bracket"
    r'\bcan\s+(?:i|you)\s+mount\b',  # "can I mount", "can you mount"
    r'\bfirmware\b',
    r'\btroubleshoot\b',
    r'\bfix\b',
    r'\brepair\b',
]

# Warranty/returns patterns (blocked - redirect to support)
# NOTE: "return" patterns must NOT match "Audio Return Channel" (ARC feature)
WARRANTY_PATTERNS = [
    r'\bwarranty\b',
    r'\bguarantee\b',
    r'\breturn\s+(?:policy|it|this|the|my|a)\b',  # "return policy", "return it", "return this", etc.
    r'\breturn(?:s|ed|ing)\b',  # "returns", "returned", "returning" (but NOT just "return")
    r'\bcan\s+i\s+return\b',  # "can I return"
    r'\bwant\s+to\s+return\b',  # "want to return"
    r'\bneed\s+to\s+return\b',  # "need to return"
    r'\bhow\s+(?:do\s+i|to)\s+return\b',  # "how do I return", "how to return"
    r'\brefund\b',
    r'\brma\b',
    r'\bexchange\b',
    r'\breplacement\b',
    r'\bdefective\b',
    r'\bbroken\b',
    r'\bdamaged\b',
    r'\bhow\s+long\s+(?:is|does)\s+(?:the\s+)?(?:warranty|guarantee)\b',
]

# Pricing/discount patterns (blocked - redirect to sales)
PRICING_PATTERNS = [
    r'\b(?:price|pricing|prices)\b',
    r'\bdiscounts?\b',
    r'\bcoupons?\b',
    r'\bpromo(?:tion|tional)?\s*(?:code)?\b',
    r'\bcheaper\b',
    r'\bbest\s+(?:deal|price)\b',
    r'\bquotes?\b',
    r'\bbulk\s+(?:pricing|discount|order)\b',
    r'\bwholesale\b',
    r'\bcost\s+(?:less|more)\b',
    r'\bhow\s+much\s+(?:does|is|do)\b',
    r'\bsale\b',
]

# Daisy-chain keywords
DAISY_CHAIN_PATTERNS = [
    r'\bdaisy[\s\-]?chain\b',
    r'\bdaisychain\b',
    r'\bchain\s+monitors\b',
    r'\bseries\s+connection\b',
]


# === Setup/Guidance Patterns ===
# These indicate complex queries that need diagnostic questions rather than direct search

# Multi-monitor setup patterns
# Note: "\d+ monitors" without length unit context indicates setup, not cable length
MULTI_MONITOR_PATTERNS = [
    r'\b(?:for|to|with)\s+(\d+)\s+monitors?\b',  # "for 3 monitors", "to 2 monitors"
    r'\b(\d+)\s+monitors?\s+(?:setup|configuration)\b',  # "3 monitor setup"
    r'\bconnect(?:ing)?\s+(\d+)\s+monitors?\b',  # "connect 3 monitors", "connecting 2 monitors"
    r'\b(?:i\s+)?(?:have|got|\'ve\s+got)\s+(\d+)\s+monitors?\b',  # "I have 2 monitors", "got 3 monitors"
    r'\b(dual|triple|quad)\s+monitors?\b',  # "dual monitor", "triple monitor"
    r'\b(two|three|four)\s+monitors?\b',  # "two monitors", "three monitors"
    r'\bmultiple\s+monitors?\b',  # "multiple monitors"
    r'\b(?:add|adding)\s+(?:a\s+)?(?:second|another|more)\s+monitors?\b',  # "add a second monitor"
    r'\bextend(?:ing)?\s+(?:to\s+)?(?:more\s+)?monitors?\b',  # "extend to more monitors"
    r'\bmulti[\s\-]?monitor\b',  # "multi-monitor", "multimonitor"
]

# Patterns that look like multi-monitor but are actually quantity requests (should NOT trigger guidance)
MULTI_MONITOR_EXCEPTIONS = [
    r'\b\d+\s+(?:hdmi|displayport|dp|usb|vga|dvi)\s+cables?\b',  # "3 HDMI cables" = quantity
    r'\b\d+\s+(?:of\s+)?(?:these|those|them)\b',  # "3 of these" = quantity
    r'\bneed\s+\d+\b(?!\s*monitors?)',  # "need 3" without monitors = quantity
]

# Single monitor connection patterns (need guidance)
# These indicate the user wants to connect a monitor but hasn't specified connectors
# "connect monitor to PC" needs guidance: What ports does your monitor/PC have?
# NOTE: "display" and "screen" are synonyms for "monitor" in this context
# IMPORTANT: "screen" patterns must NOT match "privacy screen" - see SINGLE_MONITOR_EXCEPTIONS
_DISPLAY_DEVICE = r'(?:monitors?|displays?|screens?)'  # Common synonyms for monitor

SINGLE_MONITOR_PATTERNS = [
    # Connect [device] to [computer] patterns
    r'\bconnect(?:ing)?\s+(?:a\s+)?(?:my\s+)?(?:second\s+|another\s+|additional\s+)?' + _DISPLAY_DEVICE + r'\s+to\s+(?:a\s+)?(?:my\s+)?(?:pc|computer|laptop|desktop|macbook|mac)\b',
    r'\bconnect(?:ing)?\s+(?:a\s+)?(?:my\s+)?(?:pc|computer|laptop|desktop|macbook|mac)\s+to\s+(?:a\s+)?(?:my\s+)?(?:second\s+|another\s+|additional\s+)?' + _DISPLAY_DEVICE + r'\b',
    # Connect another/second/additional [device] (no computer specified - still needs guidance)
    r'\bconnect(?:ing)?\s+(?:a\s+)?(?:my\s+)?(?:second|another|additional)\s+' + _DISPLAY_DEVICE + r'\b',
    # Hook up patterns
    r'\bhook(?:ing)?\s+up\s+(?:a\s+)?(?:my\s+)?(?:second\s+|another\s+|additional\s+)?' + _DISPLAY_DEVICE + r'\b',
    # Need/want to connect patterns
    r'\b(?:need|want)\s+to\s+connect\s+(?:a\s+)?(?:my\s+)?(?:second\s+|another\s+|additional\s+)?' + _DISPLAY_DEVICE + r'\b',
    r'\btrying\s+to\s+connect\s+(?:a\s+)?(?:my\s+)?(?:second\s+|another\s+|additional\s+)?' + _DISPLAY_DEVICE + r'\b',
    # Get [device] working patterns
    r'\bget\s+(?:a\s+)?(?:my\s+)?(?:second\s+|another\s+|additional\s+)?' + _DISPLAY_DEVICE + r'\s+(?:working|connected)\b',
    # Display on/to patterns (uses "display" as verb)
    r'\bdisplay\s+(?:on|to)\s+(?:a\s+)?(?:my\s+)?' + _DISPLAY_DEVICE + r'\b',
    # Link patterns
    r'\blink\s+(?:a\s+)?(?:my\s+)?(?:pc|computer|laptop)\s+to\s+(?:a\s+)?(?:my\s+)?(?:second\s+|another\s+|additional\s+)?' + _DISPLAY_DEVICE + r'\b',
    r'\blink\s+(?:a\s+)?(?:my\s+)?(?:second\s+|another\s+|additional\s+)?' + _DISPLAY_DEVICE + r'\s+to\s+(?:a\s+)?(?:my\s+)?(?:pc|computer|laptop)\b',
    # Cable for/to connect patterns
    r'\bcable\s+(?:to\s+)?connect\s+(?:a\s+)?(?:my\s+)?(?:second\s+|another\s+|additional\s+)?' + _DISPLAY_DEVICE + r'\b',
    r'\b' + _DISPLAY_DEVICE + r'\s+(?:cable|connection)\s+(?:for|to)\s+(?:a\s+)?(?:my\s+)?(?:pc|computer|laptop)\b',
    # "Cable for my [display/screen/monitor]" - simple request for display cable
    r'\bcable\s+for\s+(?:a\s+)?(?:my\s+)?(?:second\s+|another\s+|additional\s+)?' + _DISPLAY_DEVICE + r'\b',
    # Direct setup descriptions - user providing port info without explicit "connect" verb
    r'\b(?:my\s+)?(?:computer|laptop|pc)\s+has\s+.{0,60}(?:my\s+)?' + _DISPLAY_DEVICE + r'\s+has\b',
    r'\b' + _DISPLAY_DEVICE + r'\s+has\s+.{0,60}(?:computer|laptop|pc)\s+has\b',
    r'\b(?:computer|laptop|pc)\s*[:=]\s*(?:usb|hdmi|displayport|dp|vga|dvi).{0,80}' + _DISPLAY_DEVICE + r'\s*[:=]',
]

# Exceptions for single monitor patterns (user already knows what they need)
SINGLE_MONITOR_EXCEPTIONS = [
    r'\b(?:hdmi|displayport|dp|usb[\s\-]?c|vga|dvi|thunderbolt)\s+to\s+(?:hdmi|displayport|dp|usb[\s\-]?c|vga|dvi|thunderbolt)\b',  # "HDMI to DisplayPort" - knows connectors
    r'\b(?:hdmi|displayport|dp|usb[\s\-]?c|vga|dvi|thunderbolt)\s+cables?\b',  # "HDMI cable" - knows connector
    r'\bprivacy\s+screens?\b',  # "privacy screen" is a different product, not a monitor
    r'\bscreen\s+(?:protector|filter|cover)\b',  # Screen accessories, not monitor connections
]

# === Intent-Based Dock Detection (Order-Independent) ===
# Instead of rigid patterns, we detect dock queries by checking for:
# 1. Dock keyword (required)
# 2. Specific requirements (optional - if present, skip guidance)
#
# This handles natural language variations like:
# - "USB-C dock with power" = "Power delivery USB-C dock" = "charging dock USB-C"

# Patterns that indicate a dock request (required)
DOCK_KEYWORDS = [
    r'\bdock(?:ing)?s?\b',  # "dock", "docks", "docking"
    r'\bdocking\s+stations?\b',  # "docking station", "docking stations"
]

# Specific requirements that indicate user knows what they need (skip guidance)
# These are checked IN ADDITION to dock keywords, regardless of order
DOCK_SPECIFIC_REQUIREMENTS = {
    'monitor_count': [
        r'\b\d+\s*(?:monitors?|displays?)\b',  # "2 monitors", "3 displays"
        r'\b(?:dual|triple|quad)\s*(?:monitors?|displays?)?\b',  # "dual", "triple monitor"
        r'\b(?:two|three|four)\s*(?:monitors?|displays?)\b',  # "two monitors"
    ],
    'power_delivery': [
        r'\b(?:power\s*delivery|pd)\b',  # "power delivery", "PD"
        r'\bcharging?\b',  # "charging", "charge"
        r'\b\d+\s*w(?:att)?s?\b',  # "100W", "65 watts"
        r'\blaptop\s*(?:power|charg)',  # "laptop charging"
    ],
    'resolution': [
        r'\b(?:4k|8k|1440p|1080p|uhd|qhd)\b',  # Resolution specs
    ],
    'ethernet': [
        r'\bethernet\b',
        r'\bnetwork\b',
        r'\bgigabit\b',
        r'\blan\b',
        r'\brj[\s\-]?45\b',
    ],
    'usb_ports': [
        r'\busb\s*(?:3\.?[012]?|2\.?0?)?\s*(?:ports?|hub)\b',  # "USB ports", "USB 3.0 hub"
        r'\busb[\s\-]?c\s*(?:ports?|hub)\b',  # "USB-C ports", "USB C hub"
        r'\b\d+\s*(?:usb|ports?)\b',  # "4 USB", "6 ports"
        r'\b(?:lots?\s+of|bunch\s+of|many|multiple|several)\s+(?:usb|ports?)\b',  # "bunch of USB", "many ports"
    ],
}

# Legacy patterns for basic dock detection (still used as fallback)
VAGUE_DOCK_PATTERNS = [
    r'\b(?:need|want|looking\s+for|find)\s+(?:a\s+)?dock(?:ing)?\s*(?:station)?\b',
    r'\bdock(?:ing)?\s*(?:station)?\s+(?:for|please|recommendations?)\b',
    r'\b(?:recommend|suggest)\s+(?:a\s+)?dock(?:ing)?\b',
    r'\bwhat\s+dock(?:ing)?\b',
    r'\bwhich\s+dock(?:ing)?\b',
    r'\b(?:best|good)\s+dock(?:ing)?\b',
    r'^dock(?:ing)?\s*(?:station)?[.!?\s]*$',
    r'\bget\s+(?:a\s+)?dock(?:ing)?\b',
    r'\bbuy\s+(?:a\s+)?dock(?:ing)?\b',
    r'\b(?:usb[\s\-]?c|thunderbolt|usb[\s\-]?a)\s+dock(?:ing)?\s*(?:station)?\b',
]

# Legacy specific patterns (kept for backward compatibility, but new logic is preferred)
SPECIFIC_DOCK_PATTERNS = [
    r'\b\d+\s+monitor\s+dock\b',
    r'\b(?:dual|triple|quad)\s+monitor\s+dock\b',
]


def is_dock_query(text: str) -> bool:
    """Check if query is about docks (order-independent)."""
    import re
    return any(re.search(p, text, re.IGNORECASE) for p in DOCK_KEYWORDS)


def has_dock_specific_requirements(text: str) -> bool:
    """
    Check if dock query has specific requirements (order-independent).

    Returns True if user has specified ANY of:
    - Monitor count (2 monitors, dual display, etc.)
    - Power delivery (charging, PD, 100W, etc.)
    - Resolution (4K, 8K, etc.)
    - Ethernet/network
    - USB port requirements

    These can appear in ANY order in the query.
    """
    import re
    for requirement_type, patterns in DOCK_SPECIFIC_REQUIREMENTS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
    return False


# === KVM Switch Patterns ===

# Vague KVM patterns (need guidance)
# User wants a KVM but hasn't provided enough specifics
VAGUE_KVM_PATTERNS = [
    r'\b(?:need|want|looking\s+for|find)\s+(?:a\s+)?kvm\b',  # "need a KVM", "looking for KVM"
    r'\bkvm\s+(?:switch\s+)?(?:for|please|recommendations?)\b',  # "KVM switch for", "KVM please"
    r'\b(?:recommend|suggest)\s+(?:a\s+)?kvm\b',  # "recommend a KVM"
    r'\bwhat\s+kvm\b',  # "what KVM"
    r'\bwhich\s+kvm\b',  # "which KVM"
    r'\b(?:best|good)\s+kvm\b',  # "best KVM"
    r'^kvm(?:\s+switch)?[.!?\s]*$',  # Just "KVM" or "KVM switch"
    r'\bget\s+(?:a\s+)?kvm\b',  # "get a KVM"
    r'\bbuy\s+(?:a\s+)?kvm\b',  # "buy a KVM"
    r'\bkvm\s+(?:switch\s+)?for\s+(?:\d+|two|three|four)\s+(?:computers?|pcs?|machines?)\b',  # "KVM for 2 computers"
    r'\bshare\s+(?:a\s+)?(?:monitor|keyboard|mouse)\s+(?:between|with)\b',  # "share a monitor between"
    r'\bcontrol\s+(?:\d+|multiple|two|three)\s+(?:computers?|pcs?|machines?)\b',  # "control 2 computers"
    r'\bswitch\s+between\s+(?:\d+|multiple|two|three)\s+(?:computers?|pcs?|machines?)\b',  # "switch between computers"
]

# Specific KVM patterns (have enough info, don't need guidance)
# User knows exactly what they want - port count, video type, etc.
SPECIFIC_KVM_PATTERNS = [
    r'\b(?:\d+)[\s\-]?port\s+(?:hdmi|displayport|dp|dvi|vga)\s+kvm\b',  # "2-port HDMI KVM"
    r'\b(?:hdmi|displayport|dp|dvi|vga)\s+(?:\d+)[\s\-]?port\s+kvm\b',  # "HDMI 2-port KVM"
    r'\b(?:hdmi|displayport|dp|dvi|vga)\s+kvm\s+(?:switch\s+)?(?:with|for)\s+\d+\b',  # "HDMI KVM with 4 ports"
    r'\b(?:\d+)[\s\-]?port\s+kvm\s+(?:with|for)\s+(?:hdmi|displayport|dp|dvi|vga)\b',  # "4-port KVM for HDMI"
    r'\bkvm\s+(?:switch\s+)?(?:with|that\s+supports?)\s+(?:4k|8k|1080p)\b',  # "KVM with 4K"
    r'\b(?:4k|8k)\s+kvm\b',  # "4K KVM"
    r'\busb[\s\-]?c\s+kvm\b',  # "USB-C KVM" - knows the connection type
    r'\bthunderbolt\s+kvm\b',  # "Thunderbolt KVM"
    r'\bdual[\s\-]?monitor\s+kvm\b',  # "dual monitor KVM" - multi-display KVM
]


# === Impossible Combination Patterns ===
# Products that don't exist due to technical incompatibility
IMPOSSIBLE_COMBINATIONS = {
    # Bluetooth + wired cables - inherently contradictory
    r'\bbluetooth\s+(?:hdmi|displayport|dp|dvi|vga|usb|cable)\b': {
        'reason': "Bluetooth is wireless technology - it doesn't use cables",
        'alternative': "wireless HDMI transmitter",
        'suggestion': "Are you looking for a wireless video solution? Try 'wireless HDMI extender'."
    },
    r'\b(?:hdmi|displayport|dp|dvi|vga)\s+bluetooth\b': {
        'reason': "HDMI/video signals require wired connections",
        'alternative': "wireless HDMI extender",
        'suggestion': "For wireless video, try searching for 'wireless HDMI extender' or 'wireless display adapter'."
    },
    # WiFi cables - contradictory
    r'\bwifi\s+cable\b': {
        'reason': "WiFi is wireless - it doesn't use cables by definition",
        'alternative': "ethernet cable for wired networking",
        'suggestion': "Did you mean an ethernet cable for a wired connection, or a WiFi adapter?"
    },
    r'\bwireless\s+(?:hdmi|displayport|dp|dvi|vga)\s+cable\b': {
        'reason': "Cables are wired by definition - 'wireless cable' is contradictory",
        'alternative': "wireless HDMI extender",
        'suggestion': "For wireless video, try a 'wireless HDMI extender' instead."
    },
}

# === Out-of-Scope Product Patterns ===
# Products StarTech doesn't make/sell
# NOTE: These patterns must be VERY specific to avoid false positives on valid queries
# like "monitor mount", "triple monitor setup", "connect my laptop", etc.
# NOTE: Suggestions should NOT repeat "StarTech specializes..." - the handler adds that context
OUT_OF_SCOPE_PRODUCTS = {
    # Keyboards - but NOT "KVM" or "share keyboard"
    r'\b(?:buy|purchase|need|want|looking\s+for)\s+(?:a\s+)?(?:wireless|bluetooth|mechanical|ergonomic|gaming)?\s*keyboard\b(?!\s+(?:sharing|switch|kvm))': {
        'category': "keyboards",
        'alternative': "KVM switch to share one keyboard between multiple computers",
        'suggestion': "Need to share one keyboard between multiple computers? Try a **KVM switch** instead!"
    },
    # Mice - but NOT "share mouse"
    r'\b(?:buy|purchase|need|want|looking\s+for)\s+(?:a\s+)?(?:wireless|bluetooth|gaming)?\s*mouse\b(?!\s+(?:sharing|switch|kvm))': {
        'category': "mice",
        'alternative': "KVM switch to share one mouse between multiple computers",
        'suggestion': "To share one mouse between computers, consider a **KVM switch**."
    },
    # Buying actual monitor hardware (not mounts, cables, or multi-monitor setup queries)
    # Must have explicit "buy/purchase/need a monitor" NOT "monitor cable/mount/setup"
    r'\b(?:buy|purchase)\s+(?:a\s+)?(?:new\s+)?(?:computer|gaming|4k|curved)?\s*(?:monitor|display|tv)\b': {
        'category': "monitors and displays",
        'alternative': "monitor cables, adapters, or mounts",
        'suggestion': "We have **monitor cables**, **adapters**, and **mounts**. What connection do you need?"
    },
    # Buying actual computers/laptops (not accessories for them)
    r'\b(?:buy|purchase)\s+(?:a\s+)?(?:new\s+)?(?:laptop|notebook|desktop|computer|pc|mac|macbook)\b': {
        'category': "computers",
        'alternative': "docking stations, cables, or adapters for your computer",
        'suggestion': "Need a **dock**, **cable**, or **adapter** for your computer?"
    },
    # Buying printers (not printer cables/servers)
    r'\b(?:buy|purchase|need|want)\s+(?:a\s+)?(?:new\s+)?printer\b(?!\s+(?:cable|adapter|server))': {
        'category': "printers",
        'alternative': "printer cables or USB print servers",
        'suggestion': "We have **printer cables** and **USB print servers**. What do you need to connect?"
    },
    # Buying speakers/headphones (not audio cables)
    r'\b(?:buy|purchase|need|want)\s+(?:a\s+)?(?:new\s+)?(?:speaker|headphone|headset|earphone|earbud)s?\b(?!\s+(?:cable|adapter|splitter))': {
        'category': "speakers and audio equipment",
        'alternative': "audio cables, adapters, or splitters",
        'suggestion': "Need an **audio cable**, **adapter**, or **splitter** instead?"
    },
}


# === Compiled Patterns (for performance) ===

# Pre-compile frequently used patterns
LENGTH_PATTERN = re.compile(NUM_WITH_UNIT, re.IGNORECASE)
SKU_PATTERN = re.compile(PRODUCT_NUMBER_PATTERN)
CONNECTOR_DETECT = re.compile(CONNECTOR_PATTERN, re.IGNORECASE)


# === Helper Functions ===

def extract_lengths(text: str) -> list[tuple[float, str]]:
    """
    Extract all length measurements from text.
    
    Args:
        text: Input text
        
    Returns:
        List of (value, unit) tuples
        
    Example:
        >>> extract_lengths("I need a 6ft or 2m cable")
        [(6.0, 'ft'), (2.0, 'm')]
    """
    matches = LENGTH_PATTERN.finditer(text)
    results = []
    
    for match in matches:
        text_match = match.group(0)
        # Parse number and unit
        num_match = re.search(r'(\d+(?:\.\d+)?)', text_match)
        unit_match = re.search(LENGTH_UNIT, text_match)
        
        if num_match and unit_match:
            value = float(num_match.group(1))
            unit = unit_match.group(0)
            results.append((value, unit))
    
    return results


def has_pattern(text: str, patterns: list[str]) -> bool:
    """
    Check if any pattern matches text.
    
    Args:
        text: Input text
        patterns: List of regex patterns
        
    Returns:
        True if any pattern matches
    """
    text_lower = text.lower()
    return any(re.search(pat, text_lower) for pat in patterns)