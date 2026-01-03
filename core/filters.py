"""
Filter extraction for ST-Bot.

Extracts structured search filters from user queries:
- Length requirements (6ft, 2 meters)
- Connector types (USB-C, HDMI, DisplayPort)
- Technical specifications (4K, Thunderbolt)
- Product categories

Pure Python, no external dependencies except config modules.
"""

import re
from typing import Optional
from core.context import SearchFilters, LengthPreference
from core.structured_logging import get_logger
from config.patterns import (
    extract_lengths,
    CONNECTOR_TO_PATTERNS,
    SINGLE_CONNECTOR_PATTERNS,
    NUMBER_WORDS,
)
from config.synonyms import expand_synonyms

# Module-level logger
_logger = get_logger("core.filters")


# Patterns that indicate user is okay with shorter cables
SHORTER_OK_PATTERNS = [
    r'\bshorter\s+(?:is\s+)?(?:fine|ok|okay|good|works?|acceptable)\b',
    r'\bshorter\s+(?:would\s+)?(?:work|be\s+fine|be\s+ok)\b',
    r'\b(?:can\s+be|could\s+be)\s+shorter\b',
    r'\bdon\'?t\s+(?:need|want)\s+(?:it\s+)?(?:that\s+)?long\b',
    r'\bor\s+shorter\b',
    r'\bup\s+to\s+\d+',  # "up to 6ft" implies shorter is fine
    r'\bmax(?:imum)?\s+(?:of\s+)?\d+',  # "max 6ft" implies shorter is fine
    r'\bno\s+(?:longer|more)\s+than\b',
    r'\bunder\s+\d+',  # "under 6ft" implies shorter is fine
    r'\bless\s+than\s+\d+',  # "less than 6ft" implies shorter is fine
]

# Patterns that indicate user wants flexibility (closest match)
FLEXIBLE_PATTERNS = [
    r'\bclose(?:st)?\s+to\b',
    r'\babout\s+\d+',  # "about 6ft"
    r'\baround\s+\d+',  # "around 6ft"
    r'\bapprox(?:imately)?\s+\d+',
    r'~\s*\d+',  # "~6ft" - no word boundary needed for tilde
    r'\broughly\s+\d+',
    r'\bnearest\s+to\b',
    r'\bwhatever\s+(?:is\s+)?closest\b',
    r'\beither\s+(?:way|direction)\b',
]

# Patterns that indicate user wants exact or longer (default, but explicit)
LONGER_OK_PATTERNS = [
    r'\bat\s+least\s+\d+',  # "at least 6ft"
    r'\bminimum\s+(?:of\s+)?\d+',  # "minimum 6ft"
    r'\bno\s+(?:shorter|less)\s+than\b',
    r'\bor\s+longer\b',
    r'\bor\s+more\b',
]


class FilterExtractor:
    """
    Extracts search filters from user queries.

    Converts natural language queries into structured SearchFilters objects
    that can be used for Pinecone vector search.

    Example:
        extractor = FilterExtractor()
        filters = extractor.extract("6ft USB-C to HDMI cable")
        # Returns: SearchFilters(length=6.0, connector_from="USB-C", ...)
    """

    # Color keywords - maps user input to normalized color names
    # The values should match what's in the Excel COLOR column
    COLOR_KEYWORDS = {
        'black': 'Black',
        'white': 'White',
        'red': 'Red',
        'blue': 'Blue',
        'gray': 'Gray',
        'grey': 'Gray',  # British spelling
        'silver': 'Silver',
        'green': 'Green',
        'yellow': 'Yellow',
        'orange': 'Orange',
        'pink': 'Pink',
        'purple': 'Purple',
        'gold': 'Gold',
        'beige': 'Beige',
        'brown': 'Brown',
    }

    def __init__(self):
        """Initialize the filter extractor."""
        # Feature keywords for technical specs
        self.feature_keywords = {
            '4k': ['4k', '2160p', 'uhd', 'ultra hd'],
            '8k': ['8k', '4320p'],
            '1080p': ['1080p', 'full hd', 'fhd'],
            '1440p': ['1440p', '2k', 'qhd'],
            'hdr': ['hdr', 'hdr10', 'hdr10+', 'dolby vision'],
            'thunderbolt': ['thunderbolt', 'tb3', 'tb4', 'thunderbolt 3', 'thunderbolt 4'],
            'usb 3.0': ['usb 3.0', 'usb3.0', 'superspeed'],
            'usb 3.1': ['usb 3.1', 'usb3.1'],
            'usb 3.2': ['usb 3.2', 'usb3.2'],
            'hdcp': ['hdcp', 'hdcp 2.2'],
            'power delivery': ['power delivery', 'pd', 'usb-pd', '100w', '60w', 'charging', 'charge'],
        }
        
        # Product category keywords
        # Maps user keywords to categories that match actual data categories
        # via substring matching (e.g., 'switch' matches 'kvm switches')
        #
        # ORDER MATTERS: More specific categories should come first so
        # "kvm switch" matches "kvm switches" before matching "switches"
        self.category_keywords = {
            # Specific categories first (order matters!)
            'kvm switches': ['kvm'],  # "kvm switch" -> "kvm switches" (not "switches")
            'display mounts': ['mount', 'mounts', 'monitor mount', 'tv mount'],
            # Network switches must come before generic 'switch' and 'network'
            'ethernet switches': ['network switch', 'ethernet switch', 'gigabit switch', 'poe switch'],
            # Fiber cables must come before generic 'cables'
            'fiber cables': ['fiber optic', 'fiber cable', 'fiber patch', 'optical fiber'],
            # Storage enclosures must come before generic 'enclosures'
            'storage enclosures': ['drive enclosure', 'hard drive enclosure', 'ssd enclosure',
                                   'hdd enclosure', 'nvme enclosure', 'm.2 enclosure'],
            # Privacy screens/filters
            'privacy screens': ['privacy screen', 'privacy filter', 'screen filter'],
            # Server racks - require specific terms, not just bare "rack" which could be
            # contextual (e.g., "patch panel for rack" is about patch panels, not racks)
            'server racks': ['server rack', 'equipment rack', '19 inch rack', '42u rack',
                             'data rack', 'network rack', 'rack cabinet', 'rack enclosure'],
            # Computer/expansion cards
            'computer cards': ['pci card', 'expansion card', 'pcie card', 'network card', 'video card'],
            # Video splitters (before generic 'splitters')
            'video splitters': ['video splitter', 'hdmi splitter', 'displayport splitter', 'dp splitter'],
            # Multiport adapters (before generic 'adapters')
            'multiport adapters': ['multiport adapter', 'multiport', 'multi-port adapter', 'multi port adapter'],
            # General categories
            # Cables includes network cable types (cat5, cat6, patch cable)
            'cables': ['cable', 'cables', 'cord', 'cords', 'cat6', 'cat5e', 'cat5',
                       'cat6a', 'cat7', 'patch cable', 'ethernet cable'],
            'adapters': ['adapter', 'adapters', 'converter', 'converters'],
            'docks': ['dock', 'docking', 'docking station'],
            'hubs': ['hub', 'hubs'],
            'switches': ['switch', 'switcher'],  # Generic switch (after KVM and ethernet)
            'enclosures': ['enclosure', 'enclosures', 'case'],
            'splitters': ['splitter', 'splitters'],  # Generic splitter fallback
            'networking': ['network', 'ethernet'],
        }
    
    # Categories where connector extraction should be suppressed for ambiguous terms
    # "USB hub" → USB describes the hub type, not a cable connector pair
    # But "USB-C to HDMI dock" → connectors are meaningful
    NON_CABLE_CATEGORIES = {'hubs', 'docks', 'switches', 'kvm switches', 'enclosures',
                            'fiber cables', 'storage enclosures', 'privacy screens',
                            'server racks', 'computer cards', 'video splitters', 'splitters',
                            'multiport adapters'}

    def extract(self, query: str) -> SearchFilters:
        """
        Extract all filters from a query.

        Args:
            query: User's search query

        Returns:
            SearchFilters with extracted filters

        Example:
            >>> extractor.extract("6ft USB-C to HDMI cable")
            SearchFilters(length=6.0, connector_from="USB-C", connector_to="HDMI")

            >>> extractor.extract("6ft USB-C cable, shorter is fine")
            SearchFilters(length=6.0, length_preference=EXACT_OR_SHORTER, ...)

            >>> extractor.extract("USB hub")
            SearchFilters(category="Hubs", connector_from=None, connector_to=None)
        """
        query_lower = query.lower()
        query_expanded = expand_synonyms(query)
        query_expanded_lower = query_expanded.lower()

        # Extract category FIRST - affects how we interpret connectors
        category = self._extract_category(query_lower)

        # Extract components
        length, length_unit = self._extract_length(query_lower)
        length_preference = self._extract_length_preference(query_lower)
        connector_from, connector_to = self._extract_connectors(query_expanded)
        features = self._extract_features(query_lower)
        port_count = self._extract_port_count(query_lower)
        color = self._extract_color(query_lower)
        # Use expanded query for monitor extraction (handles typos like "moinitors")
        min_monitors = self._extract_min_monitors(query_expanded_lower)

        # Special handling for multiport adapters (MUST come before NON_CABLE_CATEGORIES)
        # Multiport adapters have ONE input (e.g., USB-C) but MULTIPLE DIFFERENT output types
        # (HDMI, USB-A, Ethernet, etc.), so connector_to is meaningless but connector_from is useful
        # "USB-C multiport adapter" → connector_from=USB-C, connector_to=None
        if 'multiport' in query_lower or 'multi-port' in query_lower or 'multi port' in query_lower:
            connector_to = None

        # For non-cable categories, suppress ambiguous connector extraction
        # "USB hub" → USB describes the hub type, not a cable connector pair
        # But "USB-C to HDMI dock" → connectors are meaningful (explicit pair)
        if category and category.lower() in self.NON_CABLE_CATEGORIES:
            # Only suppress if connectors are ambiguous (same from/to without explicit "to" pattern)
            if connector_from == connector_to and connector_from is not None:
                # Check if there's an explicit "X to Y" pattern in the query
                has_explicit_pair = bool(re.search(
                    r'\b(?:usb-?c?|hdmi|displayport|thunderbolt|vga|dvi)\s+to\s+(?:usb-?c?|hdmi|displayport|vga|dvi)\b',
                    query_lower
                ))
                if not has_explicit_pair:
                    # Ambiguous connector (e.g., "USB hub" → USB describes hub type)
                    connector_from = None
                    connector_to = None

        # Extract keywords for text matching (words not captured by other extraction)
        keywords = self._extract_keywords(query_lower)

        # Extract required port types for docks/hubs (e.g., "dock with USB-C ports")
        required_port_types = self._extract_required_port_types(query_lower, category)

        # Build filter config
        return SearchFilters(
            length=length,
            length_unit=length_unit,
            length_preference=length_preference,
            connector_from=connector_from,
            connector_to=connector_to,
            features=features,
            product_category=category,
            port_count=port_count,
            color=color,
            keywords=keywords,
            required_port_types=required_port_types,
            min_monitors=min_monitors,
        )
    
    # === Length Extraction ===
    
    def _extract_length(self, text: str) -> tuple[Optional[float], Optional[str]]:
        """
        Extract length requirement from text.
        
        Args:
            text: Query text (lowercased)
            
        Returns:
            Tuple of (length_value, length_unit) or (None, None)
            
        Examples:
            "6ft cable" → (6.0, "ft")
            "2 meter cable" → (2.0, "m")
            "six foot cable" → (6.0, "ft")
        """
        # Try numeric patterns first (e.g., "6ft", "2 meters")
        lengths = extract_lengths(text)
        if lengths:
            value, unit = lengths[0]  # Take first match
            return value, self._normalize_unit(unit)
        
        # Try word-based numbers (e.g., "six foot")
        for word, number in NUMBER_WORDS.items():
            # Pattern: "six foot", "six feet", "six ft"
            pattern = rf'\b{word}\s+(?:foot|feet|ft|meter(?:s)?|metre(?:s)?|m)\b'
            match = re.search(pattern, text)
            if match:
                unit_text = match.group(0).split()[-1]
                unit = self._normalize_unit(unit_text)
                return float(number), unit
        
        return None, None
    
    def _normalize_unit(self, unit: str) -> str:
        """
        Normalize length unit to standard form.
        
        Args:
            unit: Raw unit string (ft, feet, foot, m, meter, etc.)
            
        Returns:
            Normalized unit ("ft", "m", "in", "cm")
        """
        unit_lower = unit.lower()
        
        # Feet
        if unit_lower in ['ft', 'feet', 'foot']:
            return 'ft'
        
        # Meters
        if unit_lower in ['m', 'meter', 'meters', 'metre', 'metres']:
            return 'm'
        
        # Inches
        if unit_lower in ['in', 'inch', 'inches']:
            return 'in'
        
        # Centimeters
        if unit_lower in ['cm', 'centimeter', 'centimeters', 'centimetre', 'centimetres']:
            return 'cm'

        return unit_lower

    def _extract_length_preference(self, text: str) -> LengthPreference:
        """
        Extract user's preference for length alternatives.

        Determines how to handle cases where exact length isn't available:
        - EXACT_OR_LONGER (default): Show next size up
        - EXACT_OR_SHORTER: User indicated shorter is acceptable
        - CLOSEST: User wants whatever is closest

        Args:
            text: Query text (lowercased)

        Returns:
            LengthPreference enum value

        Examples:
            "6ft cable" → EXACT_OR_LONGER (default)
            "6ft cable, shorter is fine" → EXACT_OR_SHORTER
            "about 6ft cable" → CLOSEST
            "up to 6ft cable" → EXACT_OR_SHORTER
            "at least 6ft cable" → EXACT_OR_LONGER
        """
        # Check for "shorter is fine" patterns first (higher priority)
        for pattern in SHORTER_OK_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return LengthPreference.EXACT_OR_SHORTER

        # Check for flexible/approximate patterns
        for pattern in FLEXIBLE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return LengthPreference.CLOSEST

        # Check for explicit "longer is fine" patterns (confirms default)
        for pattern in LONGER_OK_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return LengthPreference.EXACT_OR_LONGER

        # Default: prefer next size up
        return LengthPreference.EXACT_OR_LONGER

    # === Connector Extraction ===
    
    def _extract_connectors(self, text: str) -> tuple[Optional[str], Optional[str]]:
        """
        Extract connector types from text.
        
        Handles both:
        - Connector-to-connector patterns: "USB-C to HDMI"
        - Single connector patterns: "HDMI cable"
        
        Args:
            text: Query text (expanded with synonyms)
            
        Returns:
            Tuple of (connector_from, connector_to) or (None, None)
            
        Examples:
            "USB-C to HDMI cable" → ("USB-C", "HDMI")
            "HDMI cable" → ("HDMI", "HDMI")
            "DisplayPort cable" → ("DisplayPort", "DisplayPort")
        """
        text_lower = text.lower()
        
        # Priority 1: Connector-to-connector patterns (more specific)
        for connector_pair, pattern in CONNECTOR_TO_PATTERNS.items():
            if re.search(pattern, text_lower):
                # Parse connector pair (e.g., "usb-c_to_hdmi" → "USB-C", "HDMI")
                from_conn, to_conn = connector_pair.split('_to_')
                from_conn = self._normalize_connector(from_conn)
                to_conn = self._normalize_connector(to_conn)
                return from_conn, to_conn
        
        # Priority 2: Single connector patterns
        for connector, pattern in SINGLE_CONNECTOR_PATTERNS.items():
            if re.search(pattern, text_lower):
                normalized = self._normalize_connector(connector)
                return normalized, normalized
        
        # Priority 3: Bare connector mentions (no "cable" word)
        # e.g., "Show me USB-C" or "I need HDMI"
        connector_matches = self._find_bare_connectors(text_lower)
        if connector_matches:
            normalized = self._normalize_connector(connector_matches[0])
            return normalized, normalized
        
        return None, None
    
    def _find_bare_connectors(self, text: str) -> list[str]:
        """
        Find connector mentions without "cable" keyword.
        
        Args:
            text: Query text (lowercased)
            
        Returns:
            List of connector names found
        """
        connectors = []
        
        # Simple connector keywords
        connector_patterns = {
            'hdmi': r'\bhdmi\b',
            'displayport': r'\b(?:displayport|display\s*port)\b',
            'usb-c': r'\busb[\s\-]?c\b',
            'usb-a': r'\busb[\s\-]?a\b',
            'usb': r'\busb\b(?!\s*[\-c\-a])',  # USB but not USB-C or USB-A
            'thunderbolt': r'\bthunderbolt\b',
            'vga': r'\bvga\b',
            'dvi': r'\bdvi\b',
        }
        
        for connector, pattern in connector_patterns.items():
            if re.search(pattern, text):
                connectors.append(connector)
        
        return connectors
    
    def _normalize_connector(self, connector: str) -> str:
        """
        Normalize connector name to standard form.
        
        Args:
            connector: Raw connector string
            
        Returns:
            Normalized connector name
        """
        connector_lower = connector.lower().replace('_', '-')
        
        # Mapping to standard names
        mappings = {
            'usb-c': 'USB-C',
            'usb-a': 'USB-A',
            'usb': 'USB',
            'hdmi': 'HDMI',
            'displayport': 'DisplayPort',
            'display-port': 'DisplayPort',
            'dp': 'DisplayPort',
            'vga': 'VGA',
            'dvi': 'DVI',
            'thunderbolt': 'Thunderbolt',
        }
        
        return mappings.get(connector_lower, connector.upper())
    
    # === Feature Extraction ===
    
    def _extract_features(self, text: str) -> list[str]:
        """
        Extract technical features from text.
        
        Args:
            text: Query text (lowercased)
            
        Returns:
            List of feature tags
            
        Examples:
            "4K HDMI cable" → ["4K"]
            "Thunderbolt 4 dock with power delivery" → ["Thunderbolt", "Power Delivery"]
        """
        features = []
        
        for feature, keywords in self.feature_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    # Use normalized feature name
                    normalized = self._normalize_feature(feature)
                    if normalized not in features:
                        features.append(normalized)
                    break
        
        return sorted(features)
    
    def _normalize_feature(self, feature: str) -> str:
        """
        Normalize feature name for display.
        
        Args:
            feature: Raw feature string
            
        Returns:
            Normalized feature name
        """
        mappings = {
            '4k': '4K',
            '8k': '8K',
            '1080p': '1080p',
            '1440p': '1440p',
            'hdr': 'HDR',
            'thunderbolt': 'Thunderbolt',
            'usb 3.0': 'USB 3.0',
            'usb 3.1': 'USB 3.1',
            'usb 3.2': 'USB 3.2',
            'hdcp': 'HDCP',
            'power delivery': 'Power Delivery',
        }
        
        return mappings.get(feature, feature.title())
    
    # === Category Extraction ===
    
    def _extract_category(self, text: str) -> Optional[str]:
        """
        Extract product category from text.
        
        Args:
            text: Query text (lowercased)
            
        Returns:
            Category name or None
            
        Examples:
            "USB cable" → "Cables"
            "HDMI adapter" → "Adapters"
            "Docking station" → "Docks"
        """
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return self._normalize_category(category)
        
        return None
    
    def _normalize_category(self, category: str) -> str:
        """
        Normalize category name for display.

        Args:
            category: Raw category string

        Returns:
            Normalized category name
        """
        return category.title()

    # === Port Count Extraction ===

    def _extract_port_count(self, text: str) -> Optional[int]:
        """
        Extract port count requirement from text.

        Args:
            text: Query text (lowercased)

        Returns:
            Port count as integer, or None if not specified

        Examples:
            "8 port switch" → 8
            "4-port hub" → 4
            "switch with 8 ports" → 8
            "16 port gigabit switch" → 16
        """
        # Pattern 1: "X port" or "X-port" (e.g., "8 port", "8-port", "4 port")
        match = re.search(r'\b(\d+)\s*-?\s*ports?\b', text)
        if match:
            return int(match.group(1))

        # Pattern 2: "with X ports" (e.g., "switch with 8 ports")
        match = re.search(r'\bwith\s+(\d+)\s+ports?\b', text)
        if match:
            return int(match.group(1))

        return None

    # === Monitor Count Extraction ===

    def _extract_min_monitors(self, text: str) -> Optional[int]:
        """
        Extract minimum monitor count requirement from text.

        Args:
            text: Query text (lowercased)

        Returns:
            Minimum monitor count as integer, or None if not specified

        Examples:
            "dock that supports 3 monitors" → 3
            "dual monitor dock" → 2
            "triple monitor setup" → 3
            "dock for 2 displays" → 2
        """
        # Pattern 1: Named counts (dual, triple, quad)
        named_counts = {
            'dual': 2, 'double': 2, 'two': 2,
            'triple': 3, 'three': 3,
            'quad': 4, 'four': 4,
        }
        for name, count in named_counts.items():
            if re.search(rf'\b{name}\b', text):
                return count

        # Pattern 2: Numeric with "monitor" or "display" (e.g., "3 monitors", "2 displays")
        match = re.search(r'\b(\d+)\s*(?:monitors?|displays?)\b', text)
        if match:
            return int(match.group(1))

        # Pattern 3: "support/supports X monitors" (e.g., "supports 3 monitors")
        match = re.search(r'\bsupports?\s+(\d+)\s*(?:monitors?|displays?)\b', text)
        if match:
            return int(match.group(1))

        return None

    # === Color Extraction ===

    def _extract_color(self, text: str) -> Optional[str]:
        """
        Extract color requirement from text.

        Args:
            text: Query text (lowercased)

        Returns:
            Normalized color name or None

        Examples:
            "red HDMI cable" → "Red"
            "black USB-C cable" → "Black"
            "grey adapter" → "Gray"
        """
        for color_keyword, normalized_color in self.COLOR_KEYWORDS.items():
            # Use word boundary to avoid matching "orange" in "storage"
            if re.search(rf'\b{color_keyword}\b', text):
                return normalized_color

        return None

    # === Keyword Extraction ===

    # Stop words to exclude from keyword extraction
    STOP_WORDS = {
        # Common words
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'can', 'need', 'want',
        'i', 'me', 'my', 'you', 'your', 'we', 'our', 'they', 'them', 'their',
        'it', 'its', 'this', 'that', 'these', 'those', 'what', 'which', 'who',
        'how', 'when', 'where', 'why', 'all', 'any', 'some', 'no', 'not',
        # Filler/placeholder words
        'one', 'ones', 'thing', 'things', 'stuff', 'type', 'kind',
        # Ordinal/positional words (product references, not keywords)
        'first', 'second', 'third', 'fourth', 'fifth',
        '1st', '2nd', '3rd', '4th', '5th',
        'last', 'middle', 'previous', 'next', 'other',
        # Question/conversational words (not product keywords)
        'support', 'supports', 'supporting', 'supported',
        'work', 'works', 'working',
        'compatible', 'compatibility',
        'come', 'comes', 'coming',
        # Negation/exclusion words (these trigger negation, not keywords)
        'except', 'without', 'avoid', 'excluding',
        # Shopping context words (already handled by category/intent)
        'looking', 'find', 'show', 'get', 'buy', 'purchase', 'order',
        'please', 'thanks', 'thank', 'help', 'like', 'something',
        # Conversational filler (very common in natural language queries)
        'just', 'got', 'new', 'trying', 'try', 'hook', 'right', 'keeps',
        'keep', 'keeping', 'cutting', 'cut', 'out', 'think', 'thinking',
        'better', 'best', 'actually', 'really', 'very', 'maybe', 'around',
        'about', 'also', 'matter', 'matters', 'use', 'using', 'used',
        'isn', 'doesn', 'wasn', 'aren', 'won', 'don', 'didn', 'hasn',
        'haven', 'wouldn', 'couldn', 'shouldn',  # Contractions without apostrophe
        'sure', 'guess', 'probably', 'seems', 'seem', 'well', 'even',
        'still', 'already', 'yet', 'ever', 'never', 'always', 'sometimes',
        # Context words that describe situation, not product
        # NOTE: 'desk' is NOT here - it's a valid product keyword for "desk mount"
        'home', 'office', 'room', 'setup', 'situation',
        'problem', 'issue', 'trouble', 'handle', 'handles', 'handling',
        'picture', 'image', 'signal',  # Symptoms, not product keywords
        # Comparative/quality words (too vague for product search)
        'quality', 'good', 'bad', 'great', 'nice', 'decent', 'proper',
    }

    # Negation patterns - words following these should be excluded from keywords
    # "but not the long ones" → "long" should NOT be a required keyword
    NEGATION_PATTERNS = [
        r'\bbut\s+not\b',
        r'\bnot\s+the\b',
        r'\bexcept\b',
        r'\bwithout\b',
        r'\bno\s+\w+\s+(?:ones?|cables?|adapters?)\b',
        r'\bavoid\b',
        r'\bdon\'?t\s+want\b',
    ]

    # Words already captured by other extraction (don't duplicate)
    ALREADY_EXTRACTED = {
        # Category words
        'cable', 'cables', 'adapter', 'adapters', 'dock', 'docking', 'station',
        'hub', 'hubs', 'switch', 'switches', 'mount', 'mounts', 'enclosure',
        'enclosures', 'splitter', 'splitters', 'converter', 'converters',
        # Connector words (handled by connector extraction)
        'usb', 'hdmi', 'displayport', 'dp', 'vga', 'dvi', 'thunderbolt',
        # Port count words (handled by port count extraction)
        # "port" must be excluded to prevent "2-port" matching "displayport"
        'port', 'ports',
        # Length words (handled by length extraction)
        'ft', 'feet', 'foot', 'm', 'meter', 'meters', 'inch', 'inches',
        # Color words (handled by color extraction)
        'black', 'white', 'gray', 'grey', 'red', 'blue', 'green', 'yellow',
        'orange', 'pink', 'purple', 'gold', 'beige', 'brown',
        # Device names (handled by device inference, not text search keywords)
        # NOTE: 'monitor', 'display', 'tv', etc. are NOT here - they're valid
        # product keywords for mounts ("monitor mount", "TV mount")
        'macbook', 'imac', 'ipad', 'iphone', 'mac', 'apple', 'pro', 'air',
        'laptop', 'computer', 'desktop', 'notebook', 'pc',
        'playstation', 'xbox', 'nintendo', 'ps5', 'ps4',
        # NOTE: 'fiber', 'optic', 'drive', 'ssd', etc. are NOT here - they must
        # remain as keywords for text matching. Category detection alone isn't
        # enough because not all products have proper category metadata.
    }

    def _extract_keywords(self, text: str) -> list[str]:
        """
        Extract significant keywords from query for text matching.

        Extracts words that aren't already captured by other extraction
        (category, connectors, length, color) and aren't stop words.
        Also excludes words following negation patterns.

        Args:
            text: Query text (lowercased)

        Returns:
            List of significant keywords for text matching

        Examples:
            "fiber optic cable" → ["fiber", "optic"]
            "monitor mount for desk" → ["monitor", "desk"]
            "hard drive enclosure" → ["hard", "drive"]
            "6ft USB-C cable" → []  (all words already extracted)
            "USB-C cables, but not the long ones" → []  (negation excluded)
        """
        text_lower = text.lower()

        # Find words that follow negation patterns - these should be excluded
        negated_words = set()
        for pattern in self.NEGATION_PATTERNS:
            # Find the negation pattern and capture words after it
            for match in re.finditer(pattern, text_lower):
                # Get the text after the negation pattern
                after_negation = text_lower[match.end():].strip()
                # Extract the next few words (up to end of clause or punctuation)
                clause_end = re.search(r'[,.]|$', after_negation)
                if clause_end:
                    negated_clause = after_negation[:clause_end.start()]
                    # Add all words from the negated clause
                    negated_words.update(re.findall(r'[a-z]+', negated_clause))

        # Tokenize: split on non-alphanumeric characters
        words = re.findall(r'[a-z0-9]+', text_lower)

        # Length unit patterns to skip (e.g., "6ft", "10m", "3meter")
        length_pattern = re.compile(r'^\d+(?:ft|feet|foot|m|meter|meters|in|inch|inches|cm)$')

        keywords = []
        for word in words:
            # Skip if too short (likely noise)
            if len(word) < 3:
                continue

            # Skip stop words
            if word in self.STOP_WORDS:
                continue

            # Skip words already extracted by other methods
            if word in self.ALREADY_EXTRACTED:
                continue

            # Skip pure numbers (lengths are handled separately)
            if word.isdigit():
                continue

            # Skip length measurements like "6ft", "10m", "3meter"
            if length_pattern.match(word):
                continue

            # Skip words that follow negation patterns
            # "but not the long ones" → "long" should not be a keyword
            if word in negated_words:
                continue

            keywords.append(word)

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        return unique_keywords

    # === Required Port Type Extraction (for Docks/Hubs) ===

    def _extract_required_port_types(
        self,
        text: str,
        category: Optional[str]
    ) -> list[str]:
        """
        Extract required port types for docks/hubs.

        When user asks for "dock with USB-C ports" or "hub with lots of USB-A",
        we need to filter results to only show products that have those port types.

        Args:
            text: Query text (lowercased)
            category: Extracted product category

        Returns:
            List of required port types (e.g., ["USB-C", "USB-A"])

        Examples:
            "docking station with USB-C ports" → ["USB-C"]
            "dock with USB-A and USB-C" → ["USB-A", "USB-C"]
            "hub with a bunch of USB-C ports" → ["USB-C"]
            "HDMI cable" → []  (not a dock/hub query)
        """
        # Only extract port types for dock/hub queries
        if not category:
            return []

        category_lower = category.lower()
        if category_lower not in ('docks', 'hubs', 'dock', 'hub'):
            return []

        port_types = []

        # Port type patterns - detect specific port mentions with "port(s)" context
        port_patterns = {
            'USB-C': [
                r'\busb[\s\-]?c\s*ports?\b',
                r'\btype[\s\-]?c\s*ports?\b',
                r'\busb[\s\-]?c\b.*\b(?:ports?|connections?)\b',
                r'\b(?:ports?|connections?)\b.*\busb[\s\-]?c\b',
                r'\bwith\s+(?:a\s+)?(?:bunch|lots?|many|multiple|several)\s+(?:of\s+)?usb[\s\-]?c\b',
            ],
            'USB-A': [
                r'\busb[\s\-]?a\s*ports?\b',
                r'\btype[\s\-]?a\s*ports?\b',
                r'\busb[\s\-]?a\b.*\b(?:ports?|connections?)\b',
                r'\b(?:ports?|connections?)\b.*\busb[\s\-]?a\b',
                r'\bwith\s+(?:a\s+)?(?:bunch|lots?|many|multiple|several)\s+(?:of\s+)?usb[\s\-]?a\b',
            ],
            'USB': [
                # Generic USB (when not USB-C or USB-A specific)
                r'\busb\s+ports?\b(?!\s*[\-]?[cCaA])',
                r'\b(?:bunch|lots?|many|multiple|several)\s+(?:of\s+)?usb\s+ports?\b',
            ],
            'HDMI': [
                r'\bhdmi\s*ports?\b',
                r'\bhdmi\b.*\b(?:ports?|outputs?|connections?)\b',
            ],
            'DisplayPort': [
                r'\b(?:displayport|display\s*port|dp)\s*ports?\b',
                r'\b(?:displayport|display\s*port)\b.*\b(?:ports?|outputs?)\b',
            ],
            'Thunderbolt': [
                r'\bthunderbolt\s*ports?\b',
                r'\bthunderbolt\b.*\b(?:ports?|connections?)\b',
            ],
            'Ethernet': [
                r'\bethernet\s*ports?\b',
                r'\brj[\s\-]?45\s*ports?\b',
            ],
        }

        for port_type, patterns in port_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    if port_type not in port_types:
                        port_types.append(port_type)
                    break

        return port_types