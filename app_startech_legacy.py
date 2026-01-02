"""
ST-Bot Streamlit App - StarTech.com Product Assistant
Powered by your 4,178 product Excel catalog

Run with: streamlit run app_startech.py
"""

import re
import time
import streamlit as st
from pathlib import Path
from excel_loader import load_startech_products, get_product_statistics
from core.context import ConversationContext, GuidancePhase
from core.intent import IntentClassifier
from core.filters import FilterExtractor
from core.search import SearchStrategy
from core.guidance import get_guidance_parser, get_setup_advisor, format_recommendation_response
from core.product_validator import get_best_cable, filter_valid_products
from core.structured_logging import (
    setup_logging, get_logger, Timer,
    log_query, log_intent, log_filters, log_search,
    log_products_shown, log_response, log_error, log_guidance
)
from llm.domain_rules import DomainRules
from llm.device_inference import DeviceInference
from llm.query_analyzer import QueryAnalyzer  # NEW: Technical requirement detection
from llm.product_ranker import ProductRanker
from llm.response_builder import ResponseBuilder
from llm.technical_question_handler import TechnicalQuestionHandler  # NEW: Technical Q&A
from llm.followup_handler import get_followup_handler  # NEW: Follow-up questions
from ui.responses import ResponseFormatter
from ui.state import SessionState
from ui.logging import ConversationLogger

# Initialize structured logging
setup_logging(
    log_dir="logs",
    console_level=20,  # INFO
    file_level=10,     # DEBUG
    enable_console=True,
    enable_file=True,
)
app_logger = get_logger("app")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Debug Mode - Set to False for production (customer-facing)
DEBUG_MODE = True  # Set to True to see debug output in responses

# If True: Shows debug info in UI and console
# If False: Clean customer-facing interface only
# =============================================================================


# App configuration
st.set_page_config(
    page_title="ST-Bot - StarTech.com Assistant",
    page_icon="🤖",
    layout="wide"
)


@st.cache_resource
def load_products(excel_path: str):
    """Load products from Excel (cached)."""
    try:
        products = load_startech_products(excel_path)
        stats = get_product_statistics(products)
        return products, stats, None
    except FileNotFoundError:
        return [], {}, f"File not found: {excel_path}"
    except Exception as e:
        return [], {}, f"Error loading Excel: {str(e)}"


def format_detailed_product_specs(prod) -> str:
    """
    Format detailed product specs in a structured, scannable layout.

    Used for explicit_sku views when user asks about a specific product.
    NOT for conversational responses or product lists.

    Structure:
    - Product name header
    - Basic Specs section (essential info)
    - Technical Details section (extended specs, if available)
    - Closing question
    """
    name = prod.metadata.get('name', prod.product_number)

    # Extract all specs
    category = prod.metadata.get('category', '')
    network_rating = prod.metadata.get('network_rating')
    network_rating_full = prod.metadata.get('network_rating_full')
    network_speed = prod.metadata.get('network_max_speed')
    length_display = prod.metadata.get('length_display', '')
    connectors = prod.metadata.get('connectors', [])
    features = prod.metadata.get('features', [])

    # Extended specs
    wire_gauge = prod.metadata.get('wire_gauge')
    connector_plating = prod.metadata.get('connector_plating')
    jacket_type = prod.metadata.get('jacket_type')
    shield_type = prod.metadata.get('shield_type')
    conductor_type = prod.metadata.get('conductor_type')
    fire_rating = prod.metadata.get('fire_rating')
    warranty = prod.metadata.get('warranty')
    color = prod.metadata.get('color')

    # Build response using markdown line breaks (two trailing spaces + newline)
    # Streamlit's st.markdown() ignores single newlines, so we use "  \n" for line breaks
    lines = []

    # Header
    lines.append(f"**{name}**")
    lines.append("")  # Blank line after header

    # Basic Specs section
    basic_specs = []
    basic_specs.append("**Basic Specs**")

    if category:
        basic_specs.append(f"Category: {category}")

    if network_rating:
        rating_display = network_rating_full if network_rating_full else network_rating
        basic_specs.append(f"Rating: {rating_display}")
        if network_speed:
            basic_specs.append(f"Max Speed: {network_speed}")

    if length_display:
        basic_specs.append(f"Length: {length_display}")

    if connectors and len(connectors) >= 2:
        basic_specs.append(f"Connectors: {connectors[0]} → {connectors[1]}")
    elif connectors:
        basic_specs.append(f"Connectors: {', '.join(connectors)}")

    if features:
        basic_specs.append(f"Features: {', '.join(features)}")

    # Join basic specs with markdown line breaks (two spaces + newline)
    lines.append("  \n".join(basic_specs))

    # Technical Details section (only if we have extended specs)
    tech_details = []
    if wire_gauge:
        tech_details.append(f"Wire Gauge: {wire_gauge}")
    if connector_plating:
        tech_details.append(f"Connector Plating: {connector_plating}")
    if shield_type:
        tech_details.append(f"Shielding: {shield_type}")
    if conductor_type:
        tech_details.append(f"Conductor: {conductor_type}")
    if jacket_type:
        tech_details.append(f"Jacket: {jacket_type}")
    if fire_rating:
        tech_details.append(f"Fire Rating: {fire_rating}")
    if color:
        tech_details.append(f"Color: {color}")
    if warranty:
        tech_details.append(f"Warranty: {warranty}")

    if tech_details:
        lines.append("")  # Blank line before section
        # Add header and join all tech details with markdown line breaks
        tech_section = ["**Technical Details**"] + tech_details
        lines.append("  \n".join(tech_section))

    # Closing
    lines.append("")  # Blank line before closing
    lines.append("Anything else you'd like to know about this product?")

    return "\n\n".join(lines)


def _extract_requirement_keywords(query: str) -> list[str]:
    """
    Extract requirement keywords from a refinement query.

    Looks for dock/product requirements like:
    - Monitor counts: "2 monitors", "dual monitor", "triple monitor"
    - Video features: "4K", "60Hz", "8K"
    - Charging: "charge", "power delivery", "PD", "100W"
    - Connectivity: "ethernet", "USB-A", "SD card"
    - Device compatibility: "MacBook", "Thunderbolt"

    Returns:
        List of requirement keywords found in the query
    """
    query_lower = query.lower()
    keywords = []

    # Monitor count patterns
    monitor_patterns = [
        (r'\b(?:dual|2|two)\s*monitors?\b', 'dual monitor'),
        (r'\b(?:triple|3|three)\s*monitors?\b', 'triple monitor'),
        (r'\bconnect\s+(\d+)\s*monitors?\b', None),  # Extract count
    ]
    for pattern, keyword in monitor_patterns:
        match = re.search(pattern, query_lower)
        if match:
            if keyword:
                keywords.append(keyword)
            else:
                # Extract the number for "connect X monitors"
                count = int(match.group(1))
                if count == 2:
                    keywords.append('dual monitor')
                elif count == 3:
                    keywords.append('triple monitor')
                elif count >= 4:
                    keywords.append(f'{count} monitors')

    # Video features
    if re.search(r'\b4k\b', query_lower):
        keywords.append('4K')
    if re.search(r'\b8k\b', query_lower):
        keywords.append('8K')
    if re.search(r'\b60\s*hz\b', query_lower):
        keywords.append('60Hz')
    if re.search(r'\b144\s*hz\b', query_lower):
        keywords.append('144Hz')

    # Charging/Power
    if re.search(r'\bcharg(?:e|ing)\b', query_lower):
        keywords.append('power delivery')
    if re.search(r'\bpower\s*delivery\b', query_lower):
        keywords.append('power delivery')
    if re.search(r'\bpd\b', query_lower):
        keywords.append('power delivery')
    if re.search(r'\b(?:100|60|45|30)w\b', query_lower):
        keywords.append('power delivery')

    # Connectivity
    if re.search(r'\bethernet\b', query_lower):
        keywords.append('ethernet')
    if re.search(r'\busb[\s-]?a\b', query_lower):
        keywords.append('USB-A')
    if re.search(r'\bsd\s*card\b', query_lower):
        keywords.append('SD card')
    if re.search(r'\bhdmi\b', query_lower):
        keywords.append('HDMI')
    if re.search(r'\bdisplayport\b', query_lower):
        keywords.append('DisplayPort')

    # Device compatibility
    if re.search(r'\bmacbook\b', query_lower):
        keywords.append('USB-C')  # MacBooks use USB-C/Thunderbolt
    if re.search(r'\bthunderbolt\b', query_lower):
        keywords.append('Thunderbolt')
    if re.search(r'\busb[\s-]?c\b', query_lower):
        keywords.append('USB-C')

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for kw in keywords:
        if kw.lower() not in seen:
            seen.add(kw.lower())
            unique.append(kw)

    return unique


def _score_product_by_requirements(product, requirements: list[str]) -> int:
    """
    Score a product based on how many requirements it matches.

    Searches both the product content (full text) and metadata features.
    For docks, also checks specific metadata fields like power_delivery,
    network_speed, DOCKNUMDISPLAYS, etc.

    Args:
        product: Product object
        requirements: List of requirement keywords

    Returns:
        Integer score (number of requirements matched)
    """
    score = 0
    content_lower = product.content.lower()
    features_lower = [f.lower() for f in product.metadata.get('features', [])]
    name_lower = product.metadata.get('name', '').lower()
    meta = product.metadata

    for req in requirements:
        req_lower = req.lower()

        # Dock-specific metadata checks
        if meta.get('category') in ('dock', 'hub'):
            # Power Delivery
            if req_lower == 'power delivery':
                pd = meta.get('power_delivery') or meta.get('hub_power_delivery')
                if pd:
                    score += 1
                    continue

            # Ethernet
            if req_lower == 'ethernet':
                if meta.get('network_speed') or 'RJ-45' in meta.get('CONNTYPE', ''):
                    score += 1
                    continue

            # Monitor count (dual monitor, 2 monitors)
            if 'monitor' in req_lower:
                num_displays = meta.get('DOCKNUMDISPLAYS')
                if num_displays:
                    num = int(float(num_displays))
                    if 'dual' in req_lower and num >= 2:
                        score += 1
                        continue
                    elif 'triple' in req_lower and num >= 3:
                        score += 1
                        continue
                    elif num > 0:
                        score += 1
                        continue

            # 4K support - use unified Product method for consistency
            if req_lower == '4k':
                if product.supports_4k():
                    score += 1
                    continue

            # USB-C connection
            if req_lower == 'usb-c':
                sub_cat = meta.get('sub_category', '').lower()
                if 'usb-c' in sub_cat or 'usb c' in sub_cat:
                    score += 1
                    continue

        # Check in content (full text)
        if req_lower in content_lower:
            score += 1
            continue

        # Check in features list
        if any(req_lower in f for f in features_lower):
            score += 1
            continue

        # Check in product name
        if req_lower in name_lower:
            score += 1
            continue

        # Special handling for multi-word requirements
        # "dual monitor" -> check for "dual" AND "monitor"
        words = req_lower.split()
        if len(words) > 1:
            if all(w in content_lower for w in words):
                score += 1
                continue

        # Special handling for "power delivery" variants (non-dock fallback)
        if req_lower == 'power delivery':
            if 'pd' in content_lower or 'power delivery' in content_lower:
                score += 1
                continue

    return score


def _format_dock_specs(dock) -> list[str]:
    """
    Extract and format dock specifications from metadata.

    Returns a list of spec lines showing:
    - Number of monitors supported
    - Power Delivery wattage
    - Ethernet (Yes/No + speed)
    - USB ports count/types
    - Video outputs
    - 4K support

    Args:
        dock: Product object with dock metadata

    Returns:
        List of formatted spec strings
    """
    specs = []
    meta = dock.metadata

    # Monitor support
    num_displays = meta.get('DOCKNUMDISPLAYS')
    if num_displays:
        num_displays = int(float(num_displays))
        specs.append(f"Monitors: {num_displays}")

    # 4K Support - use unified Product method for consistency
    if dock.supports_4k():
        specs.append("4K: Yes")
    elif num_displays:
        # Has monitor support but not 4K - check for 1080p
        max_res = meta.get('max_dvi_resolution', '')
        if '1080p' in max_res.lower():
            specs.append("Resolution: 1080p")

    # Power Delivery
    pd_wattage = meta.get('power_delivery') or meta.get('hub_power_delivery')
    if pd_wattage:
        # Clean up format like "65W" or "65"
        pd_str = str(pd_wattage).replace('W', '').strip()
        try:
            pd_val = int(float(pd_str))
            specs.append(f"Power Delivery: {pd_val}W")
        except ValueError:
            if pd_wattage:
                specs.append(f"Power Delivery: {pd_wattage}")

    # Ethernet
    network_speed = meta.get('network_speed')
    if network_speed:
        # Has ethernet
        if 'Gbps' in network_speed or '1000' in network_speed:
            specs.append("Ethernet: Gigabit")
        else:
            specs.append(f"Ethernet: {network_speed}")
    else:
        # Check CONNTYPE for RJ-45
        conn_type = meta.get('CONNTYPE', '')
        if 'RJ-45' in conn_type:
            specs.append("Ethernet: Yes")

    # USB Ports
    hub_ports = meta.get('hub_ports') or meta.get('TOTALPORTS')
    if hub_ports:
        port_count = int(float(hub_ports))
        usb_type = meta.get('hub_usb_type', 'USB')
        if 'USB 3' in str(usb_type):
            specs.append(f"USB Ports: {port_count}x USB 3.0")
        else:
            specs.append(f"USB Ports: {port_count}")

    # Video Outputs (from CONNTYPE)
    conn_type = meta.get('CONNTYPE', '')
    video_outputs = []
    if 'HDMI' in conn_type:
        # Count HDMI ports
        import re
        hdmi_match = re.search(r'(\d+)\s*x\s*HDMI', conn_type)
        if hdmi_match:
            video_outputs.append(f"{hdmi_match.group(1)}x HDMI")
        else:
            video_outputs.append("HDMI")
    if 'DisplayPort' in conn_type:
        dp_match = re.search(r'(\d+)\s*x\s*DisplayPort', conn_type)
        if dp_match:
            video_outputs.append(f"{dp_match.group(1)}x DP")
        else:
            video_outputs.append("DisplayPort")
    if 'VGA' in conn_type:
        video_outputs.append("VGA")
    if video_outputs:
        specs.append(f"Video: {', '.join(video_outputs)}")

    # Audio
    features = meta.get('features', [])
    if 'Audio' in features or 'audio' in conn_type.lower():
        specs.append("Audio: 3.5mm jack")

    # Host Connection Type (what plugs into laptop)
    host_conn = meta.get('connector_from') or meta.get('hub_host_connector')
    if host_conn:
        # Simplify the connector name
        host_simple = host_conn
        if 'USB-C' in str(host_conn) or 'Type-C' in str(host_conn):
            host_simple = 'USB-C'
        elif 'USB-A' in str(host_conn) or 'Type-A' in str(host_conn):
            host_simple = 'USB-A'
        elif 'Thunderbolt' in str(host_conn):
            host_simple = 'Thunderbolt'
        specs.append(f"Host Connection: {host_simple}")

    return specs


@st.cache_resource
def get_components(_products):
    """Initialize ST-Bot components (cached)."""
    classifier = IntentClassifier()
    filter_extractor = FilterExtractor()
    query_analyzer = QueryAnalyzer()  # NEW: Technical requirement detection
    device_inference = DeviceInference()
    product_ranker = ProductRanker()
    response_builder = ResponseBuilder()
    
    # Create search strategy with a search function that uses our product list
    search_strategy = SearchStrategy()
    
    # Create a simple search function that filters products based on criteria
    def product_search_func(filter_dict):
        """Simple product filter function."""
        results = []
        
        # DEBUG: Print what filters we're searching with
        if DEBUG_MODE:
            print(f"\nDEBUG - Searching with filters: {filter_dict}")
            
            # Find and show a USB-C to HDMI product if one exists
            usbc_sample = None
            usbc_count = 0
            usbc_indices = []
            
            for i, p in enumerate(_products):  # Check ALL products
                conn = p.metadata.get('connectors')
                if conn and len(conn) >= 2:  # Check conn is not None first
                    if 'usb' in str(conn[0]).lower() and 'c' in str(conn[0]).lower():
                        if 'hdmi' in str(conn[1]).lower():
                            usbc_count += 1
                            usbc_indices.append(i)
                            if usbc_sample is None:
                                usbc_sample = p
            
            print(f"DEBUG - Found {usbc_count} USB-C to HDMI products total")
            if usbc_indices:
                print(f"DEBUG - They appear at indices: {usbc_indices[:10]}... (showing first 10)")
            else:
                print(f"DEBUG - NO USB-C to HDMI products found at all!")
            
            # Show first 5 cable products with connectors to see data format
            print(f"\nDEBUG - Sample cable products with connectors:")
            cable_samples = 0
            for p in _products:
                if p.metadata.get('category') == 'cable':
                    conn = p.metadata.get('connectors')
                    if conn and len(conn) >= 2:
                        print(f"  {p.product_number}: {conn[0][:60]} → {conn[1][:60]}")
                        cable_samples += 1
                        if cable_samples >= 5:
                            break
            
            if usbc_sample:
                print(f"\nDEBUG - Found USB-C to HDMI sample product:")
                print(f"  SKU: {usbc_sample.product_number}")
                print(f"  Category: {usbc_sample.metadata.get('category')}")
                print(f"  Connectors: {usbc_sample.metadata.get('connectors')}")
                
                # Check if it would pass category filter
                if 'category' in filter_dict:
                    product_cat = usbc_sample.metadata.get('category', '').lower()
                    search_cat = filter_dict['category'].lower()
                    if search_cat.endswith('s'):
                        search_cat = search_cat[:-1]
                    if product_cat.endswith('s'):
                        product_cat = product_cat[:-1]
                    category_match = (product_cat == search_cat)
                    print(f"  Category match? product='{product_cat}' vs search='{search_cat}' → {category_match}")
                
                # Check if it would pass connector filter
                if 'connector_from' in filter_dict:
                    conn = usbc_sample.metadata.get('connectors')
                    if conn and len(conn) >= 1:
                        source_lower = str(conn[0]).lower()
                        search_term = filter_dict['connector_from'].lower()
                        variations = ['usb-c', 'usb c', 'type-c', 'type c', 'usb type-c']
                        matches = [v for v in variations if v in source_lower]
                        print(f"  Connector[0] lowercase: '{source_lower}'")
                        print(f"  Matches any of {variations}? {matches if matches else 'NO'}")
                
                if 'connector_to' in filter_dict:
                    conn = usbc_sample.metadata.get('connectors')
                    if conn and len(conn) >= 2:
                        target_lower = str(conn[1]).lower()
                        search_term = filter_dict['connector_to'].lower()
                        variations = ['hdmi', 'hdmi (19 pin)']
                        matches = [v for v in variations if v in target_lower]
                        print(f"  Connector[1] lowercase: '{target_lower}'")
                        print(f"  Matches any of {variations}? {matches if matches else 'NO'}")
        
        # Continue with actual search logic
        matched_count = 0
        failed_reasons = {}
        
        for product in _products:
            match = True
            fail_reason = None
            
            # Check category - use substring matching for flexibility
            # E.g., search for "switch" should match "kvm switches"
            # E.g., search for "cable" should match "digital display cables"
            if 'category' in filter_dict:
                product_cat = product.metadata.get('category', '').lower()
                search_cat = filter_dict['category'].lower()

                # Normalize: underscores to spaces, remove trailing 's' for singular form
                # Handle "switches" -> "switch" (not "switche")
                def normalize_category(word):
                    # Convert underscores to spaces for matching
                    word = word.replace('_', ' ')
                    # Singularize
                    if word.endswith('ches'):  # switches -> switch
                        return word[:-2]
                    elif word.endswith('s'):
                        return word[:-1]
                    return word

                search_norm = normalize_category(search_cat)
                product_norm = normalize_category(product_cat)

                # Match if:
                # 1. Exact match (after normalization)
                # 2. Search term is contained in product category
                #    (e.g., "switch" in "kvm switch", "cable" in "digital display cable")
                # 3. Product category contains search term
                category_match = (
                    product_norm == search_norm or
                    search_norm in product_norm or
                    product_norm in search_norm
                )

                # Special handling for KVM switches: exclude KVM cables
                # Both switches and cables have category='kvm_switch'
                # Key differentiator: actual switches have KVMPORTS, cables don't
                if category_match and 'kvm' in search_norm:
                    sub_category = product.metadata.get('sub_category', '').lower()
                    kvm_ports = product.metadata.get('kvm_ports')

                    # Explicit KVM Cables sub_category → always filter out
                    if 'cable' in sub_category:
                        category_match = False
                    # No port count = it's a cable (e.g., RKCONSUV10 in "Desktop KVMs" but no ports)
                    elif not kvm_ports:
                        category_match = False

                if not category_match:
                    match = False
            
            # Check length - convert to feet if needed
            if 'length' in filter_dict and match:
                product_length = product.metadata.get('length_ft')
                search_length = filter_dict['length']
                
                # Convert search length to feet if in meters
                if filter_dict.get('length_unit') == 'm':
                    search_length = search_length * 3.28084
                
                if not product_length or abs(product_length - search_length) > 0.5:
                    match = False
            
            # Check connectors - improved matching for verbose connector names
            if 'connector_from' in filter_dict and match:
                connectors = product.metadata.get('connectors')
                if connectors and len(connectors) >= 1:
                    source = str(connectors[0]).lower()
                    search_term = filter_dict['connector_from'].lower()
                    
                    # Handle common variations
                    search_variations = [search_term]
                    if 'usb-c' in search_term or 'usb c' in search_term:
                        search_variations.extend(['usb-c', 'usb c', 'type-c', 'type c', 'usb type-c'])
                    elif 'displayport' in search_term or 'display port' in search_term:
                        search_variations.extend(['displayport', 'display port'])
                    
                    # Check if any variation matches
                    if not any(var in source for var in search_variations):
                        match = False
                        fail_reason = f"connector_from mismatch: '{source}' doesn't contain any of {search_variations}"
                else:
                    match = False
                    fail_reason = "no connector_from data"
            
            if 'connector_to' in filter_dict and match:
                connectors = product.metadata.get('connectors')
                if connectors and len(connectors) >= 2:
                    target = str(connectors[1]).lower()
                    search_term = filter_dict['connector_to'].lower()
                    
                    # Handle common variations
                    search_variations = [search_term]
                    if 'hdmi' in search_term:
                        search_variations.extend(['hdmi', 'hdmi (19 pin)'])
                    elif 'displayport' in search_term or 'display port' in search_term:
                        search_variations.extend(['displayport', 'display port'])
                    elif 'dvi' in search_term:
                        search_variations.extend(['dvi', 'dvi-d'])
                    
                    # Check if any variation matches
                    if not any(var in target for var in search_variations):
                        match = False
                        fail_reason = f"connector_to mismatch: '{target}' doesn't contain any of {search_variations}"
                else:
                    match = False
                    fail_reason = "no connector_to data"
            
            # Check features (case-insensitive)
            # Use unified resolution methods for resolution features
            if 'features' in filter_dict and match:
                product_features = product.metadata.get('features', [])
                product_features_lower = [f.lower() for f in product_features]
                resolution_features = {'4k', '8k', '1080p', '1440p'}

                for required_feature in filter_dict['features']:
                    required_lower = required_feature.lower()
                    if required_lower in resolution_features:
                        # Use unified Product method for resolution features
                        if not product.supports_resolution(required_lower):
                            match = False
                            break
                    elif not any(required_lower in f for f in product_features_lower):
                        match = False
                        break

            # Check port count (for hubs, switches, etc.)
            if 'port_count' in filter_dict and match:
                required_ports = filter_dict['port_count']
                # Check hub_ports OR kvm_ports (KVM switches use kvm_ports)
                product_ports = product.metadata.get('hub_ports') or product.metadata.get('kvm_ports')
                if product_ports:
                    # Accept products with exactly the requested ports or more
                    if product_ports < required_ports:
                        match = False
                        fail_reason = f"port_count mismatch: has {product_ports}, need {required_ports}+"
                else:
                    # No port info - don't match if user specifically asked for ports
                    match = False
                    fail_reason = "no port count data"

            # Check color
            if 'color' in filter_dict and match:
                required_color = filter_dict['color'].lower()
                product_color = product.metadata.get('color', '').lower()
                if product_color:
                    # Exact match or partial match (e.g., "black" in "black/gray")
                    if required_color not in product_color:
                        match = False
                        fail_reason = f"color mismatch: has '{product_color}', need '{required_color}'"
                else:
                    # No color data - don't strictly filter, just lower priority
                    # Keep the match but note the missing data
                    pass

            # Check keywords (text matching on product name/content)
            # Keywords are essential for non-cable products and specific product types
            keyword_score = 0
            if 'keywords' in filter_dict and filter_dict['keywords'] and match:
                keywords = filter_dict['keywords']
                # Build searchable text from product
                search_text = (
                    product.metadata.get('name', '') + ' ' +
                    product.metadata.get('excel_category', '') + ' ' +
                    product.metadata.get('sub_category', '') + ' ' +
                    product.content
                ).lower()

                # Count how many keywords match
                for kw in keywords:
                    if kw in search_text:
                        keyword_score += 1

                # If keywords were provided but none match, exclude the product
                # This is the key fix: without keyword matching, "fiber optic cable"
                # would return ALL cables instead of just fiber cables
                if keyword_score == 0:
                    match = False
                    fail_reason = f"no keyword match: keywords={keywords}"

            if match:
                # Store keyword score for ranking
                product._keyword_score = keyword_score
                results.append(product)
                if DEBUG_MODE:
                    matched_count += 1
                    # Show first 3 matches
                    if matched_count <= 3:
                        print(f"  ✅ MATCH #{matched_count}: {product.product_number}")
            else:
                # Track failure reasons
                if DEBUG_MODE and fail_reason:
                    failed_reasons[fail_reason] = failed_reasons.get(fail_reason, 0) + 1
        
        # Print summary
        if DEBUG_MODE:
            print(f"\nDEBUG - Search results:")
            print(f"  Total matches: {len(results)}")
            if failed_reasons:
                print(f"  Top failure reasons:")
                for reason, count in sorted(failed_reasons.items(), key=lambda x: x[1], reverse=True)[:3]:
                    print(f"    - {reason}: {count} products")
        
        return results
    
    # Create a wrapper class that mimics the old ProductSearchEngine interface
    class SearchEngineWrapper:
        def __init__(self, strategy, search_func):
            self.strategy = strategy
            self.search_func = search_func
        
        def search(self, filters):
            """Search using the strategy."""
            return self.strategy.search(filters, self.search_func)
    
    search_engine = SearchEngineWrapper(search_strategy, product_search_func)
    
    domain_rules = DomainRules()
    formatter = ResponseFormatter()
    logger = ConversationLogger("startech_conversations.csv")
    
    return classifier, filter_extractor, search_engine, domain_rules, formatter, logger, device_inference, product_ranker, response_builder, query_analyzer


def save_guidance_to_session(context):
    """
    Save pending guidance to Streamlit session state as a simple dict.

    Streamlit can't serialize dataclasses with Enums properly, so we
    convert to a plain dict for persistence.
    """
    if context.pending_guidance:
        pg = context.pending_guidance
        data = {
            'setup_type': pg.setup_type,
            'monitor_count': pg.monitor_count,
            'phase': pg.phase.value,  # Convert Enum to string
            'computer_ports': pg.computer_ports,
            'computer_port_counts': pg.computer_port_counts,
            'monitor_inputs': pg.monitor_inputs,
            'preference': pg.preference,
        }
        st.session_state.pending_guidance_data = data
        # DEBUG: Show what we saved
        if DEBUG_MODE:
            st.sidebar.warning(f"🔵 SAVED guidance: phase={data['phase']}, ports={data['computer_ports']}")
    else:
        st.session_state.pending_guidance_data = None
        if DEBUG_MODE:
            st.sidebar.warning("🔵 SAVED guidance: None")


def load_guidance_from_session(context):
    """
    Load pending guidance from Streamlit session state back into context.

    Reconstructs the PendingGuidance dataclass from the stored dict.
    """
    from core.context import PendingGuidance

    data = st.session_state.get('pending_guidance_data')
    # DEBUG: Show what we're loading
    if DEBUG_MODE:
        st.sidebar.info(f"🟢 LOADING guidance data: {data}")

    if data:
        context.pending_guidance = PendingGuidance(
            setup_type=data['setup_type'],
            monitor_count=data['monitor_count'],
            phase=GuidancePhase(data['phase']),  # Convert string back to Enum
            computer_ports=data['computer_ports'],
            computer_port_counts=data['computer_port_counts'],
            monitor_inputs=data['monitor_inputs'],
            preference=data['preference'],
        )
        if DEBUG_MODE:
            st.sidebar.info(f"🟢 LOADED: phase={context.pending_guidance.phase.value}, has_pending={context.has_pending_guidance()}")
    else:
        context.pending_guidance = None
        if DEBUG_MODE:
            st.sidebar.info("🟢 LOADED: No guidance data in session")


def save_pending_question_to_session(context):
    """
    Save pending question to Streamlit session state as a simple dict.

    Streamlit can't serialize dataclasses with Enums properly, so we
    convert to a plain dict for persistence.
    """
    if context.pending_question:
        pq = context.pending_question
        data = {
            'question_type': pq.question_type.value,  # Convert Enum to string
            'context_data': pq.context_data,
        }
        st.session_state.pending_question_data = data
        if DEBUG_MODE:
            st.sidebar.warning(f"🔵 SAVED pending question: type={data['question_type']}")
    else:
        st.session_state.pending_question_data = None
        if DEBUG_MODE:
            st.sidebar.warning("🔵 SAVED pending question: None")


def load_pending_question_from_session(context):
    """
    Load pending question from Streamlit session state back into context.

    Reconstructs the PendingQuestion dataclass from the stored dict.
    """
    from core.context import PendingQuestion, PendingQuestionType

    data = st.session_state.get('pending_question_data')
    if DEBUG_MODE:
        st.sidebar.info(f"🟢 LOADING pending question data: {data}")

    if data:
        context.pending_question = PendingQuestion(
            question_type=PendingQuestionType(data['question_type']),
            context_data=data['context_data'],
        )
        if DEBUG_MODE:
            st.sidebar.info(f"🟢 LOADED pending question: type={context.pending_question.question_type.value}")
    else:
        context.pending_question = None
        if DEBUG_MODE:
            st.sidebar.info("🟢 LOADED: No pending question data in session")


def process_query(query, context, session, components):
    """Process user query and return bot response."""
    classifier, filter_extractor, search_engine, domain_rules, formatter, logger, device_inference, product_ranker, response_builder, query_analyzer = components

    # Start timing the entire request
    request_start = time.perf_counter()
    session_id = session.session_id[:16]  # Use truncated session ID for logging

    # Log incoming query
    log_query(
        session_id=session_id,
        query=query,
    )

    # DEBUG: Build debug info that will be PREPENDED to response
    debug_lines = []

    # DEBUG: Check raw session state FIRST (before any loading)
    raw_data = st.session_state.get('pending_guidance_data')
    debug_lines.append(f"🟡 RAW SESSION STATE: {raw_data}")
    debug_lines.append(f"🟡 context.pending_guidance BEFORE load: {context.pending_guidance}")

    # CRITICAL: Load guidance state from session BEFORE classifying intent
    # Streamlit can't persist dataclasses with Enums, so we use a dict intermediary
    load_guidance_from_session(context)

    # Also load pending question state (for educational follow-ups like daisy-chain)
    load_pending_question_from_session(context)

    # DEBUG: Show context state before classification
    has_pending = context.has_pending_guidance()
    debug_lines.append(f"🔴 AFTER LOAD: has_pending_guidance = {has_pending}")
    if context.pending_guidance:
        debug_lines.append(f"🔴 pending_guidance.phase = {context.pending_guidance.phase.value}")

    # Classify intent with timing
    intent_start = time.perf_counter()
    intent = classifier.classify(query, context)
    intent_time_ms = (time.perf_counter() - intent_start) * 1000

    # Log intent classification
    log_intent(
        session_id=session_id,
        query=query,
        intent=intent.type.value,
        confidence=intent.confidence,
        reasoning=intent.reasoning,
        classification_time_ms=intent_time_ms,
    )

    # DEBUG: Show classified intent
    debug_lines.append(f"🔵 INTENT CLASSIFIED: {intent.type.value}")
    debug_lines.append(f"🔵 Intent reasoning: {intent.reasoning}")
    
    # Handle different intents
    if intent.type.value == "greeting":
        response = formatter.format_greeting()
        
    elif intent.type.value == "farewell":
        response = formatter.format_farewell()

    elif intent.type.value == "install_help":
        # User is asking for installation/setup/mounting help
        # We don't provide installation support - redirect them appropriately
        response = (
            "I specialize in helping you find the right cables, adapters, and accessories, "
            "but I can't provide installation or mounting instructions.\n\n"
            "**For installation help:**\n"
            "• Check the product manual or quick start guide\n"
            "• Visit [StarTech.com Support](https://www.startech.com/support)\n"
            "• Contact StarTech.com technical support\n\n"
            "Is there a product I can help you find instead?"
        )

    elif intent.type.value == "warranty_question":
        # User is asking about warranty, returns, or RMA
        response = (
            "I specialize in helping you find the right products, but I can't answer "
            "questions about warranty, returns, or exchanges.\n\n"
            "**For warranty and returns:**\n"
            "• Visit [StarTech.com Support](https://www.startech.com/support)\n"
            "• Contact StarTech.com customer support\n\n"
            "Is there a product I can help you find instead?"
        )

    elif intent.type.value == "pricing_question":
        # User is asking about pricing, discounts, or quotes
        response = (
            "I specialize in helping you find the right products, but I can't provide "
            "pricing information, discounts, or quotes.\n\n"
            "**For pricing and quotes:**\n"
            "• Visit [StarTech.com](https://www.startech.com) for current prices\n"
            "• Contact StarTech.com sales for volume pricing or quotes\n\n"
            "Is there a product I can help you find instead?"
        )

    elif intent.type.value == "impossible_product":
        # User is asking for something technically impossible (e.g., "Bluetooth HDMI cable")
        meta = intent.meta_info or {}
        reason = meta.get('reason', 'This product combination is not technically possible')
        suggestion = meta.get('suggestion', 'Would you like help finding a related product?')

        response = (
            f"**{reason}.**\n\n"
            f"{suggestion}"
        )

    elif intent.type.value == "out_of_scope":
        # User is asking for products StarTech doesn't sell (e.g., "wireless keyboard")
        meta = intent.meta_info or {}
        category = meta.get('category', 'this type of product')
        suggestion = meta.get('suggestion', "Let me know what you're trying to connect!")

        response = (
            f"StarTech.com specializes in cables, adapters, docks, and connectivity products - "
            f"we don't sell {category}.\n\n"
            f"{suggestion}"
        )

    elif intent.type.value == "feature_search_accept":
        # User said "yes" to offered feature search (e.g., "Would you like me to find 4K cables?")
        meta = intent.meta_info or {}
        feature = meta.get('feature', '')
        product_type = meta.get('product_type', 'products')
        connector_from = meta.get('connector_from')
        connector_to = meta.get('connector_to')
        category = meta.get('category')

        if DEBUG_MODE:
            debug_lines.append(f"🔍 FEATURE_SEARCH: Searching for {feature} {product_type}")
            debug_lines.append(f"🔍 Context: connector_from={connector_from}, connector_to={connector_to}, category={category}")

        # Clear the pending feature search, stale comparison context, and pending questions
        context.clear_pending_feature_search()
        if context.has_comparison_context():
            if DEBUG_MODE:
                debug_lines.append(f"🧹 CLEARING STALE COMPARISON CONTEXT: indices={context.last_comparison_indices}")
            context.clear_comparison_context()
        if context.has_pending_question():
            if DEBUG_MODE:
                debug_lines.append(f"🧹 CLEARING STALE PENDING QUESTION: {context.pending_question.question_type}")
            context.clear_pending_question()
            save_pending_question_to_session(context)

        # Create filters based on the feature AND original context
        from core.context import SearchFilters
        filters = SearchFilters()
        filters.features = [feature]

        # Use the original category if available, otherwise infer from product_type
        if category:
            filters.product_category = category
        elif 'cable' in product_type.lower():
            filters.product_category = 'Cables'
        elif 'adapter' in product_type.lower():
            filters.product_category = 'Adapters'
        elif 'dock' in product_type.lower():
            filters.product_category = 'Docking Stations'
        elif 'hub' in product_type.lower():
            filters.product_category = 'USB Hubs'

        # IMPORTANT: Preserve the original connector context
        # If user asked for "HDMI cables with 4K", we should find HDMI cables, not USB cables
        if connector_from:
            filters.connector_from = connector_from
        if connector_to:
            filters.connector_to = connector_to

        # Perform search using the search_engine from components
        search_result = search_engine.search(filters)

        # CRITICAL: Post-filter to ensure only products with the feature are shown
        # This prevents contradiction: bot says "these don't have 4K" then shows them as 4K options
        def has_feature(product, feat: str) -> bool:
            """Check if product actually has the requested feature (case-insensitive)."""
            prod_features = product.metadata.get('features', [])
            feat_lower = feat.lower()
            return any(feat_lower in f.lower() for f in prod_features)

        filtered_products = [p for p in search_result.products if has_feature(p, feature)]

        if DEBUG_MODE:
            debug_lines.append(f"🔍 Search returned {len(search_result.products)} products, {len(filtered_products)} have '{feature}' feature")

        if filtered_products:
            # Found products - format and display them
            response = f"Here are {product_type} with **{feature}** support:\n\n"
            for i, prod in enumerate(filtered_products[:5], 1):  # Limit to 5 for readability
                sku = prod.product_number
                length = prod.metadata.get('length_display', '')
                features = prod.metadata.get('features', [])
                feature_str = ', '.join(features[:3]) if features else ''

                response += f"**{i}. {sku}**"
                if length:
                    response += f" - {length}"
                if feature_str:
                    response += f" ({feature_str})"
                response += "\n\n"  # Double newline for proper line breaks

            response = response.rstrip("\n") + "\n\n"  # Clean up trailing newlines
            response += "Would you like more details on any of these?"

            # Update context with new products (filtered list)
            context.current_products = filtered_products
            session.set_product_context(filtered_products, intent.type)
        else:
            response = (
                f"I couldn't find any {product_type} with {feature} support in our catalog.\n\n"
                f"Would you like me to search for something else?"
            )

    elif intent.type.value == "explicit_sku":
        # User mentioned a specific product SKU
        target_sku = intent.sku.upper() if intent.sku else None

        # Clear stale comparison context and pending questions - we're looking at a specific product now
        if context.has_comparison_context():
            if DEBUG_MODE:
                debug_lines.append(f"🧹 CLEARING STALE COMPARISON CONTEXT: indices={context.last_comparison_indices}")
            context.clear_comparison_context()
        if context.has_pending_question():
            if DEBUG_MODE:
                debug_lines.append(f"🧹 CLEARING STALE PENDING QUESTION: {context.pending_question.question_type}")
            context.clear_pending_question()
            save_pending_question_to_session(context)

        if DEBUG_MODE:
            debug_lines.append(f"🔍 EXPLICIT_SKU: Looking for SKU '{target_sku}'")

        # First, check if SKU is in current product context
        found_in_context = None
        if context.current_products:
            for prod in context.current_products:
                if prod.product_number.upper() == target_sku:
                    found_in_context = prod
                    break

        if found_in_context:
            # Product is in context - show detailed specs using structured format
            if DEBUG_MODE:
                debug_lines.append(f"✅ Found SKU in context: {found_in_context.product_number}")

            response = format_detailed_product_specs(found_in_context)
        else:
            # SKU not in context - search for it
            if DEBUG_MODE:
                debug_lines.append(f"🔍 SKU not in context, searching catalog")

            # Get all products from session and search for the SKU
            all_products_list = st.session_state.get('_all_products', [])
            found_product = None

            for prod in all_products_list:
                if prod.product_number.upper() == target_sku:
                    found_product = prod
                    break

            if found_product:
                # Found the product - show it using structured format
                response = format_detailed_product_specs(found_product)

                # Set product in context for follow-up questions
                context.set_multi_products([found_product])
                session.set_product_context([found_product], intent.type)
            else:
                # Product not found in catalog
                response = f"I couldn't find product '{target_sku}' in our catalog. Could you double-check the SKU or describe what you're looking for?"

    elif intent.type.value == "setup_guidance":
        # Complex query needing diagnostic questions (e.g., multi-monitor setup)
        # DO NOT search for products - instead, ask clarifying questions

        # CONTEXT CLEANUP: Clear stale offers when user starts a new guidance flow
        # This prevents "Would you like 4K cables?" offers from polluting dock guidance
        if context.has_pending_feature_search():
            if DEBUG_MODE:
                debug_lines.append(f"🧹 CLEARING STALE FEATURE SEARCH OFFER: {context.pending_feature_search}")
            context.clear_pending_feature_search()

        # Clear stale comparison context and pending questions - starting fresh guidance flow
        if context.has_comparison_context():
            if DEBUG_MODE:
                debug_lines.append(f"🧹 CLEARING STALE COMPARISON CONTEXT: indices={context.last_comparison_indices}")
            context.clear_comparison_context()
        if context.has_pending_question():
            if DEBUG_MODE:
                debug_lines.append(f"🧹 CLEARING STALE PENDING QUESTION: {context.pending_question.question_type}")
            context.clear_pending_question()
            save_pending_question_to_session(context)

        setup_type = intent.meta_info.get('type', 'unknown')
        monitor_count = intent.meta_info.get('monitor_count')

        # Log guidance start
        log_guidance(
            session_id=session_id,
            setup_type=setup_type,
            phase="initial_questions",
            monitor_count=monitor_count,
        )

        # Start pending guidance in context so we can track the conversation
        context.start_guidance(setup_type, monitor_count)

        # For single_monitor: Check if the query ALREADY contains the needed info
        # e.g., "My computer has USB-C and HDMI. My monitor has HDMI. 4 feet apart"
        # If so, skip questions and go directly to recommendation
        skip_to_recommendation = False
        if setup_type == 'single_monitor':
            guidance_parser = get_guidance_parser()
            pending = context.pending_guidance
            if pending:
                # Try to parse the original query as if it were an answer
                pending = guidance_parser.parse_response(query, pending)
                debug_lines.append(f"📝 PRE-PARSE: computer={pending.computer_ports}, monitor={pending.monitor_inputs}, length={pending.cable_length}")

                # If we got both computer and monitor ports, skip to recommendations
                if pending.computer_ports and pending.monitor_inputs:
                    debug_lines.append(f"✅ SINGLE_MONITOR: Query already has all info, skipping questions")
                    pending.phase = GuidancePhase.READY_TO_RECOMMEND
                    context._pending_guidance = pending
                    skip_to_recommendation = True

        # CRITICAL: Save guidance to session state for persistence across Streamlit reruns
        save_guidance_to_session(context)

        # DEBUG: Verify save worked and add to debug output
        verify_data = st.session_state.get('pending_guidance_data')
        debug_lines.append(f"✅ SETUP_GUIDANCE SAVE: data={verify_data}")

        if skip_to_recommendation:
            # Handle single_monitor recommendation directly (same logic as in SETUP_FOLLOWUP)
            pending = context.pending_guidance
            debug_lines.append(f"📺 SINGLE MONITOR (direct): computer={pending.computer_ports}, monitor={pending.monitor_inputs}, length={pending.cable_length}")

            # Determine best port to use - PREFER DIRECT CONNECTION
            monitor_port = pending.monitor_inputs[0] if pending.monitor_inputs else None
            computer_ports = pending.computer_ports or []

            selected_port = None
            if monitor_port and monitor_port in computer_ports:
                selected_port = monitor_port
                debug_lines.append(f"📺 Direct match found! Using {selected_port}-to-{monitor_port}")
            elif computer_ports:
                selected_port = computer_ports[0]
                debug_lines.append(f"📺 No direct match, using {selected_port}-to-{monitor_port}")

            # Build filters
            filters = filter_extractor.extract("")
            if selected_port:
                filters.connector_from = selected_port
            if monitor_port:
                filters.connector_to = monitor_port

            # Apply length preference
            user_requested_length = None
            if pending.cable_length:
                if pending.cable_length == 'short':
                    filters.length_max = 6.0
                    user_requested_length = 3.0
                elif pending.cable_length == 'long':
                    filters.length_min = 10.0
                    user_requested_length = 15.0
                elif 'ft' in pending.cable_length:
                    try:
                        length_val = float(pending.cable_length.split()[0])
                        filters.length_min = length_val
                        filters.length_max = length_val + 6.0
                        user_requested_length = length_val + 2.0
                    except ValueError:
                        pass

            # Search for cables
            results = search_engine.search(filters)
            debug_lines.append(f"📺 Found {len(results.products)} products")

            # Build response
            source = selected_port or "unknown"
            target = monitor_port or "unknown"

            if source == target:
                cable_desc = f"{source} cable"
                intro_note = f"Since both your computer and monitor have {source}, you just need a standard {source} cable."
            else:
                cable_desc = f"{source} to {target} cable"
                intro_note = f"For connecting your {source} port to your {target} monitor, you need a {cable_desc}."

            response_parts = []
            response_parts.append(f"**Perfect!** {intro_note}")
            response_parts.append("")

            if results.products:
                preferred_length = user_requested_length if user_requested_length else 6.0
                best_product = get_best_cable(results.products, source, target, preferred_length_ft=preferred_length)

                if best_product:
                    name = best_product.metadata.get('name', best_product.product_number)
                    length = best_product.metadata.get('length_display', '')
                    sku = best_product.product_number
                    resolution = best_product.metadata.get('max_resolution', '')
                    features = best_product.metadata.get('features', [])

                    response_parts.append(f"**Recommended:** {name}")
                    response_parts.append(f"- SKU: {sku}")
                    if length:
                        response_parts.append(f"- Length: {length}")
                    if resolution:
                        response_parts.append(f"- Supports: {resolution}")
                    if features:
                        response_parts.append(f"- Features: {', '.join(features[:3])}")
                    response_parts.append("")

                    if source != target:
                        response_parts.append(f"_This cable converts the digital signal from {source} to {target}._")

                    context.set_single_product(best_product)
                    session.set_product_context([best_product], intent.type)
                else:
                    response_parts.append(f"I found adapters but no direct cables. Search for '{cable_desc}' to see all options.")
            else:
                response_parts.append(
                    f"I couldn't find an exact {cable_desc}. "
                    f"Could you tell me more about your setup? For example, what ports does your computer have?"
                )

            response = "\n".join(response_parts)
            pending.phase = GuidancePhase.COMPLETE
            save_guidance_to_session(context)
        else:
            # Show guidance questions
            response = formatter.format_setup_guidance(setup_type, intent.meta_info or {})

    elif intent.type.value == "setup_followup":
        # User is answering guidance questions - parse their response
        guidance_parser = get_guidance_parser()
        setup_advisor = get_setup_advisor()

        # Get pending guidance
        pending = context.pending_guidance
        if pending:
            # For OFFERED_DOCK phase, don't parse - just check the phase directly
            # For other phases, parse the user's answer
            if pending.phase != GuidancePhase.OFFERED_DOCK:
                # Parse response (handles both single-line and multi-line)
                pending = guidance_parser.parse_response(query, pending)

                # DEBUG: Show what was parsed
                debug_lines.append(f"📝 AFTER PARSE: phase={pending.phase.value}")
                debug_lines.append(f"📝 computer_ports={pending.computer_ports}")
                debug_lines.append(f"📝 monitor_inputs={pending.monitor_inputs}")
                debug_lines.append(f"📝 preference={pending.preference}")
            else:
                debug_lines.append(f"📝 OFFERED_DOCK phase - skipping parse, handling yes/no")

            # Check what phase we're in after parsing
            if pending.phase == GuidancePhase.PORT_COUNT_CLARIFICATION:
                # Need to ask for port counts
                response = setup_advisor.get_clarification_question(pending)
                # Save updated guidance state
                save_guidance_to_session(context)

            elif pending.phase == GuidancePhase.READY_TO_RECOMMEND:
                # Check if this is dock_selection - handle separately
                if pending.setup_type == 'dock_selection':
                    # Import dock-specific helpers
                    from core.guidance import format_dock_recommendation_intro, build_dock_search_query

                    # Build intro with user's parsed requirements
                    intro_text = format_dock_recommendation_intro(pending)

                    # Build search query based on parsed requirements
                    dock_query = build_dock_search_query(pending)
                    debug_lines.append(f"🔍 DOCK SEARCH: query='{dock_query}'")

                    # Search for matching docks - use "dock" category (normalized value)
                    dock_filters = filter_extractor.extract(dock_query)
                    dock_filters.product_category = "dock"  # Normalized category value

                    dock_results = search_engine.search(dock_filters)
                    debug_lines.append(f"🔍 DOCK SEARCH: found {len(dock_results.products)} products")

                    # If no results, try with "hub" category (some docks are categorized as hubs)
                    if not dock_results.products:
                        debug_lines.append("🔍 DOCK SEARCH: No docks found, trying hubs")
                        dock_filters.product_category = "hub"
                        dock_results = search_engine.search(dock_filters)
                        debug_lines.append(f"🔍 DOCK SEARCH: found {len(dock_results.products)} hubs")

                    # NEVER remove category filter - we only want docks/hubs, not cables
                    # Filter to ensure only actual docks/hubs are shown
                    dock_products = [
                        p for p in dock_results.products
                        if p.metadata.get('category') in ('dock', 'hub')
                    ]
                    debug_lines.append(f"🔍 DOCK SEARCH: after category filter: {len(dock_products)} products")

                    if dock_products:
                        # Rank and get top 3
                        ranked_products = product_ranker.get_top_n(
                            products=dock_products,
                            query=dock_query,
                            n=3
                        )
                        top_docks = [rp.product for rp in ranked_products]

                        # Format response with intro and products
                        response = intro_text + "\n"
                        for i, dock in enumerate(top_docks, 1):
                            name = dock.metadata.get('name', dock.product_number)
                            sku = dock.product_number
                            sub_category = dock.metadata.get('sub_category', '')

                            response += f"**{i}. {name}** ({sku})\n"

                            # Show comprehensive dock specs
                            specs = _format_dock_specs(dock)
                            if specs:
                                for spec in specs:
                                    response += f"   - {spec}\n"
                            elif sub_category:
                                response += f"   - Type: {sub_category}\n"
                            response += "\n"

                        response += "Would you like more details on any of these docks?"

                        # Update context
                        context.set_multi_products(top_docks)
                        session.set_product_context(top_docks, intent.type)
                    else:
                        # No exact match - show all docks and explain
                        all_docks = [p for p in all_products_list if p.metadata.get('category') in ('dock', 'hub')]
                        if all_docks:
                            all_docks = sorted(all_docks, key=lambda p: p.score, reverse=True)[:3]
                            response = intro_text + "\n"
                            response += "I couldn't find an exact match for all your criteria, but here are some popular options:\n\n"
                            for i, dock in enumerate(all_docks, 1):
                                sku = dock.product_number
                                specs = _format_dock_specs(dock)
                                response += f"**{i}. {sku}**\n"
                                if specs:
                                    response += f"   - {', '.join(specs)}\n\n"
                            response += "Would you like more details on any of these?"
                            context.set_multi_products(all_docks)
                            session.set_product_context(all_docks, intent.type)
                        else:
                            response = (
                                intro_text + "\n" +
                                "I'm having trouble finding docks right now. "
                                "Could you tell me more about what features are most important to you?"
                            )

                    # Complete the guidance
                    pending.phase = GuidancePhase.COMPLETE
                    save_guidance_to_session(context)

                elif pending.setup_type == 'kvm_selection':
                    # KVM switch recommendation flow
                    debug_lines.append(f"🔌 KVM SELECTION: Starting KVM recommendation")
                    debug_lines.append(f"🔌 KVM PARAMS: port_count={pending.kvm_port_count}, video={pending.kvm_video_type}, usb={pending.kvm_usb_switching}")

                    # Build intro based on user's requirements
                    intro_parts = ["Based on your requirements"]
                    if pending.kvm_port_count:
                        intro_parts.append(f"({pending.kvm_port_count}-port")
                    if pending.kvm_video_type:
                        if pending.kvm_port_count:
                            intro_parts.append(f"{pending.kvm_video_type}")
                        else:
                            intro_parts.append(f"({pending.kvm_video_type}")
                    if pending.kvm_port_count or pending.kvm_video_type:
                        intro_parts.append("KVM)")
                    intro_text = " ".join(intro_parts) + ", here are KVM switches that match:\n"

                    # Get all products and filter to KVM switches
                    all_products_list = st.session_state.get('_all_products', [])

                    # Filter to actual KVM switches (not cables)
                    # Category is 'kvm_switch', use subcategory to exclude "KVM Cables"
                    kvm_switches = [
                        p for p in all_products_list
                        if p.metadata.get('category') == 'kvm_switch' and
                        p.metadata.get('sub_category') in ('Desktop KVMs', 'Enterprise KVMs', 'KVM Extenders')
                    ]
                    debug_lines.append(f"🔌 KVM SEARCH: Found {len(kvm_switches)} KVM switches (excluding cables)")

                    # Filter by port count if specified
                    if pending.kvm_port_count:
                        # Check KVMPORTS field in metadata
                        filtered_by_ports = []
                        for p in kvm_switches:
                            kvm_ports = p.metadata.get('kvm_ports')
                            if kvm_ports and int(kvm_ports) == pending.kvm_port_count:
                                filtered_by_ports.append(p)
                        if filtered_by_ports:
                            kvm_switches = filtered_by_ports
                            debug_lines.append(f"🔌 KVM SEARCH: After port filter ({pending.kvm_port_count}): {len(kvm_switches)}")

                    # Filter by video type if specified
                    # IMPORTANT: Prefer EXACT matches first (e.g., "HDMI" only, not "USB-C + HDMI")
                    if pending.kvm_video_type:
                        video_type_lower = pending.kvm_video_type.lower()
                        exact_video_matches = []
                        partial_video_matches = []

                        for p in kvm_switches:
                            # Check kvm_video_type metadata field (from KVMPCVIDEO column)
                            kvm_video = p.metadata.get('kvm_video_type', '').lower()
                            name = p.metadata.get('name', '').lower()
                            content = p.content.lower() if p.content else ''
                            sku = p.product_number.lower()

                            # Exact match: KVMPCVIDEO is exactly the requested type (e.g., "HDMI" not "USB-C + HDMI")
                            if kvm_video == video_type_lower:
                                exact_video_matches.append(p)
                            elif (video_type_lower in kvm_video or
                                  video_type_lower in name or
                                  video_type_lower in content or
                                  video_type_lower in sku):
                                partial_video_matches.append(p)

                        # Prefer exact matches, but include partial matches if needed
                        if exact_video_matches:
                            kvm_switches = exact_video_matches
                            debug_lines.append(f"🔌 KVM SEARCH: After exact video filter ({pending.kvm_video_type}): {len(kvm_switches)}")
                        elif partial_video_matches:
                            kvm_switches = partial_video_matches
                            debug_lines.append(f"🔌 KVM SEARCH: After partial video filter ({pending.kvm_video_type}): {len(kvm_switches)}")

                    # Take top 3 by score
                    kvm_switches = sorted(kvm_switches, key=lambda p: p.score, reverse=True)[:3]

                    if kvm_switches:
                        response = intro_text + "\n"
                        for i, kvm in enumerate(kvm_switches, 1):
                            sku = kvm.product_number
                            kvm_ports = kvm.metadata.get('kvm_ports')
                            kvm_video = kvm.metadata.get('kvm_video_type', '')
                            kvm_interface = kvm.metadata.get('kvm_interface', '')
                            kvm_audio = kvm.metadata.get('kvm_audio', False)
                            kvm_cables = kvm.metadata.get('kvm_cables_included', False)
                            features = kvm.metadata.get('features', [])

                            response += f"**{i}. {sku}**\n"

                            # Computer Inputs (based on video type and port count)
                            num_ports = int(kvm_ports) if kvm_ports else 2
                            if kvm_video:
                                # Check if this is a mixed-port KVM (e.g., "USB-C + HDMI")
                                if '+' in kvm_video:
                                    # Mixed ports - explain the configuration
                                    response += f"   • Computer Inputs: {kvm_video} (flexible connections)\n"
                                else:
                                    # Single video type - all ports same type
                                    response += f"   • Computer Inputs: {num_ports}x {kvm_video}\n"

                            # Monitor Output (typically 1 output of the video type)
                            if kvm_video:
                                # For mixed KVMs, output is usually HDMI
                                if '+' in kvm_video and 'hdmi' in kvm_video.lower():
                                    response += f"   • Monitor Output: 1x HDMI\n"
                                elif '+' not in kvm_video:
                                    response += f"   • Monitor Output: 1x {kvm_video}\n"

                            # USB Switching - ALWAYS show if user requested it
                            if pending.kvm_usb_switching:
                                if kvm_interface and 'usb' in kvm_interface.lower():
                                    response += f"   • USB Switching: ✓ (keyboard, mouse, peripherals)\n"
                                else:
                                    response += f"   • USB Switching: ✓\n"

                            # Audio
                            if kvm_audio:
                                response += f"   • Audio: ✓\n"

                            # Cables included
                            if kvm_cables:
                                response += f"   • Cables Included: ✓\n"

                            # Resolution from features
                            if features:
                                res_features = [f for f in features if '4k' in f.lower() or '8k' in f.lower()]
                                if res_features:
                                    response += f"   • Resolution: {res_features[0]}\n"

                            response += "\n"

                        # Add helpful note about USB switching
                        if pending.kvm_usb_switching:
                            response += "_All these KVMs let you switch your keyboard, mouse, and USB devices between computers with one button press._\n\n"
                        else:
                            response += "_KVM switches include cables to connect your computers. Just plug in and you're ready to switch!_\n\n"

                        response += "Would you like more details on any of these?"

                        # Update context
                        context.set_multi_products(kvm_switches)
                        session.set_product_context(kvm_switches, intent.type)
                    else:
                        # No matches with strict filters - relax and try again
                        debug_lines.append("🔌 KVM SEARCH: No exact matches, relaxing filters")

                        # Try without port count filter
                        relaxed_kvm = [
                            p for p in all_products_list
                            if p.metadata.get('category') == 'kvm_switch' and
                            p.metadata.get('sub_category') in ('Desktop KVMs', 'Enterprise KVMs', 'KVM Extenders')
                        ]

                        # Still apply video filter if specified
                        if pending.kvm_video_type and relaxed_kvm:
                            video_type_lower = pending.kvm_video_type.lower()
                            relaxed_kvm = [
                                p for p in relaxed_kvm
                                if video_type_lower in p.metadata.get('kvm_video_type', '').lower() or
                                video_type_lower in p.metadata.get('name', '').lower() or
                                video_type_lower in (p.content or '').lower()
                            ]

                        relaxed_kvm = sorted(relaxed_kvm, key=lambda p: p.score, reverse=True)[:3]

                        if relaxed_kvm:
                            response = intro_text + "\n"
                            port_note = f"We don't have an exact {pending.kvm_port_count}-port match, but these should work:\n\n" if pending.kvm_port_count else ""
                            response += port_note

                            for i, kvm in enumerate(relaxed_kvm, 1):
                                sku = kvm.product_number
                                kvm_ports = kvm.metadata.get('kvm_ports')
                                kvm_video = kvm.metadata.get('kvm_video_type', '')
                                kvm_interface = kvm.metadata.get('kvm_interface', '')
                                kvm_audio = kvm.metadata.get('kvm_audio', False)
                                kvm_cables = kvm.metadata.get('kvm_cables_included', False)
                                features = kvm.metadata.get('features', [])

                                response += f"**{i}. {sku}**\n"

                                # Computer Inputs
                                num_ports = int(kvm_ports) if kvm_ports else 2
                                if kvm_video:
                                    if '+' in kvm_video:
                                        response += f"   • Computer Inputs: {kvm_video} (flexible connections)\n"
                                    else:
                                        response += f"   • Computer Inputs: {num_ports}x {kvm_video}\n"

                                # Monitor Output
                                if kvm_video:
                                    if '+' in kvm_video and 'hdmi' in kvm_video.lower():
                                        response += f"   • Monitor Output: 1x HDMI\n"
                                    elif '+' not in kvm_video:
                                        response += f"   • Monitor Output: 1x {kvm_video}\n"

                                # USB Switching
                                if pending.kvm_usb_switching:
                                    response += f"   • USB Switching: ✓\n"

                                # Audio
                                if kvm_audio:
                                    response += f"   • Audio: ✓\n"

                                response += "\n"

                            response += "Would you like more details on any of these?"
                            context.set_multi_products(relaxed_kvm)
                            session.set_product_context(relaxed_kvm, intent.type)
                        else:
                            # Truly no KVM switches found - shouldn't happen with real data
                            response = (
                                "I'm having trouble finding KVM switches right now. "
                                "Could you tell me more about your setup? For example:\n\n"
                                "- What video connections do your computers have? (HDMI, DisplayPort, VGA)\n"
                                "- How many computers do you need to switch between?"
                            )

                    # Complete the guidance
                    pending.phase = GuidancePhase.COMPLETE
                    save_guidance_to_session(context)

                elif pending.setup_type == 'single_monitor':
                    # Single monitor connection - simpler flow
                    debug_lines.append(f"📺 SINGLE MONITOR: computer={pending.computer_ports}, monitor={pending.monitor_inputs}, length={pending.cable_length}")

                    # Determine best port to use
                    # PREFER DIRECT CONNECTION: If computer has same port as monitor, use it!
                    # Example: Computer has USB-C and HDMI, Monitor has HDMI → use HDMI (direct)
                    monitor_port = pending.monitor_inputs[0] if pending.monitor_inputs else None
                    computer_ports = pending.computer_ports or []

                    # Check for direct match first (simplest solution)
                    selected_port = None
                    if monitor_port and monitor_port in computer_ports:
                        selected_port = monitor_port
                        debug_lines.append(f"📺 SINGLE MONITOR: Direct match found! Using {selected_port}-to-{monitor_port}")
                    elif computer_ports:
                        # No direct match - use first available port
                        selected_port = computer_ports[0]
                        debug_lines.append(f"📺 SINGLE MONITOR: No direct match, using {selected_port}-to-{monitor_port}")

                    # Build filters for the cable search
                    filters = filter_extractor.extract("")
                    if selected_port:
                        filters.connector_from = selected_port
                    if monitor_port:
                        filters.connector_to = monitor_port

                    # Apply length preference if specified
                    # IMPORTANT: Cable must be AT LEAST as long as user needs (can't stretch a short cable!)
                    user_requested_length = None
                    if pending.cable_length:
                        if pending.cable_length == 'short':
                            filters.length_max = 6.0  # Max 6ft for "short"
                            user_requested_length = 3.0
                        elif pending.cable_length == 'long':
                            filters.length_min = 10.0  # Min 10ft for "long"
                            user_requested_length = 15.0
                        elif 'ft' in pending.cable_length:
                            try:
                                length_val = float(pending.cable_length.split()[0])
                                # Cable must be AT LEAST as long as user needs, preferably slightly longer
                                filters.length_min = length_val  # Minimum = what they need
                                filters.length_max = length_val + 6.0  # Allow up to 6ft extra
                                user_requested_length = length_val + 2.0  # Prefer slightly longer than exact
                            except ValueError:
                                pass
                        elif 'm' in pending.cable_length:
                            try:
                                length_val = float(pending.cable_length.split()[0])
                                # Convert meters to feet (approx)
                                length_ft = length_val * 3.28
                                filters.length_min = length_ft  # Minimum = what they need
                                filters.length_max = length_ft + 6.0
                                user_requested_length = length_ft + 2.0
                            except ValueError:
                                pass

                    # Search for matching cables
                    results = search_engine.search(filters)
                    debug_lines.append(f"📺 SINGLE MONITOR: found {len(results.products)} products")

                    # Build response using selected port
                    source = selected_port or "unknown"
                    target = monitor_port or "unknown"

                    if source == target:
                        cable_desc = f"{source} cable"
                        intro_note = f"Since both your computer and monitor have {source}, you just need a standard {source} cable."
                    else:
                        cable_desc = f"{source} to {target} cable"
                        intro_note = f"For connecting your {source} port to your {target} monitor, you need a {cable_desc}."

                    response_parts = []
                    response_parts.append(f"**Perfect!** {intro_note}")
                    response_parts.append("")

                    if results.products:
                        # Get best cable using existing helper
                        # Use user's requested length (with buffer) or default to 6ft
                        preferred_length = user_requested_length if user_requested_length else 6.0

                        best_product = get_best_cable(
                            results.products,
                            source,
                            target,
                            preferred_length_ft=preferred_length
                        )

                        if best_product:
                            name = best_product.metadata.get('name', best_product.product_number)
                            length = best_product.metadata.get('length_display', '')
                            sku = best_product.product_number
                            resolution = best_product.metadata.get('max_resolution', '')
                            features = best_product.metadata.get('features', [])

                            response_parts.append(f"**Recommended:** {name}")
                            response_parts.append(f"- SKU: {sku}")
                            if length:
                                response_parts.append(f"- Length: {length}")
                            if resolution:
                                response_parts.append(f"- Supports: {resolution}")
                            if features:
                                response_parts.append(f"- Features: {', '.join(features[:3])}")
                            response_parts.append("")

                            # Add educational note for adapters
                            if source != target:
                                response_parts.append(f"_This cable converts the digital signal from {source} to {target}._")

                            # Update context with product
                            context.set_single_product(best_product)
                            session.set_product_context([best_product], intent.type)
                        else:
                            response_parts.append(f"I found adapters but no direct cables. Search for '{cable_desc}' to see all options.")
                    else:
                        response_parts.append(
                    f"I couldn't find an exact {cable_desc}. "
                    f"Could you tell me more about your setup? For example, what ports does your computer have?"
                )

                    response = "\n".join(response_parts)

                    # Complete the guidance
                    pending.phase = GuidancePhase.COMPLETE
                    save_guidance_to_session(context)

                else:
                    # Multi-monitor recommendation flow
                    recommendation = setup_advisor.recommend(pending)

                    # Build response with clear port mapping and educational explanations
                    response_parts = []
                    response_parts.append("Perfect! Here's your setup:")
                    response_parts.append("")

                    # Preferred length for all monitor cables (consistent)
                    preferred_length_ft = 6.0

                    # Search for products for each requirement
                    all_products = []
                    for req in recommendation.requirements:
                        # Build filters for this specific requirement
                        filters = filter_extractor.extract("")
                        filters.connector_from = req.source_port
                        filters.connector_to = req.target_input

                        # Search
                        results = search_engine.search(filters)

                        # Format header with explicit port mapping
                        response_parts.append(f"**Monitor {req.monitor_number} ({req.target_input} input):**")

                        # Explicit connection path
                        if req.source_port == req.target_input:
                            response_parts.append(
                                f"Connect: Laptop's {req.source_port} → {req.source_port} cable → Monitor's {req.target_input}"
                            )
                        else:
                            response_parts.append(
                                f"Connect: Laptop's {req.source_port} → {req.source_port} to {req.target_input} cable → Monitor's {req.target_input}"
                            )

                        if results.products:
                            # Find best cable with consistent length preference
                            best_product = get_best_cable(
                                results.products,
                                req.source_port,
                                req.target_input,
                                preferred_length_ft=preferred_length_ft
                            )

                            if best_product:
                                all_products.append(best_product)

                                # Format product recommendation
                                name = best_product.metadata.get('name', best_product.product_number)
                                length = best_product.metadata.get('length_display', '')
                                sku = best_product.product_number

                                response_parts.append(f"Recommended: **{name}** ({sku})")
                                if length:
                                    response_parts.append(f"Length: {length}")

                                # Why this works explanation
                                response_parts.append(f"_Why: {req.explanation}_")
                            else:
                                response_parts.append(
                                    f"_No suitable cable found - search for '{req.source_port} to {req.target_input} cable'_"
                                )
                        else:
                            response_parts.append(
                                f"_No exact match found - search for '{req.source_port} to {req.target_input}'_"
                            )
                        response_parts.append("")

                    # Add setup efficiency note if we solved all monitors
                    if recommendation.is_complete and len(recommendation.requirements) > 1:
                        # Check if we used different laptop ports
                        ports_used = set(req.source_port for req in recommendation.requirements)
                        if len(ports_used) > 1:
                            response_parts.append(
                                f"_This setup uses {len(ports_used)} different laptop ports efficiently - one for each monitor._"
                            )
                            response_parts.append("")

                    # If incomplete solution, explain what else is needed
                    if not recommendation.is_complete:
                        response_parts.append("")
                        unsolved = ", ".join([f"Monitor {n}" for n in recommendation.unsolved_monitors])
                        response_parts.append(f"**For {unsolved}:** You'll need a USB-C docking station or hub first.")
                        response_parts.append("  _A dock adds additional video outputs to your computer._")
                        response_parts.append("")
                        response_parts.append("Would you like me to show you USB-C docking stations?")
                        pending.phase = GuidancePhase.OFFERED_DOCK
                    else:
                        response_parts.append("Need different lengths or want to see alternatives?")
                        pending.phase = GuidancePhase.COMPLETE

                    response = "\n".join(response_parts)

                    # Update context with found products
                    if all_products:
                        context.set_multi_products(all_products)
                        session.set_product_context(all_products, intent.type)

                    # Save guidance state (may be OFFERED_DOCK or COMPLETE)
                    save_guidance_to_session(context)

            elif pending.phase == GuidancePhase.OFFERED_DOCK:
                # User is responding to our offer to show docking stations
                response_lower = query.lower()

                # Check for affirmative response
                is_yes = any(word in response_lower for word in [
                    'yes', 'yeah', 'yep', 'sure', 'please', 'show', 'ok', 'okay'
                ])

                if is_yes:
                    # Search for USB-C docking stations
                    debug_lines.append("🔍 DOCK SEARCH: User said yes, searching for docks")

                    # Get all products and filter to USB-C docking stations
                    # Note: Docks don't have connector data, so we filter by category and name
                    all_products_list = st.session_state.get('_all_products', [])

                    # Filter to USB-C docking stations
                    docks = []
                    for p in all_products_list:
                        category = p.metadata.get('category', '').lower()
                        name = p.metadata.get('name', '').lower()
                        sku = p.product_number.lower()

                        # Must be a dock
                        is_dock = 'dock' in category or 'docking' in name

                        # Prefer USB-C docks (check SKU patterns or name)
                        is_usbc = ('usb-c' in name or 'usbc' in name or
                                   'type-c' in name or 'typec' in name or
                                   sku.startswith('dk3') or  # DK30, DK31 are USB-C docks
                                   'thunderbolt' in name.lower())

                        if is_dock and is_usbc:
                            docks.append(p)

                    debug_lines.append(f"🔍 DOCK SEARCH: Found {len(docks)} USB-C docking stations")

                    if docks:
                        # Show top 3 docks
                        top_docks = docks[:3]

                        response_parts = [
                            "Here are some USB-C docking stations that would work for your multi-monitor setup:",
                            ""
                        ]

                        for i, dock in enumerate(top_docks, 1):
                            name = dock.metadata.get('name', dock.product_number)
                            sku = dock.product_number

                            response_parts.append(f"**{i}. {name}** ({sku})")

                            # Show comprehensive dock specs
                            specs = _format_dock_specs(dock)
                            if specs:
                                for spec in specs:
                                    response_parts.append(f"   - {spec}")

                        response_parts.append("")
                        response_parts.append("These docks connect to your USB-C port and provide multiple video outputs for your monitors.")
                        response_parts.append("")
                        response_parts.append("Would you like more details on any of these?")

                        response = "\n".join(response_parts)

                        # Update context with docks
                        context.set_multi_products(top_docks)
                        session.set_product_context(top_docks, intent.type)
                    else:
                        # No exact match - show all USB-C docks
                        all_usbc_docks = [
                            p for p in all_products_list
                            if p.metadata.get('category') in ('dock', 'hub') and
                            ('USB-C' in p.metadata.get('CONNTYPE', '') or 'usb-c' in (p.content or '').lower())
                        ]
                        if all_usbc_docks:
                            all_usbc_docks = sorted(all_usbc_docks, key=lambda p: p.score, reverse=True)[:3]
                            response = "Here are some USB-C docking stations that might work for you:\n\n"
                            for i, dock in enumerate(all_usbc_docks, 1):
                                sku = dock.product_number
                                specs = _format_dock_specs(dock)
                                response += f"**{i}. {sku}**\n"
                                if specs:
                                    response += f"   - {', '.join(specs)}\n\n"
                            response += "Would you like more details on any of these?"
                            context.set_multi_products(all_usbc_docks)
                            session.set_product_context(all_usbc_docks, intent.type)
                        else:
                            response = (
                                "I'm having trouble finding USB-C docks right now. "
                                "What features are most important for you - monitors, charging, or extra ports?"
                            )

                    # Mark as complete
                    pending.phase = GuidancePhase.COMPLETE
                    save_guidance_to_session(context)
                else:
                    # User said no - that's fine, guidance is complete
                    response = "No problem! Let me know if you have any other questions about your setup."
                    pending.phase = GuidancePhase.COMPLETE
                    save_guidance_to_session(context)

            else:
                # Still in initial phase - shouldn't happen, ask for more info
                response = formatter.format_ambiguous_query()
        else:
            # No pending guidance - this can happen if session state was lost
            # Log this for debugging
            if DEBUG_MODE:
                print("WARNING: SETUP_FOLLOWUP triggered but no pending_guidance in context")
                print(f"  session pending_guidance_data: {st.session_state.get('pending_guidance_data')}")
            response = formatter.format_ambiguous_query()

    elif intent.type.value == "educational_followup":
        # User is answering an educational question (e.g., about daisy-chaining)
        from core.context import PendingQuestionType

        question_type = intent.meta_info.get('question_type') if intent.meta_info else None
        context_data = intent.meta_info.get('context_data', {}) if intent.meta_info else {}

        debug_lines.append(f"🎓 EDUCATIONAL FOLLOWUP: question_type={question_type}")

        if question_type == 'daisy_chain_dp_check':
            # User answered the daisy-chain DisplayPort question
            # Determine if it's affirmative or negative

            affirmative_patterns = [
                r'\byes\b', r'\byeah\b', r'\byep\b', r'\bsure\b',
                r'\bit\s+does\b', r'\bthey\s+do\b', r'\bmine\s+do\b',
                r'\bi\s+do\b', r'\bi\s+have\b', r'\bthey\s+have\b',
                r'\bgot\s+(?:them|it|both)\b',
            ]

            query_lower = query.lower()
            # Use explicit loop to avoid scoping issues with re in generator
            is_affirmative = False
            for pat in affirmative_patterns:
                if re.search(pat, query_lower):
                    is_affirmative = True
                    break

            if is_affirmative:
                # User confirmed they have DP monitors with in/out
                debug_lines.append("🎓 DAISY-CHAIN: User confirmed DP monitors - searching for DP cables")

                response = (
                    "Great! Since your monitors have DisplayPort inputs and outputs, "
                    "you can daisy-chain them using DisplayPort cables.\n\n"
                    "Here are some DisplayPort cables I'd recommend:\n\n"
                )

                # Search for DisplayPort cables
                dp_filters = filter_extractor.extract("DisplayPort to DisplayPort cable")
                dp_results = search_engine.search(dp_filters)

                if dp_results.products:
                    # Rank and get top 3 DP cables
                    ranked = product_ranker.get_top_n(
                        products=dp_results.products,
                        query="DisplayPort daisy chain cable",
                        n=3
                    )

                    for i, rp in enumerate(ranked, 1):
                        prod = rp.product
                        name = prod.metadata.get('name', prod.product_number)
                        sku = prod.product_number
                        length_display = prod.metadata.get('length_display', '')
                        features = prod.metadata.get('features', [])

                        response += f"**{i}. {name}**\n"
                        response += f"   SKU: {sku}\n"
                        if length_display:
                            response += f"   Length: {length_display}\n"
                        if features:
                            response += f"   Features: {', '.join(features[:3])}\n"
                        response += "\n"

                    response += (
                        "**Tip:** Connect your PC to the first monitor's DisplayPort input, "
                        "then run a cable from that monitor's DP output to the next monitor's DP input. "
                        "Each monitor in the chain needs both DP in and DP out ports (except the last one)."
                    )

                    # Update context with products
                    top_products = [rp.product for rp in ranked]
                    context.set_multi_products(top_products)
                    session.set_product_context(top_products, intent.type)
                else:
                    # No DP cables found - search for them
                    dp_cables = [
                        p for p in all_products_list
                        if 'displayport' in (p.content or '').lower() and
                        p.metadata.get('category') == 'cable'
                    ]
                    if dp_cables:
                        dp_cables = sorted(dp_cables, key=lambda p: p.score, reverse=True)[:3]
                        response = (
                            "Great! Since your monitors support DisplayPort daisy-chaining, "
                            "you'll need DisplayPort cables to connect them in series.\n\n"
                            "Here are some options:\n\n"
                        )
                        for i, cable in enumerate(dp_cables, 1):
                            sku = cable.product_number
                            length = cable.metadata.get('length_ft')
                            length_str = f" ({length} ft)" if length else ""
                            response += f"**{i}. {sku}**{length_str}\n"
                        response += "\nWould you like more details on any of these?"
                        context.set_multi_products(dp_cables)
                        session.set_product_context(dp_cables, intent.type)
                    else:
                        response = (
                            "Great! Since your monitors support DisplayPort daisy-chaining, "
                            "you'll need DisplayPort cables to connect them in series.\n\n"
                            "How long do the cables need to be? This will help me find the right ones."
                        )
            else:
                # User said no - their monitors don't have DP in/out
                response = (
                    "No problem! If your monitors don't have DisplayPort outputs, "
                    "daisy-chaining won't work for them. Here are your alternatives:\n\n"
                    "**Option 1: Multiple cables** - Run a separate cable from your "
                    "computer to each monitor.\n\n"
                    "**Option 2: Docking station** - Use a USB-C or Thunderbolt dock "
                    "with multiple video outputs.\n\n"
                    "Would you like help finding individual cables or a docking station for your monitors?"
                )

            # Clear the pending question
            context.clear_pending_question()
            save_pending_question_to_session(context)
        else:
            # Unknown question type
            response = formatter.format_ambiguous_query()
            context.clear_pending_question()
            save_pending_question_to_session(context)

    elif intent.type.value in ["new_search", "constraint_update"]:
        # CONTEXT CLEANUP: Clear abandoned guidance when user starts a new search
        # This prevents old guidance state from polluting unrelated searches
        if context.has_pending_guidance():
            pending = context.pending_guidance
            # Only clear if guidance is NOT complete (user abandoned mid-flow)
            if pending.phase != GuidancePhase.COMPLETE:
                if DEBUG_MODE:
                    debug_lines.append(f"🧹 CLEARING ABANDONED GUIDANCE: was {pending.setup_type}, phase={pending.phase.name}")
                context.clear_guidance()
                save_guidance_to_session(context)

        # Also clear any stale pending feature search offers
        if context.has_pending_feature_search():
            if DEBUG_MODE:
                debug_lines.append(f"🧹 CLEARING STALE FEATURE SEARCH OFFER: {context.pending_feature_search}")
            context.clear_pending_feature_search()

        # Also clear any stale pending questions (educational followups)
        if context.has_pending_question():
            if DEBUG_MODE:
                debug_lines.append(f"🧹 CLEARING STALE PENDING QUESTION: {context.pending_question.question_type}")
            context.clear_pending_question()
            save_pending_question_to_session(context)

        # Clear comparison context since we're getting new products
        # Old comparison indices would point to wrong products after new search
        if context.has_comparison_context():
            if DEBUG_MODE:
                debug_lines.append(f"🧹 CLEARING STALE COMPARISON CONTEXT: indices={context.last_comparison_indices}")
            context.clear_comparison_context()

        # PRIORITY-BASED FILTER EXTRACTION:
        # 1. Extract base filters from query (with timing)
        filter_start = time.perf_counter()
        filters = filter_extractor.extract(query)
        filter_time_ms = (time.perf_counter() - filter_start) * 1000

        # Log filter extraction
        log_filters(
            session_id=session_id,
            query=query,
            filters={
                "connector_from": filters.connector_from,
                "connector_to": filters.connector_to,
                "length": filters.length,
                "length_unit": filters.length_unit,
                "category": filters.product_category,
                "features": filters.features,
                "port_count": filters.port_count,
                "color": filters.color,
            },
            extraction_time_ms=filter_time_ms,
        )

        # 2. Check for explicit technical requirements (HIGHEST PRIORITY)
        tech_reqs = query_analyzer.analyze(query)

        # 3. Override filters with explicit requirements
        # BUT NOT for docks/hubs - they use host port, not cable connectors
        is_dock_or_hub = filters.product_category and filters.product_category.lower() in ('dock', 'docks', 'hub', 'hubs', 'docking station', 'docking stations')
        connector_req = query_analyzer.get_connector_requirement(query)
        if connector_req and not is_dock_or_hub:
            # User explicitly mentioned connector - override device inference
            if connector_req.priority == 1:  # Explicit mention
                # Check if they mentioned both connectors or just one
                # BUT skip for multiport adapters - they have one input but multiple outputs
                is_multiport = 'multiport' in query.lower() or 'multi-port' in query.lower() or 'multi port' in query.lower()
                if not filters.connector_to and connector_req.value and not is_multiport:
                    # They only mentioned one connector - assume it's the target
                    filters.connector_to = connector_req.value
                    
                    if DEBUG_MODE:
                        print(f"\nDEBUG - QueryAnalyzer Override:")
                        print(f"  Found explicit requirement: {connector_req.value}")
                        print(f"  Reason: {connector_req.reason}")
                        print(f"  Priority: {connector_req.priority} (explicit)")
        
        # DEBUG: Print extracted filters
        if DEBUG_MODE:
            print(f"\nDEBUG - Final filters after analysis:")
            print(f"  connector_from: {filters.connector_from}")
            print(f"  connector_to: {filters.connector_to}")
            print(f"  length: {filters.length}")
            print(f"  category: {filters.product_category}")
            print(f"  features: {filters.features}")
            if tech_reqs:
                print(f"\nDEBUG - Technical requirements detected:")
                for req in tech_reqs:
                    print(f"    - {req.value}: {req.reason} (Priority {req.priority})")
        
        # Check pre-search rules
        pre_flight = domain_rules.apply_pre_search_rules(query)

        if pre_flight.should_block:
            response = formatter.format_blocked_request(
                pre_flight.block_reason, pre_flight.suggestions
            )

            # Track if this response asks a question that needs follow-up
            if pre_flight.asks_question and pre_flight.question_type:
                from core.context import PendingQuestionType
                try:
                    question_type = PendingQuestionType(pre_flight.question_type)
                    context.set_pending_question(
                        question_type=question_type,
                        context_data={'original_query': query}
                    )
                    save_pending_question_to_session(context)
                except ValueError:
                    pass  # Unknown question type, don't track
        else:
            # Initialize fallback_note (will be set if fallback search is used)
            fallback_note = None

            # Search with timing
            search_start = time.perf_counter()
            results = search_engine.search(filters)
            search_time_ms = (time.perf_counter() - search_start) * 1000

            # Log search results
            log_search(
                session_id=session_id,
                filters={
                    "connector_from": filters.connector_from,
                    "connector_to": filters.connector_to,
                    "category": filters.product_category,
                },
                products_found=len(results.products),
                tier=results.tier,
                search_time_ms=search_time_ms,
                dropped_filters=[df.filter_name for df in results.dropped_filters] if results.dropped_filters else None,
            )

            if not results.products:
                # FALLBACK: Try progressively simpler searches before giving up
                from core.context import SearchFilters as FallbackFilters
                fallback_results = None
                fallback_note = None

                if DEBUG_MODE:
                    debug_lines.append("🔄 FALLBACK: Primary search returned 0 results, trying simpler searches")

                # DOCK-SPECIFIC FALLBACK: Handle dock searches differently from cables
                is_dock_search = filters.product_category and filters.product_category.lower() in ('docks', 'dock', 'hubs', 'hub')
                if is_dock_search:
                    if DEBUG_MODE:
                        debug_lines.append("🔌 DOCK FALLBACK: Searching for docks with relaxed filters")

                    # Get all docks/hubs
                    all_products_list = st.session_state.get('_all_products', [])
                    all_docks = [p for p in all_products_list if p.metadata.get('category') in ('dock', 'hub')]

                    # Filter out power adapters - they're not docking stations
                    # Power adapters may be categorized as 'dock' but should be excluded
                    all_docks = [d for d in all_docks if not (
                        'power-adapter' in d.product_number.lower() or
                        'power adapter' in (d.metadata.get('name', '') or '').lower() or
                        d.metadata.get('sub_category', '').lower() == 'power adapters'
                    )]

                    # Filter by dock type (Thunderbolt, USB-C, etc.)
                    dock_type = None
                    if 'Thunderbolt' in (filters.features or []):
                        dock_type = 'Thunderbolt'
                        # Filter to Thunderbolt docks only
                        all_docks = [d for d in all_docks if 'thunderbolt' in d.content.lower() or
                                     'tb3' in d.product_number.lower() or 'tb4' in d.product_number.lower() or
                                     'THUNDERBOLT' in str(d.metadata.get('features', [])).upper()]
                    elif filters.connector_from and 'usb-c' in filters.connector_from.lower():
                        dock_type = 'USB-C'

                    if DEBUG_MODE:
                        debug_lines.append(f"🔌 DOCK FALLBACK: dock_type={dock_type}, found {len(all_docks)} docks")

                    if all_docks:
                        # Parse user requirements from query
                        query_lower = query.lower()
                        req_monitors = None
                        req_ethernet = 'ethernet' in query_lower or 'gigabit' in query_lower or 'rj-45' in query_lower
                        req_power = None

                        # Parse monitor count
                        monitor_match = re.search(r'(\d+)\s*(?:x\s*)?(?:4k\s*)?monitors?|dual\s*(?:4k\s*)?monitors?|triple\s*monitors?', query_lower)
                        if monitor_match:
                            if 'dual' in query_lower:
                                req_monitors = 2
                            elif 'triple' in query_lower:
                                req_monitors = 3
                            elif monitor_match.group(1):
                                req_monitors = int(monitor_match.group(1))

                        # Parse power requirement (e.g., "85W", "at least 85W")
                        power_match = re.search(r'(\d+)\s*w(?:att)?s?\s*charg', query_lower)
                        if power_match:
                            req_power = int(power_match.group(1))

                        # Also detect general charging/power delivery requests without specific wattage
                        # e.g., "dock for charging", "charging dock", "power delivery dock"
                        req_power_delivery = bool(re.search(
                            r'\b(?:charging?|charge|power\s*delivery|pd|laptop\s*charg)',
                            query_lower
                        ))

                        if DEBUG_MODE:
                            debug_lines.append(f"🔌 DOCK REQUIREMENTS: monitors={req_monitors}, ethernet={req_ethernet}, power={req_power}W, power_delivery={req_power_delivery}")

                        # Score docks by how many requirements they meet
                        scored_docks = []
                        for dock in all_docks:
                            meta = dock.metadata
                            score = 0
                            met = []
                            unmet = []

                            # Check monitor count
                            num_displays = meta.get('DOCKNUMDISPLAYS')
                            if num_displays:
                                num_displays = int(float(num_displays))
                                if req_monitors and num_displays >= req_monitors:
                                    score += 2
                                    met.append(f"{num_displays} monitors")
                                elif req_monitors:
                                    unmet.append(f"only {num_displays} monitors")
                                else:
                                    met.append(f"{num_displays} monitors")

                            # Check ethernet
                            network_speed = meta.get('network_speed')
                            has_ethernet = network_speed or 'RJ-45' in meta.get('CONNTYPE', '')
                            if req_ethernet:
                                if has_ethernet:
                                    score += 1
                                    met.append("Gigabit Ethernet")
                                else:
                                    unmet.append("no Ethernet")
                            elif has_ethernet:
                                met.append("Gigabit Ethernet")

                            # Check power delivery
                            pd_wattage = meta.get('power_delivery') or meta.get('hub_power_delivery')
                            pd_num = 0
                            if pd_wattage:
                                pd_match = re.search(r'(\d+)', str(pd_wattage))
                                if pd_match:
                                    pd_num = int(pd_match.group(1))

                            if req_power:
                                # User specified a wattage (e.g., "85W charging")
                                if pd_num >= req_power:
                                    score += 2
                                    met.append(f"{pd_num}W charging")
                                elif pd_num > 0:
                                    unmet.append(f"only {pd_num}W (not {req_power}W)")
                                else:
                                    unmet.append("no charging")
                            elif req_power_delivery:
                                # User wants charging but no specific wattage (e.g., "dock for charging")
                                if pd_num > 0:
                                    score += 2
                                    met.append(f"{pd_num}W charging")
                                else:
                                    unmet.append("no charging")
                            elif pd_num > 0:
                                met.append(f"{pd_num}W charging")

                            # Check 4K using unified method
                            if dock.supports_4k():
                                met.append("4K")

                            scored_docks.append((dock, score, met, unmet))

                        # Sort by score (descending)
                        scored_docks.sort(key=lambda x: x[1], reverse=True)

                        if scored_docks:
                            # Use top docks as results
                            from core.context import SearchResult
                            fallback_results = SearchResult(
                                products=[d[0] for d in scored_docks[:5]],
                                filters_used={'category': 'dock', 'dock_type': dock_type},
                                tier=4,
                                dropped_filters=[]
                            )

                            # Build note about what's available/unavailable
                            # Check if ANY dock meets all requirements
                            best_dock = scored_docks[0]
                            unmet_reqs = best_dock[3]

                            if unmet_reqs:
                                fallback_note = f"**Note:** No {dock_type or ''} docks found with {', '.join(unmet_reqs).replace('only ', '')}. Showing closest matches."
                            else:
                                # All requirements met by top dock
                                fallback_note = None

                            if DEBUG_MODE:
                                debug_lines.append(f"🔌 DOCK FALLBACK: Best dock unmet={unmet_reqs}")

                # Cable fallback (only if not a dock search or dock fallback failed)
                if not fallback_results:
                    # Fallback 1: Keep just connectors (drop length, features, keywords)
                    if filters.connector_from or filters.connector_to:
                        simple_filters = FallbackFilters()
                        simple_filters.connector_from = filters.connector_from
                        simple_filters.connector_to = filters.connector_to
                        simple_filters.product_category = filters.product_category

                        if DEBUG_MODE:
                            debug_lines.append(f"🔄 FALLBACK 1: connector_from={simple_filters.connector_from}, connector_to={simple_filters.connector_to}")

                        fallback_results = search_engine.search(simple_filters)

                        if fallback_results.products:
                            # Analyze what user requested vs what products actually have
                            # to build an accurate note about what's missing
                            missing_features = []
                            available_lengths = set()
                            available_features = set()

                            if DEBUG_MODE:
                                debug_lines.append(f"🔍 FALLBACK ANALYSIS: filters.features = {filters.features}")

                            for product in fallback_results.products:
                                # Collect lengths
                                length = product.metadata.get('length')
                                if length:
                                    available_lengths.add(length)
                                # Collect features
                                product_features = product.metadata.get('features', [])
                                if isinstance(product_features, list):
                                    available_features.update(f.upper() for f in product_features)

                            # Check which requested features are actually available
                            if DEBUG_MODE:
                                debug_lines.append(f"🔍 FALLBACK ANALYSIS: available_features = {available_features}")

                            if filters.features:
                                for feature in filters.features:
                                    feature_upper = feature.upper()
                                    if feature_upper not in available_features:
                                        missing_features.append(feature)
                                if DEBUG_MODE:
                                    debug_lines.append(f"🔍 FALLBACK ANALYSIS: missing_features = {missing_features}")

                            # Check if requested length is available
                            length_available = False
                            if filters.length and available_lengths:
                                # Check if any product has the requested length (within tolerance)
                                for avail_len in available_lengths:
                                    if abs(avail_len - filters.length) < 0.5:  # 0.5ft tolerance
                                        length_available = True
                                        break
                            elif not filters.length:
                                length_available = True  # User didn't request specific length

                            # Build accurate note about what's NOT available
                            if missing_features:
                                feature_str = ', '.join(missing_features)
                                if len(available_lengths) > 0:
                                    min_len = min(available_lengths)
                                    max_len = max(available_lengths)
                                    if min_len == max_len:
                                        len_range = f"{min_len:.0f}ft"
                                    else:
                                        len_range = f"{min_len:.0f}ft-{max_len:.0f}ft"
                                    fallback_note = f"**Note:** None have {feature_str} support. Available lengths: {len_range}."
                                else:
                                    fallback_note = f"**Note:** None have {feature_str} support."
                            elif not length_available and filters.length:
                                if available_lengths:
                                    sorted_lengths = sorted(available_lengths)
                                    len_options = ', '.join(f"{l:.0f}ft" for l in sorted_lengths[:4])
                                    fallback_note = f"**Note:** No {int(filters.length)}ft options. Available: {len_options}."
                                else:
                                    fallback_note = f"**Note:** No {int(filters.length)}ft options available."
                            else:
                                # Both length and features are available - shouldn't hit fallback
                                fallback_note = None

                            if DEBUG_MODE:
                                debug_lines.append(f"🔍 FALLBACK ANALYSIS: fallback_note = {fallback_note}")

                    # Fallback 2: Try just one connector at a time
                    if not fallback_results or not fallback_results.products:
                        for connector in [filters.connector_from, filters.connector_to]:
                            if connector:
                                simple_filters = FallbackFilters()
                                simple_filters.connector_from = connector
                                simple_filters.product_category = filters.product_category or 'Cables'

                                if DEBUG_MODE:
                                    debug_lines.append(f"🔄 FALLBACK 2: Single connector search: {connector}")

                                fallback_results = search_engine.search(simple_filters)

                                if fallback_results.products:
                                    conn_from = filters.connector_from or "?"
                                    conn_to = filters.connector_to or "?"
                                    fallback_note = f"**Note:** No {conn_from} to {conn_to} products found. Here are {connector} products instead:"
                                    break

                if fallback_results and fallback_results.products:
                    # Use fallback results
                    results = fallback_results
                else:
                    # All fallbacks failed - show generic message
                    suggestions = [
                        "Try searching by connector type (USB-C, HDMI, DisplayPort)",
                        "Try a different cable length",
                        "Use simpler terms like 'USB-C cable' or 'HDMI adapter'"
                    ]
                    response = formatter.format_no_results(query, suggestions)
                    fallback_note = None

            if results.products:
                # NEW: Rank products by relevance and pick top 3
                filters_dict = {
                    'length': filters.length,
                    'features': filters.features,
                    'length_preference': filters.length_preference,
                }

                ranked_products = product_ranker.get_top_n(
                    products=results.products,
                    query=query,
                    n=3,  # Top 3 instead of 5
                    extracted_filters=filters_dict
                )
                
                # Post-search rules
                post_search = domain_rules.apply_post_search_rules(
                    results, query, filters
                )
                
                # Build intro text (if domain rules added context OR fallback was used)
                intro_text = None
                if fallback_note:
                    intro_text = f"{fallback_note}\n"
                elif post_search.modified_response:
                    intro_text = f"{post_search.modified_response}\n"
                
                # NEW: Generate friendly, conversational response
                # Pass dropped_filters and original_filters for color mismatch notice
                original_filters_dict = {
                    'color': filters.color,
                    'length': filters.length,
                    'connector_from': filters.connector_from,
                    'connector_to': filters.connector_to,
                }
                response = response_builder.build_response(
                    ranked_products=ranked_products,
                    query=query,
                    intro_text=intro_text,
                    dropped_filters=results.dropped_filters,
                    original_filters=original_filters_dict,
                )
                
                # Update context with top 3 products
                top_products = [rp.product for rp in ranked_products]
                context.set_multi_products(top_products)
                session.set_product_context(top_products, intent.type)
                
    elif intent.type.value in ["multi_followup", "single_followup"]:
        # Handle refinement requests (e.g., "I need 3 foot cables")
        if intent.meta_info and intent.meta_info.get('refinement') and context.current_products:
            # User wants the same product types but with different length
            debug_lines.append("📏 REFINEMENT: User refining products with new constraint")

            # Extract the new length from the query
            new_filters = filter_extractor.extract(query)
            requested_length = new_filters.length
            length_unit = new_filters.length_unit or 'ft'

            if requested_length:
                debug_lines.append(f"📏 REFINEMENT: Requested length = {requested_length} {length_unit}")

                # Re-search for each UNIQUE product type with new length
                # First, collect unique connector pairs from context products
                unique_connector_pairs = {}  # (source, target) -> first product with this pair
                for original_product in context.current_products:
                    connectors = original_product.metadata.get('connectors', [])
                    source_port = None
                    target_port = None

                    if connectors and len(connectors) >= 2:
                        conn_from = connectors[0].lower()
                        conn_to = connectors[1].lower()

                        port_mapping = {
                            'hdmi': 'HDMI', 'usb-c': 'USB-C', 'usb c': 'USB-C',
                            'type-c': 'USB-C', 'displayport': 'DisplayPort', 'dp': 'DisplayPort',
                            'vga': 'VGA', 'dvi': 'DVI'
                        }
                        for key, value in port_mapping.items():
                            if key in conn_from:
                                source_port = value
                                break
                        for key, value in port_mapping.items():
                            if key in conn_to:
                                target_port = value
                                break

                    if source_port:
                        pair_key = (source_port, target_port or source_port)
                        if pair_key not in unique_connector_pairs:
                            unique_connector_pairs[pair_key] = original_product

                response_parts = []
                response_parts.append(f"Here are {int(requested_length)}{length_unit} options for your setup:")
                response_parts.append("")

                all_refined_products = []
                for (source_port, target_port), original_product in unique_connector_pairs.items():
                    # Build filters for this cable type with new length
                    refined_filters = filter_extractor.extract("")
                    refined_filters.connector_from = source_port
                    # Always set connector_to to ensure we get the right cable type
                    # (e.g., HDMI-to-HDMI, not HDMI-to-DVI)
                    refined_filters.connector_to = target_port or source_port
                    refined_filters.length = requested_length
                    refined_filters.length_unit = length_unit

                    # Search
                    refined_results = search_engine.search(refined_filters)

                    # Format cable type
                    if source_port == target_port or not target_port:
                        cable_desc = f"{source_port} cable"
                    else:
                        cable_desc = f"{source_port} to {target_port}"

                    response_parts.append(f"**{cable_desc}:**")

                    if refined_results.products:
                        # Get best cable at requested length
                        best_product = get_best_cable(
                            refined_results.products,
                            source_port,
                            target_port or source_port,
                            preferred_length_ft=requested_length if length_unit == 'ft' else requested_length * 3.28
                        )

                        if best_product:
                            all_refined_products.append(best_product)
                            name = best_product.metadata.get('name', best_product.product_number)
                            length_display = best_product.metadata.get('length_display', '')
                            sku = best_product.product_number

                            response_parts.append(f"Recommended: **{name}** ({sku})")
                            if length_display:
                                response_parts.append(f"Length: {length_display}")
                        else:
                            response_parts.append(f"_No {int(requested_length)}{length_unit} cable found - closest options may differ_")
                    else:
                        response_parts.append(f"_No {int(requested_length)}{length_unit} version available_")
                    response_parts.append("")

                response_parts.append("Would you like more details on any of these?")
                response = "\n".join(response_parts)

                # Update context
                if all_refined_products:
                    context.set_multi_products(all_refined_products)
                    session.set_product_context(all_refined_products, intent.type)
            else:
                # No length specified - check for feature/requirement-based refinement
                # This handles cases like "I need it for my MacBook Pro. I want to connect 2 monitors with 4K, charge my laptop, ethernet"
                requirement_keywords = _extract_requirement_keywords(query)

                if requirement_keywords and context.current_products:
                    debug_lines.append(f"🎯 REQUIREMENT REFINEMENT: Keywords = {requirement_keywords}")

                    # Score each product based on how many requirements it matches
                    scored_products = []
                    for product in context.current_products:
                        score = _score_product_by_requirements(product, requirement_keywords)
                        scored_products.append((product, score))

                    # Sort by score (descending) then by original relevance
                    scored_products.sort(key=lambda x: (x[1], x[0].score), reverse=True)

                    # Filter to products that match at least some requirements
                    matching_products = [(p, s) for p, s in scored_products if s > 0]

                    if matching_products:
                        # Build response showing best matches
                        response_parts = []

                        # Summarize what we matched
                        matched_reqs = [k for k in requirement_keywords if any(
                            k.lower() in p.content.lower() or k.lower() in str(p.metadata.get('features', [])).lower()
                            for p, _ in matching_products[:3]
                        )]
                        if matched_reqs:
                            response_parts.append(f"Based on your requirements ({', '.join(matched_reqs[:4])}), here are the best options:")
                        else:
                            response_parts.append("Here are the best matches for your requirements:")
                        response_parts.append("")

                        # Show top 3-5 matches
                        shown = 0
                        for product, score in matching_products[:5]:
                            if shown >= 3 and score < matching_products[0][1]:
                                break  # Stop if score drops significantly

                            name = product.metadata.get('name', product.product_number)
                            category = product.metadata.get('category', '')

                            response_parts.append(f"**{name}** ({product.product_number})")

                            # For docks, show comprehensive specs
                            if category in ('dock', 'hub'):
                                specs = _format_dock_specs(product)
                                if specs:
                                    for spec in specs:
                                        response_parts.append(f"   - {spec}")
                            else:
                                # For other products, show features
                                features = product.metadata.get('features', [])
                                if features:
                                    response_parts.append(f"   Features: {', '.join(features[:5])}")

                                # Show which requirements this product matches
                                matches = [k for k in requirement_keywords
                                          if k.lower() in product.content.lower()
                                          or k.lower() in str(features).lower()]
                                if matches:
                                    response_parts.append(f"   Matches: {', '.join(matches)}")
                            response_parts.append("")
                            shown += 1

                        response_parts.append("Would you like more details on any of these?")
                        response = "\n".join(response_parts)

                        # Update context with filtered products
                        filtered_products = [p for p, _ in matching_products[:5]]
                        context.set_multi_products(filtered_products)
                        session.set_product_context(filtered_products, intent.type)
                    else:
                        # No products match the requirements - suggest searching again
                        response = (
                            f"None of the current products match all your requirements. "
                            f"Would you like me to search for products with {', '.join(requirement_keywords[:3])}?"
                        )
                else:
                    # No length and no requirements detected - ask for clarification
                    response = "I couldn't determine what you're looking for. Could you specify what features or length you need?"

        # Handle follow-up questions about products in context
        elif context.current_products:
            # Priority 1: Try follow-up handler (specific product, comparisons, superlatives)
            followup_handler = get_followup_handler()
            followup_answer = followup_handler.handle_followup(
                query=query,
                products=context.current_products,
                intent=intent,
                context=context  # Pass context for comparison tracking
            )

            if followup_answer:
                # Follow-up handler answered the question
                response = followup_answer
            else:
                # Priority 2: Try technical question handler
                tech_handler = TechnicalQuestionHandler()
                tech_answer = tech_handler.answer_technical_question(
                    query=query,
                    products=context.current_products
                )

                if tech_answer:
                    # User asked a technical question - answer it directly
                    response = tech_answer
                else:
                    # Fallback: List the products with comprehensive specs
                    response = "Here are the full specs:\n\n"
                    for i, prod in enumerate(context.current_products, 1):
                        category = prod.metadata.get('category', '').lower()
                        name = prod.metadata.get('name', prod.product_number)
                        sku = prod.product_number

                        response += f"**{i}. {name}** ({sku})\n\n"

                        # Use comprehensive dock specs for docks/hubs
                        if category in ('dock', 'hub', 'docking_station'):
                            dock_specs = _format_dock_specs(prod)
                            if dock_specs:
                                for spec in dock_specs:
                                    response += f"- {spec}\n"
                            else:
                                # Fallback if no specs extracted
                                features = prod.metadata.get('features', [])
                                if features:
                                    response += f"- Features: {', '.join(features)}\n"
                        else:
                            # Standard format for cables/adapters
                            length_display = prod.metadata.get('length_display')
                            if length_display:
                                response += f"- Length: {length_display}\n"

                            connectors = prod.metadata.get('connectors')
                            if connectors and len(connectors) >= 2:
                                conn_str = f"{connectors[0]} → {connectors[1]}"
                                response += f"- Connectors: {conn_str}\n"

                            features = prod.metadata.get('features', [])
                            features_str = ', '.join(features) if features else 'Standard features'
                            response += f"- Features: {features_str}\n"

                        response += "\n"

                    response += "Would you like me to compare any of these, or do you have specific questions?"
        else:
            response = "I can help you find StarTech.com products. What are you looking for?"
            
    else:
        # Ambiguous query - Check domain rules FIRST (educational questions like daisy-chain)
        pre_flight = domain_rules.apply_pre_search_rules(query)

        if pre_flight.should_block:
            response = formatter.format_blocked_request(
                pre_flight.block_reason, pre_flight.suggestions
            )

            # Track if this response asks a question that needs follow-up
            if pre_flight.asks_question and pre_flight.question_type:
                from core.context import PendingQuestionType
                try:
                    question_type = PendingQuestionType(pre_flight.question_type)
                    context.set_pending_question(
                        question_type=question_type,
                        context_data={'original_query': query}
                    )
                    save_pending_question_to_session(context)
                except ValueError:
                    pass  # Unknown question type, don't track
        else:
            # No domain rule triggered - Check for explicit requirements
            tech_reqs = query_analyzer.analyze(query)
            connector_req = query_analyzer.get_connector_requirement(query)

            # Priority 1: Explicit technical requirements (e.g., "DP 1.4", "daisy-chain")
            if connector_req and connector_req.priority <= 2:
                # User has explicit requirements - skip device inference
                filters = filter_extractor.extract(query)

                # Apply explicit connector requirement
                # BUT skip for:
                # - Multiport adapters (one input, multiple outputs)
                # - Docks/hubs (USB-C is the HOST port, not a cable connector pair)
                is_multiport = 'multiport' in query.lower() or 'multi-port' in query.lower() or 'multi port' in query.lower()
                is_dock_or_hub = filters.product_category and filters.product_category.lower() in (
                    'dock', 'docks', 'hub', 'hubs', 'docking station', 'docking stations'
                )
                if not filters.connector_to and not is_multiport and not is_dock_or_hub:
                    filters.connector_to = connector_req.value

                # Search with explicit requirements
                results = search_engine.search(filters)

                if results.products:
                    # Rank products
                    filters_dict = {
                        'length': filters.length,
                        'features': filters.features,
                        'length_preference': filters.length_preference,
                    }

                    ranked_products = product_ranker.get_top_n(
                        products=results.products,
                        query=query,
                        n=3,
                        extracted_filters=filters_dict
                    )

                    # Build intro explaining the technical requirement
                    intro_text = f"{connector_req.reason}\n"

                    # Generate response
                    response = response_builder.build_response(
                        ranked_products=ranked_products,
                        query=query,
                        intro_text=intro_text
                    )

                    # Update context
                    top_products = [rp.product for rp in ranked_products]
                    context.set_multi_products(top_products)
                    session.set_product_context(top_products, intent.type)
                else:
                    # No products found with explicit requirements
                    response = formatter.format_ambiguous_query()

            # Priority 2: Device inference (if no explicit requirements)
            else:
                inference_result = device_inference.infer(query)

                if inference_result.should_suggest:
                    # Confidence is high enough - show products with explanation

                    # Build filters from inference
                    filters = filter_extractor.extract(query)

                    # Override with inferred connectors if we found them
                    if inference_result.connector_from:
                        filters.connector_from = inference_result.connector_from
                    if inference_result.connector_to:
                        filters.connector_to = inference_result.connector_to

                    # NEW: Apply preferred category if detected (multi-monitor, dock preference)
                    if inference_result.preferred_category:
                        filters.product_category = inference_result.preferred_category

                    # Search with inferred filters
                    results = search_engine.search(filters)

                    # NEW: If user wanted a dock but none found, try adapters instead
                    if not results.products and inference_result.preferred_category == "dock":
                        # No docks available - fallback to adapters
                        filters.product_category = "adapter"
                        results = search_engine.search(filters)

                    if results.products:
                        # NEW: Rank products by relevance and pick top 3
                        filters_dict = {
                            'length': filters.length,
                            'features': filters.features,
                        }

                        ranked_products = product_ranker.get_top_n(
                            products=results.products,
                            query=query,
                            n=3,  # Top 3
                            extracted_filters=filters_dict
                        )

                        # Build intro with device inference explanation (no emoji)
                        intro_text = f"{inference_result.reasoning}\n"

                        # NEW: If they wanted a dock but we're showing adapters, explain
                        if inference_result.preferred_category == "dock" and filters.product_category == "adapter":
                            monitor_count = inference_result.multi_monitor_count
                            intro_text += (
                                f"\n**Note:** A USB-C docking station would be ideal for {monitor_count} monitors "
                                f"(one cable solution), but I don't have any in my catalog. Here are USB-C to HDMI "
                                f"adapters instead - you'd need one per monitor plus HDMI cables.\n"
                            )

                        # Add confirming questions if any
                        if inference_result.suggestions:
                            intro_text += "\n**Quick check:**\n"
                            for suggestion in inference_result.suggestions:
                                intro_text += f"• {suggestion}\n"
                            intro_text += "\n"

                        # NEW: Generate friendly, conversational response
                        response = response_builder.build_response(
                            ranked_products=ranked_products,
                            query=query,
                            intro_text=intro_text
                        )

                        # Update context with top 3 products
                        top_products = [rp.product for rp in ranked_products]
                        context.set_multi_products(top_products)
                        session.set_product_context(top_products, intent.type)
                    else:
                        # Inference didn't find products - ask for clarification
                        response = formatter.format_ambiguous_query()
                else:
                    # Confidence too low - ask for clarification
                    response = formatter.format_ambiguous_query()
    
    # Calculate total response time
    total_response_time_ms = (time.perf_counter() - request_start) * 1000
    products_count = len(context.current_products) if context.current_products else 0

    # Log final response with structured logging
    log_response(
        session_id=session_id,
        intent=intent.type.value,
        products_found=products_count,
        response_time_ms=total_response_time_ms,
    )

    # Log products shown (if any)
    if context.current_products:
        log_products_shown(
            session_id=session_id,
            products=context.current_products,
            query=query,
        )

    # Log conversation (existing CSV logger)
    logger.log_conversation(
        session_id=session.session_id,
        user_message=query,
        bot_response=response,
        intent_type=intent.type.value,
        products_shown=products_count
    )

    # DEBUG: Prepend debug info to response so it's ALWAYS visible
    if DEBUG_MODE:
        debug_header = "**🔍 DEBUG OUTPUT:**\n```\n" + "\n".join(debug_lines) + "\n```\n\n---\n\n"
        response = debug_header + response

    return response, intent.type.value


def main():
    st.title("🤖 ST-Bot - StarTech.com Product Assistant")
    st.markdown("*Powered by your 4,178+ product catalog*")
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        excel_path = st.text_input(
            "📁 Excel File Path",
            value="Main Data AI Bot.xlsx",
            help="Path to your Excel file"
        )
        
        st.markdown("---")
    
    # Load products
    products, stats, error = load_products(excel_path)

    if error:
        st.error(f"❌ {error}")
        st.info("💡 Please ensure Main_Data_AI_Bot.xlsx is in the same folder as this app")
        st.stop()

    if not products:
        st.warning("⚠️ No products loaded. Check your Excel file.")
        st.stop()

    # Store products in session state for dock search access
    st.session_state._all_products = products
    
    # Sidebar - Product Statistics
    with st.sidebar:
        st.header("📦 Product Catalog")
        st.metric("Total Products", stats['total'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("With Length", stats['with_length'])
        with col2:
            st.metric("With Connectors", stats['with_connectors'])
        
        with st.expander("📊 Categories"):
            for cat, count in sorted(stats['by_category'].items(), key=lambda x: x[1], reverse=True)[:10]:
                st.write(f"• **{cat.title()}:** {count}")
        
        # Show sample products
        with st.expander("🔍 Sample Products"):
            for i, prod in enumerate(products[:5], 1):
                st.write(f"**{i}. {prod.product_number}**")
                st.write(f"   {prod.metadata.get('name', '')[:50]}...")
        
        st.markdown("---")
    
    # Initialize session state
    if "session" not in st.session_state:
        st.session_state.session = SessionState()
        st.session_state.context = ConversationContext()
        st.session_state.messages = []
        # Store pending guidance as dict (Streamlit can't serialize Enum dataclasses properly)
        st.session_state.pending_guidance_data = None
    
    # Get components
    components = get_components(products)
    
    # Sidebar - Session Stats
    with st.sidebar:
        st.header("📊 Session Stats")
        st.write(f"**Session ID:** `{st.session_state.session.session_id[:16]}...`")
        st.write(f"**Messages:** {st.session_state.session.get_message_count()}")

        if st.session_state.context.current_products:
            st.write(f"**Products in Context:** {len(st.session_state.context.current_products)}")

        # DEBUG: Show guidance state at page render time
        if DEBUG_MODE:
            guidance_data = st.session_state.get('pending_guidance_data')
            st.markdown("---")
            st.markdown("**🔍 Debug - Guidance State (at render):**")
            if guidance_data:
                st.write(f"Phase: `{guidance_data.get('phase')}`")
                st.write(f"Ports: `{guidance_data.get('computer_ports')}`")
                st.write(f"Inputs: `{guidance_data.get('monitor_inputs')}`")
            else:
                st.write("*No pending guidance*")
            st.markdown("---")
        
        if st.button("🔄 New Session"):
            # Clear ALL session-specific state for a fresh start
            st.session_state.session = SessionState()
            st.session_state.context = ConversationContext()
            st.session_state.messages = []
            st.session_state.pending_guidance_data = None  # Clear guidance state
            st.session_state.pending_question_data = None  # Clear educational questions
            # Note: _all_products is the product database - don't clear it
            st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    prompt = st.chat_input("What StarTech.com product are you looking for?")
    
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Searching 4,000+ products..."):
                response, intent_type = process_query(
                    prompt,
                    st.session_state.context,
                    st.session_state.session,
                    components
                )
                st.markdown(response)
                
                # Debug info (only shown if DEBUG_MODE is True)
                if DEBUG_MODE:
                    with st.expander("🔍 Debug Info"):
                        st.write(f"**Intent Detected:** {intent_type}")
                        st.write(f"**Total Products Available:** {len(products)}")
                        
                        if st.session_state.context.current_products:
                            st.write(f"**Products in Context:** {len(st.session_state.context.current_products)}")
                            for i, p in enumerate(st.session_state.context.current_products[:3], 1):
                                st.write(f"{i}. {p.product_number} - {p.metadata.get('name', '')[:40]}...")
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Save to session
        st.session_state.session.add_message("user", prompt)
        st.session_state.session.add_message("assistant", response)


if __name__ == "__main__":
    main()