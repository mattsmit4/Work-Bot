"""
Response Builder Module - Conversational Product Responses

Generates natural, conversational responses like a helpful sales rep.
Follows CLAUDE.md guidelines: concise, no filler, varied structure.

Design:
- Conversational: Reads like a person talking, not a template
- Concise: No repeated phrases, no decorative elements
- Factual: Only mentions features that exist in product data
- Varied: Each product description has different structure
"""

from typing import List, Optional, Dict
from dataclasses import dataclass
import re


# Cable types and their commonly missing/refinable attributes
# Order matters: compound types (usb-c-to-hdmi) must come before simple types (hdmi, usb-c)
CABLE_REFINEMENT_CONFIG = [
    ('usb-c-to-hdmi', {
        'keywords': [('usb-c', 'hdmi'), ('usb c', 'hdmi'), ('type-c', 'hdmi'), ('usb-c to hdmi',), ('type-c to hdmi',)],
        'refinements': {
            'length': "Need a specific length?",
            'features': "Looking for 4K support?",
        }
    }),
    ('usb-c-to-displayport', {
        'keywords': [('usb-c', 'displayport'), ('usb c', 'displayport'), ('type-c', 'displayport')],
        'refinements': {
            'length': "Need a specific length?",
            'features': "Looking for 4K or 8K support?",
        }
    }),
    ('displayport-to-hdmi', {
        'keywords': [('displayport', 'hdmi'), ('display port', 'hdmi'), (' dp ', 'hdmi')],
        'refinements': {
            'length': "Need a specific length?",
            'features': "Looking for 4K support?",
        }
    }),
    ('hdmi', {
        'keywords': ['hdmi'],
        'refinements': {
            'length': "Need a different length?",
            'features': "Looking for specific features like 4K support or Audio Return Channel?",
        }
    }),
    ('usb-c', {
        'keywords': ['usb-c', 'usb c', 'type-c', 'type c'],
        'refinements': {
            'length': "Need a specific length?",
            'features': "Looking for 4K support or Power Delivery?",
        }
    }),
    ('displayport', {
        'keywords': ['displayport', 'display port', ' dp '],
        'refinements': {
            'length': "Need a different length?",
            'features': "Looking for a specific DisplayPort version or 4K/8K support?",
        }
    }),
    ('ethernet', {
        'keywords': ['ethernet', 'cat5', 'cat6', 'cat7', 'network cable', 'lan cable'],
        'refinements': {
            'length': "Need a different length?",
            'features': "Looking for a specific category (Cat5e, Cat6, Cat6a)?",
        }
    }),
]


@dataclass
class ProductExplanation:
    """
    Formatted explanation for a single product.
    """
    product_number: str
    name: str
    formatted_text: str


class ResponseBuilder:
    """
    Builds conversational product responses.

    Avoids templated/robotic responses by varying sentence structure
    and avoiding repeated phrases.
    """

    def __init__(self):
        """Initialize response builder."""
        pass

    def _detect_cable_type(self, query: str) -> Optional[str]:
        """
        Detect what type of cable the user is searching for.

        Returns the cable type key from CABLE_REFINEMENT_CONFIG, or None.
        Order matters - compound types (usb-c-to-hdmi) are checked before simple types.
        """
        query_lower = ' ' + query.lower() + ' '  # Pad for word boundary matching

        # Check in order (compound types first due to list ordering)
        for cable_type, config in CABLE_REFINEMENT_CONFIG:
            keywords = config['keywords']
            for kw in keywords:
                if isinstance(kw, tuple):
                    # Compound keyword - all parts must be present
                    if all(part in query_lower for part in kw):
                        return cable_type
                else:
                    if kw in query_lower:
                        return cable_type

        return None

    def _check_what_is_missing(self, query: str) -> Dict[str, bool]:
        """
        Check what details are missing from the user's cable query.

        Returns dict with 'length' and 'features' keys indicating if they're missing.
        """
        query_lower = query.lower()
        missing = {'length': True, 'features': True}

        # Check if length is specified
        length_patterns = [
            r'\d+\s*(?:ft|foot|feet|m|meter|meters)',
            r'\b(?:short|long|extra long)\b',
            r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:ft|foot|feet)\b',
        ]
        for pattern in length_patterns:
            if re.search(pattern, query_lower):
                missing['length'] = False
                break

        # Check if features are specified
        feature_patterns = [
            r'\b4k\b', r'\b8k\b', r'\b1080p\b', r'\bhdr\b',
            r'\bpower delivery\b', r'\bpd\b',
            r'\bhigh[\s-]?speed\b', r'\bpremium\b', r'\bcertified\b',
            r'\bcat\s*\d', r'\bgigabit\b', r'\b10g\b',
            r'\barc\b', r'\baudio return\b',
            r'\bactive\b', r'\bpassive\b',
        ]
        for pattern in feature_patterns:
            if re.search(pattern, query_lower):
                missing['features'] = False
                break

        return missing

    def _is_specific_but_incomplete(self, query: str) -> bool:
        """
        Check if query is specific enough to show products but could benefit
        from refinement offers.

        Specific = mentions a cable type
        Incomplete = missing length OR features
        """
        cable_type = self._detect_cable_type(query)
        if not cable_type:
            return False

        missing = self._check_what_is_missing(query)
        return missing['length'] or missing['features']

    def _generate_refinement_offer(self, query: str) -> Optional[str]:
        """
        Generate a friendly refinement offer based on what's missing.

        Returns None if query is already fully specified or not a cable search.
        """
        query_lower = query.lower()

        # Skip for dock/hub searches - they don't have length refinements
        if any(kw in query_lower for kw in ['dock', 'docking', 'hub']):
            return "Would you like specs on any of these?"

        # Skip for multiport adapter searches - they don't have cable features
        if any(kw in query_lower for kw in ['multiport', 'multi-port', 'multi port',
                                             'usb-c adapter', 'usb c adapter', 'travel adapter']):
            return "Would you like specs on any of these?"

        # Skip for PCIe card searches - they don't have length/cable features
        if any(kw in query_lower for kw in ['pcie', 'pci express', 'pci-e', 'expansion card',
                                             'network card', 'usb card', 'serial card']):
            return "Would you like specs on any of these?"

        cable_type = self._detect_cable_type(query)
        if not cable_type:
            return None

        missing = self._check_what_is_missing(query)

        # If nothing is missing, no refinement needed
        if not missing['length'] and not missing['features']:
            return None

        # Find the config for this cable type
        config = None
        for ct, cfg in CABLE_REFINEMENT_CONFIG:
            if ct == cable_type:
                config = cfg
                break

        if not config:
            return None

        refinements = config.get('refinements', {})

        offers = []
        if missing['length'] and 'length' in refinements:
            offers.append(refinements['length'])
        if missing['features'] and 'features' in refinements:
            offers.append(refinements['features'])

        if not offers:
            return None

        # Combine into a friendly closing
        if len(offers) == 1:
            return f"{offers[0]} Just let me know!"
        else:
            return f"{offers[0]} {offers[1]} Just let me know!"

    def _simplify_connector_name(self, connector: str) -> str:
        """Simplify connector names for natural language."""
        import re
        connector_clean = re.sub(r'^\d+\s*x\s*', '', connector, flags=re.IGNORECASE).strip()
        connector_lower = connector_clean.lower()

        if 'usb-c' in connector_lower or 'type-c' in connector_lower:
            return "USB-C"
        elif 'hdmi' in connector_lower:
            return "HDMI"
        elif 'displayport' in connector_lower or 'display port' in connector_lower:
            return "DisplayPort"
        elif 'thunderbolt' in connector_lower:
            return "Thunderbolt"
        elif 'usb' in connector_lower and ('type-a' in connector_lower or 'type a' in connector_lower):
            return "USB-A"
        elif 'dvi' in connector_lower:
            return "DVI"
        elif 'vga' in connector_lower:
            return "VGA"
        else:
            cleaned = re.sub(r'\([^)]*\)', '', connector_clean).strip()
            return cleaned.rstrip('.,') if cleaned else connector_clean

    def _get_feature_bullets(self, product) -> List[str]:
        """
        Get key features as bullet points.
        Only includes features that actually exist in product data.

        Priority order for display (most important first):
        1. Resolution (4K, 8K, 1080p)
        2. HDR
        3. Refresh rate
        4. Thunderbolt
        5. Power Delivery
        6. Shielded (only if nothing else to show)
        """
        features = product.metadata.get('features', [])
        metadata = product.metadata
        bullets = []

        # Check for high-value features first (resolution-related)
        # Use unified Product methods for consistent resolution detection
        has_resolution_feature = False

        # Resolution from features - use unified methods for consistency
        if product.supports_4k():
            # Check for more specific resolution info
            max_res = metadata.get('max_resolution', '')
            if max_res and '@' in str(max_res):
                # Has refresh rate info like "3840x2160@60Hz"
                bullets.append(f"4K ({max_res})")
            else:
                bullets.append("4K")
            has_resolution_feature = True
        elif product.supports_resolution('8k'):
            bullets.append("8K")
            has_resolution_feature = True
        elif product.supports_resolution('1440p'):
            bullets.append("1440p")
            has_resolution_feature = True
        elif product.supports_resolution('1080p'):
            bullets.append("1080p Full HD")
            has_resolution_feature = True

        # HDR
        if 'HDR' in features:
            bullets.append("HDR")

        # Thunderbolt
        if any('thunderbolt' in str(f).lower() for f in features):
            bullets.append("Thunderbolt")

        # Power Delivery
        if 'Power Delivery' in features:
            bullets.append("Power Delivery")

        # Shielded - show for network cables, otherwise only if nothing else to show
        if 'Shielded' in features:
            # For network cables, shielded is more important (show it)
            # For video cables, it's less important (only show if nothing else)
            network_rating = metadata.get('network_rating')
            if network_rating or not bullets:
                bullets.append("Shielded")

        # PoE support - important for network cables
        if 'PoE' in features:
            bullets.append("PoE")

        # Outdoor rating - check jacket_type and content
        jacket_type = metadata.get('jacket_type', '').lower()
        content = product.content.lower() if product.content else ''
        if 'outdoor' in jacket_type or 'outdoor' in content or jacket_type == 'pe':
            bullets.append("Outdoor-rated")

        return bullets

    def _is_hub_product(self, product) -> bool:
        """Check if this is a USB hub product."""
        category = product.metadata.get('category', '').lower()
        return category == 'hub'

    def _format_hub_product_line(self, product, position: int) -> str:
        """
        Format a USB hub product with hub-specific information.

        Format: 1. SKU - 4-Port USB 3.0 Hub (Powered)
                   Features: Charging ports, individual switches

        Shows port count, USB version, and power status instead of length/connectors.
        """
        sku = product.product_number
        hub_ports = product.metadata.get('hub_ports')
        hub_usb_version = product.metadata.get('hub_usb_version')
        hub_powered = product.metadata.get('hub_powered')
        hub_power_delivery = product.metadata.get('hub_power_delivery')
        sub_category = product.metadata.get('sub_category', '')

        # Build the main description
        parts = [f"**{position}. {sku}**"]

        # Add port count and USB version
        desc_parts = []
        if hub_ports:
            desc_parts.append(f"{hub_ports}-Port")
        if hub_usb_version:
            desc_parts.append(hub_usb_version)
        else:
            desc_parts.append("USB Hub")

        if desc_parts:
            parts.append(f" - {' '.join(desc_parts)}")

        # Add power status
        power_info = []
        if hub_powered is True:
            power_info.append("Powered")
        elif hub_powered is False:
            power_info.append("Bus-powered")

        if hub_power_delivery:
            power_info.append(f"PD {hub_power_delivery}")

        # Add subcategory info (e.g., "Industrial USB Hubs")
        if 'industrial' in sub_category.lower():
            power_info.append("Industrial")

        if power_info:
            parts.append(f" ({', '.join(power_info)})")

        return "".join(parts)

    def _is_dock_product(self, product) -> bool:
        """Check if this is a docking station product."""
        category = product.metadata.get('category', '').lower()
        return category == 'dock'

    def _parse_usb_ports(self, conntype: str) -> dict:
        """
        Parse USB port counts from CONNTYPE field.

        Returns dict with 'usb_c', 'usb_a', and 'total_usb' counts.
        Example CONNTYPE: "1 x USB 3.2 Type-C, 6 x USB 3.2 Type-A"
        """
        import re
        result = {'usb_c': 0, 'usb_a': 0, 'total_usb': 0}

        if not conntype:
            return result

        # Find patterns like "2 x USB Type-A" or "1 x USB 3.2 Type-C"
        # Match: count x USB [version] Type-A/Type-C
        usb_patterns = re.findall(
            r'(\d+)\s*x\s*USB[^,]*?Type-([AC])',
            conntype, re.IGNORECASE
        )

        for count_str, port_type in usb_patterns:
            count = int(count_str)
            if port_type.upper() == 'C':
                result['usb_c'] += count
            elif port_type.upper() == 'A':
                result['usb_a'] += count

        result['total_usb'] = result['usb_c'] + result['usb_a']
        return result

    def _detect_query_interest(self, query: str) -> set:
        """
        Detect what specs the user cares about from their query.

        Returns set of interest categories: 'usb_ports', 'usb_c', 'monitors', 'charging', 'ethernet', etc.
        """
        import re
        query_lower = query.lower()
        interests = set()

        # USB port interests
        if re.search(r'\busb[\s-]?c\b.*\bports?\b|\bports?\b.*\busb[\s-]?c\b', query_lower):
            interests.add('usb_c')
            interests.add('usb_ports')
        if re.search(r'\busb[\s-]?a\b.*\bports?\b|\bports?\b.*\busb[\s-]?a\b', query_lower):
            interests.add('usb_a')
            interests.add('usb_ports')
        if re.search(r'\busb\b.*\bports?\b|\bports?\b', query_lower):
            interests.add('usb_ports')
        if re.search(r'\b(?:lots?|bunch|many|multiple|several)\s+(?:of\s+)?(?:usb|ports)', query_lower):
            interests.add('usb_ports')

        # Other interests
        if re.search(r'\bmonitors?\b|\bdisplays?\b', query_lower):
            interests.add('monitors')
        if re.search(r'\bcharg(?:ing|e)\b|\bpower\s*delivery\b|\bpd\b', query_lower):
            interests.add('charging')
        if re.search(r'\bethernet\b|\bnetwork\b|\blan\b', query_lower):
            interests.add('ethernet')
        if re.search(r'\b4k\b|\b8k\b|\bresolution\b', query_lower):
            interests.add('resolution')

        return interests

    def _format_dock_product_line(self, product, position: int, query: str = None) -> str:
        """
        Format a docking station with dock-specific information.

        Format: 1. SKU - Thunderbolt 4 Dock
                   2 monitors, 96W charging, Gigabit Ethernet

        Shows monitor count, power delivery, ethernet, and connection type.
        When user asks about specific features (USB ports, etc.), those are shown first.
        """
        import re
        sku = product.product_number
        meta = product.metadata
        features = meta.get('features', [])
        conntype = meta.get('CONNTYPE', '')

        # Detect what the user cares about
        interests = self._detect_query_interest(query) if query else set()

        # Build the main line
        parts = [f"**{position}. {sku}**"]

        # Determine dock type from features or content
        dock_type = None
        if 'Thunderbolt' in features or 'thunderbolt' in product.content.lower():
            if 'tb4' in sku.lower() or 'thunderbolt 4' in product.content.lower():
                dock_type = "Thunderbolt 4 Dock"
            else:
                dock_type = "Thunderbolt 3 Dock"
        elif 'USB-C' in str(conntype) or 'usb-c' in product.content.lower():
            dock_type = "USB-C Dock"
        else:
            dock_type = "Docking Station"

        parts.append(f" - {dock_type}")

        # Build specs list - prioritize what user asked about
        specs = []
        usb_ports = self._parse_usb_ports(conntype)

        # If user asked about USB/ports, show that FIRST
        if 'usb_ports' in interests or 'usb_c' in interests:
            port_info = []
            if usb_ports['usb_c'] > 0:
                port_info.append(f"{usb_ports['usb_c']} USB-C")
            if usb_ports['usb_a'] > 0:
                port_info.append(f"{usb_ports['usb_a']} USB-A")
            if port_info:
                specs.append(' + '.join(port_info) + " ports")
            elif usb_ports['total_usb'] > 0:
                specs.append(f"{usb_ports['total_usb']} USB ports")

        # Monitor count
        num_displays = meta.get('DOCKNUMDISPLAYS')
        if num_displays:
            num_displays = int(float(num_displays))
            if num_displays == 1:
                specs.append("1 monitor")
            else:
                specs.append(f"{num_displays} monitors")

        # Power Delivery wattage
        pd_wattage = meta.get('power_delivery') or meta.get('hub_power_delivery')
        if pd_wattage:
            pd_match = re.search(r'(\d+)', str(pd_wattage))
            if pd_match:
                specs.append(f"{pd_match.group(1)}W charging")

        # Ethernet
        network_speed = meta.get('network_speed')
        has_ethernet = network_speed or 'RJ-45' in conntype
        if has_ethernet:
            if network_speed and ('Gbps' in network_speed or '1000' in network_speed or 'Gigabit' in network_speed):
                specs.append("Gigabit Ethernet")
            else:
                specs.append("Ethernet")

        # If user didn't ask about USB but dock has good USB ports, mention it at the end
        if 'usb_ports' not in interests and usb_ports['total_usb'] >= 4:
            specs.append(f"{usb_ports['total_usb']} USB ports")

        # Add specs as second line
        line = "".join(parts)
        if specs:
            line += f"\n   {', '.join(specs)}"

        return line

    def _is_ethernet_switch(self, product) -> bool:
        """Check if this is an ethernet/network switch product."""
        category = product.metadata.get('category', '').lower()
        return category == 'ethernet_switch'

    def _is_kvm_switch(self, product) -> bool:
        """Check if this is a KVM switch or video switch product."""
        category = product.metadata.get('category', '').lower()
        # KVM switches, video switches, and generic switches with KVM in name
        if category in ('kvm_switch', 'video_switch'):
            return True
        if category == 'switch':
            # Check if it's actually a KVM/video switch by name or content
            name = product.metadata.get('name', '').lower()
            content = (product.content or '').lower()
            if 'kvm' in name or 'kvm' in content or 'video switch' in name:
                return True
        return False

    def _is_mount_product(self, product) -> bool:
        """Check if this is a monitor/display mount product."""
        category = product.metadata.get('category', '').lower()
        return category == 'mount'

    def _is_video_splitter(self, product) -> bool:
        """Check if this is a video splitter product."""
        category = product.metadata.get('category', '').lower()
        return category == 'video_splitter'

    def _format_ethernet_switch_line(self, product, position: int) -> str:
        """
        Format an ethernet switch product with networking-specific information.

        Format: 1. SKU - 8-Port Gigabit Ethernet Switch (PoE, Managed)

        Shows port count, speed, and key features instead of length/connectors.
        """
        sku = product.product_number
        hub_ports = product.metadata.get('hub_ports')
        network_speed = product.metadata.get('network_speed', '')
        features = product.metadata.get('features', [])
        sub_category = product.metadata.get('sub_category', '')
        name = product.metadata.get('name', '')

        # Build the main description
        parts = [f"**{position}. {sku}**"]

        # Determine the product type from SKU (more reliable than name which has subcategory)
        sku_lower = sku.lower()
        if 'isolator' in sku_lower:
            product_type = "Network Isolator"
        elif 'extender' in sku_lower:
            product_type = "Ethernet Extender"
        elif sku_lower.startswith('ies') or 'switch' in sku_lower:
            # IES prefix = Industrial Ethernet Switch
            product_type = "Ethernet Switch"
        else:
            product_type = "Ethernet Switch"

        # Add port count
        desc_parts = []
        if hub_ports and hub_ports > 1:
            desc_parts.append(f"{hub_ports}-Port")

        # Add speed info (avoid redundant "Ethernet" for Fast Ethernet + Ethernet Switch)
        if network_speed:
            if '1000' in network_speed or 'gigabit' in network_speed.lower():
                desc_parts.append("Gigabit")
            elif '10000' in network_speed or '10g' in network_speed.lower():
                desc_parts.append("10GbE")
            elif '100' in network_speed:
                # "Fast Ethernet" already includes "Ethernet", so just say "10/100Mbps"
                desc_parts.append("10/100Mbps")

        desc_parts.append(product_type)

        if desc_parts:
            parts.append(f" - {' '.join(desc_parts)}")

        # Add key features in parentheses
        feature_parts = []
        if features:
            if 'PoE' in features:
                feature_parts.append("PoE")
            if 'Managed' in features:
                feature_parts.append("Managed")

        # Check for industrial
        if 'industrial' in sub_category.lower() or 'industrial' in sku_lower:
            feature_parts.append("Industrial")

        if feature_parts:
            parts.append(f" ({', '.join(feature_parts)})")

        return "".join(parts)

    def _format_kvm_switch_line(self, product, position: int) -> str:
        """
        Format a KVM switch or video switch product with switch-specific information.

        Format: 1. SV231DPUA - 2-Port DisplayPort KVM (4K, Audio, USB)

        Shows port count, video type, resolution, and key features.
        Also handles KVM cables that are miscategorized as KVM switches.
        """
        sku = product.product_number
        meta = product.metadata
        category = meta.get('category', '').lower()
        kvm_ports = meta.get('kvm_ports') or meta.get('hub_ports')
        kvm_video = meta.get('kvm_video_type', '')
        kvm_interface = meta.get('kvm_interface', '')
        kvm_audio = meta.get('kvm_audio', False)
        max_resolution = meta.get('max_resolution', '')
        features = meta.get('features', [])
        sub_category = meta.get('sub_category', '')
        name = meta.get('name', '').lower()
        content = (product.content or '').lower()

        # Check if this is actually a KVM cable (miscategorized as switch)
        # KVM cables have no port count and content/ZCONTENTITEM mentions "cable"
        # ZCONTENTITEM field often contains "1 x KVM Cable" for cable products
        zcontentitem = str(meta.get('ZCONTENTITEM', '')).lower()
        is_kvm_cable = (not kvm_ports and ('cable' in content or 'cable' in zcontentitem)) or 'kvm cable' in content or 'kvm cable' in zcontentitem

        if is_kvm_cable:
            # Format as a KVM cable instead
            return self._format_kvm_cable_line(product, position)

        # Determine if this is a KVM or video switch
        is_kvm = 'kvm' in category or 'kvm' in name or 'kvm' in content
        switch_type = "KVM" if is_kvm else "Switch"

        # Build the main description
        parts = [f"**{position}. {sku}**"]

        # Determine port count and video type
        desc_parts = []

        if kvm_ports:
            desc_parts.append(f"{int(kvm_ports)}-Port")

        # Determine video type from various sources
        video_type = None
        if kvm_video:
            video_type = kvm_video
        elif kvm_interface:
            video_type = kvm_interface
        else:
            # Try to infer from SKU or content
            sku_lower = sku.lower()
            if 'dp' in sku_lower or 'displayport' in content:
                video_type = 'DisplayPort'
            elif 'hdmi' in sku_lower or 'hdmi' in content:
                video_type = 'HDMI'
            elif 'dvi' in sku_lower or 'dvi' in content:
                video_type = 'DVI'
            elif 'vga' in sku_lower or 'vga' in content:
                video_type = 'VGA'

        if video_type:
            desc_parts.append(video_type)

        desc_parts.append(switch_type)

        if desc_parts:
            parts.append(f" - {' '.join(desc_parts)}")

        # Add key features in parentheses
        feature_parts = []

        # Resolution support
        if max_resolution:
            res_str = str(max_resolution).lower()
            if '4k' in res_str or '2160' in res_str or '3840' in res_str:
                feature_parts.append("4K")
            elif '8k' in res_str or '4320' in res_str or '7680' in res_str:
                feature_parts.append("8K")
            elif '1440' in res_str or '2560' in res_str:
                feature_parts.append("1440p")

        # Check for 4K/8K in features if not found in resolution
        if not feature_parts:
            if any('4k' in str(f).lower() for f in features):
                feature_parts.append("4K")
            elif any('8k' in str(f).lower() for f in features):
                feature_parts.append("8K")
            elif '4k' in content or '2160' in content:
                feature_parts.append("4K")
            elif '8k' in content:
                feature_parts.append("8K")

        # Audio support
        if kvm_audio:
            feature_parts.append("Audio")

        # USB switching (common KVM feature)
        if 'usb' in content or any('usb' in str(f).lower() for f in features):
            feature_parts.append("USB")

        # Check for specific KVM types
        if 'desktop' in sub_category.lower():
            feature_parts.append("Desktop")
        elif 'rack' in sub_category.lower() or 'enterprise' in sub_category.lower():
            feature_parts.append("Rack-mount")

        if feature_parts:
            parts.append(f" ({', '.join(feature_parts)})")

        return "".join(parts)

    def _format_kvm_cable_line(self, product, position: int) -> str:
        """
        Format a KVM cable product (often miscategorized as KVM switch).

        Format: 1. RKCONSUV10 - 10ft VGA/USB KVM Cable

        Shows cable length and connector types.
        """
        import re
        sku = product.product_number
        meta = product.metadata
        content = (product.content or '').lower()
        length_ft = meta.get('length_ft')
        connectors = meta.get('connectors', [])

        # Build the main description
        parts = [f"**{position}. {sku}**"]
        desc_parts = []

        # Add length - try multiple sources
        if length_ft:
            desc_parts.append(f"{int(length_ft)}ft")
        else:
            # Try product_length (stored in mm for KVM cables)
            product_length_mm = meta.get('product_length')
            if product_length_mm:
                try:
                    length_ft_calc = round(float(product_length_mm) / 304.8)
                    if length_ft_calc > 0:
                        desc_parts.append(f"{int(length_ft_calc)}ft")
                except (ValueError, TypeError):
                    pass

            # Fallback: try to extract length from SKU (e.g., RKCONSUV10 -> 10ft)
            if not desc_parts:
                length_match = re.search(r'(\d+)$', sku)
                if length_match:
                    desc_parts.append(f"{length_match.group(1)}ft")

        # Determine connector types from content, interfaces, and SKU
        interface_a = str(meta.get('interface_a', '')).lower()
        interface_b = str(meta.get('interface_b', '')).lower()
        sku_lower = sku.lower()

        video_type = None
        # Check multiple sources for video type
        check_str = f"{content} {interface_a} {interface_b} {sku_lower}"
        if 'vga' in check_str:
            video_type = 'VGA'
        elif 'dvi' in check_str:
            video_type = 'DVI'
        elif 'hdmi' in check_str:
            video_type = 'HDMI'
        elif 'displayport' in check_str or 'dp' in sku_lower:
            video_type = 'DisplayPort'

        # Check for USB
        has_usb = 'usb' in check_str

        if video_type and has_usb:
            desc_parts.append(f"{video_type}/USB")
        elif video_type:
            desc_parts.append(video_type)

        desc_parts.append("KVM Cable")

        if desc_parts:
            parts.append(f" - {' '.join(desc_parts)}")

        return "".join(parts)

    def _format_mount_product_line(self, product, position: int) -> str:
        """
        Format a monitor/display mount product with mount-specific information.

        Format: 1. 1MP1ACG-MONITOR-ARM - Single Monitor Desk Mount (17-30")
                   ^ VESA 75x75, 100x100, Steel/Aluminum

        Shows mount type, display size range, VESA patterns, and material.
        """
        sku = product.product_number
        sub_category = product.metadata.get('sub_category', '')
        mount_type = product.metadata.get('mount_type', '')
        display_range = product.metadata.get('mount_display_range', '')
        vesa = product.metadata.get('mount_vesa', '')
        num_displays = product.metadata.get('mount_num_displays')
        material = product.metadata.get('mount_material', '')
        curved_support = product.metadata.get('mount_curved_support', False)

        # Build main description
        parts = [f"**{position}. {sku}**"]

        # Determine mount type description
        desc_parts = []

        # Number of displays
        if num_displays and num_displays > 1:
            if num_displays == 2:
                desc_parts.append("Dual Monitor")
            elif num_displays == 3:
                desc_parts.append("Triple Monitor")
            elif num_displays == 4:
                desc_parts.append("Quad Monitor")
            else:
                desc_parts.append(f"{num_displays}-Monitor")
        elif num_displays == 1:
            desc_parts.append("Single Monitor")

        # Mount type from data or infer from subcategory/SKU
        sku_lower = sku.lower()
        sub_lower = sub_category.lower()

        if mount_type:
            mount_lower = mount_type.lower()
            if 'wall' in mount_lower:
                desc_parts.append("Wall Mount")
            elif 'desk' in mount_lower or 'clamp' in mount_lower or 'grommet' in mount_lower:
                desc_parts.append("Desk Mount")
            elif 'pole' in mount_lower:
                desc_parts.append("Pole Mount")
            elif 'stand' in mount_lower or 'desktop' in mount_lower:
                desc_parts.append("Monitor Stand")
            elif 'cart' in mount_lower or 'mobile' in mount_lower:
                desc_parts.append("Mobile Cart")
            else:
                desc_parts.append("Mount")
        elif 'wall' in sub_lower or 'wall' in sku_lower:
            desc_parts.append("Wall Mount")
        elif 'desk' in sub_lower or 'arm' in sku_lower:
            desc_parts.append("Desk Mount")
        elif 'cart' in sub_lower or 'cart' in sku_lower:
            desc_parts.append("Mobile Cart")
        elif 'stand' in sub_lower or 'stand' in sku_lower:
            desc_parts.append("Monitor Stand")
        elif 'shelf' in sub_lower:
            desc_parts.append("Wall Shelf")
        else:
            desc_parts.append("Display Mount")

        if desc_parts:
            parts.append(f" - {' '.join(desc_parts)}")

        # Add display size range in parentheses
        if display_range:
            parts.append(f" ({display_range})")

        main_line = "".join(parts)

        # Build secondary line with specs
        specs = []
        if vesa:
            specs.append(f"VESA {vesa}")
        if material:
            # Simplify material description
            mat_lower = material.lower()
            if 'steel' in mat_lower and 'aluminum' in mat_lower:
                specs.append("Steel/Aluminum")
            elif 'steel' in mat_lower:
                specs.append("Steel")
            elif 'aluminum' in mat_lower:
                specs.append("Aluminum")
        if curved_support:
            specs.append("Curved TV compatible")

        if specs:
            spec_line = f"\n   ^ {', '.join(specs)}"
            return main_line + spec_line

        return main_line

    def _format_video_splitter_line(self, product, position: int) -> str:
        """
        Format a video splitter product with splitter-specific information.

        Format: 1. ST124HD4K - 4-Port HDMI Splitter (1 In → 4 Out, 4K@30Hz, Audio)

        Shows output count, video type, input/output config, resolution, and audio.
        """
        sku = product.product_number
        meta = product.metadata

        # Get splitter-specific fields
        num_ports = meta.get('NUMBERPORTS') or meta.get('number_ports') or meta.get('hub_ports')
        av_input = meta.get('AVINPUT', '') or meta.get('av_input', '')
        av_output = meta.get('AVOUTPUT', '') or meta.get('av_output', '')
        kvm_audio = meta.get('KVMAUDIO', '') or meta.get('kvm_audio', '')
        interface_a = meta.get('INTERFACEA', '') or meta.get('interface_a', '')
        interface_b = meta.get('INTERFACEB', '') or meta.get('interface_b', '')
        max_resolution = meta.get('max_resolution', '') or meta.get('MAXRESOLUTION', '')
        name = meta.get('name', '')

        # Build main description
        parts = [f"**{position}. {sku}**"]
        desc_parts = []

        # Number of outputs
        if num_ports:
            try:
                port_count = int(num_ports)
                if port_count > 1:
                    desc_parts.append(f"{port_count}-Port")
            except (ValueError, TypeError):
                pass

        # Video type - determine from interfaces, input/output, or name
        video_type = ''
        name_lower = name.lower()
        sku_lower = sku.lower()

        if av_input:
            av_in_lower = str(av_input).lower()
            if 'hdmi' in av_in_lower:
                video_type = 'HDMI'
            elif 'displayport' in av_in_lower or 'dp' in av_in_lower:
                video_type = 'DisplayPort'
            elif 'dvi' in av_in_lower:
                video_type = 'DVI'
            elif 'vga' in av_in_lower:
                video_type = 'VGA'

        if not video_type and interface_a:
            int_a_lower = str(interface_a).lower()
            if 'hdmi' in int_a_lower:
                video_type = 'HDMI'
            elif 'displayport' in int_a_lower or 'dp' in int_a_lower:
                video_type = 'DisplayPort'
            elif 'dvi' in int_a_lower:
                video_type = 'DVI'
            elif 'vga' in int_a_lower:
                video_type = 'VGA'

        # Fallback to SKU/name detection
        if not video_type:
            if 'hdmi' in name_lower or 'hdmi' in sku_lower:
                video_type = 'HDMI'
            elif 'displayport' in name_lower or 'dp' in sku_lower:
                video_type = 'DisplayPort'
            elif 'dvi' in name_lower or 'dvi' in sku_lower:
                video_type = 'DVI'
            elif 'vga' in name_lower or 'vga' in sku_lower:
                video_type = 'VGA'

        if video_type:
            desc_parts.append(f"{video_type} Splitter")
        else:
            desc_parts.append("Video Splitter")

        if desc_parts:
            parts.append(f" - {' '.join(desc_parts)}")

        # Add features in parentheses
        feature_parts = []

        # Input/output configuration (1 In → 4 Out)
        if num_ports:
            try:
                out_count = int(num_ports)
                feature_parts.append(f"1 In → {out_count} Out")
            except (ValueError, TypeError):
                pass

        # Resolution
        if max_resolution:
            res_str = str(max_resolution)
            # Simplify resolution display
            if '4k' in res_str.lower() or '4096' in res_str or '3840' in res_str:
                if '60' in res_str:
                    feature_parts.append("4K@60Hz")
                elif '30' in res_str:
                    feature_parts.append("4K@30Hz")
                else:
                    feature_parts.append("4K")
            elif '1080' in res_str:
                feature_parts.append("1080p")
            elif '1920' in res_str:
                feature_parts.append("1080p")
            else:
                # Show raw resolution if not standard
                feature_parts.append(res_str)

        # Audio support
        if kvm_audio:
            audio_str = str(kvm_audio).lower()
            if audio_str in ('yes', 'true', '1') or 'audio' in audio_str:
                feature_parts.append("Audio")

        if feature_parts:
            parts.append(f" ({', '.join(feature_parts)})")

        return "".join(parts)

    def _is_fiber_cable(self, product) -> bool:
        """Check if this is a fiber optic cable product."""
        category = product.metadata.get('category', '').lower()
        return category == 'fiber_cable'

    def _is_storage_enclosure(self, product) -> bool:
        """Check if this is a storage/drive enclosure product."""
        category = product.metadata.get('category', '').lower()
        return category == 'storage_enclosure'

    def _is_privacy_screen(self, product) -> bool:
        """Check if this is a privacy screen/filter product."""
        category = product.metadata.get('category', '').lower()
        return category == 'privacy_screen'

    def _is_multiport_adapter(self, product) -> bool:
        """Check if this is a multiport adapter (USB-C hub, travel dock, etc.)."""
        meta = product.metadata
        category = meta.get('category', '').lower()
        if category == 'multiport_adapter':
            return True

        # Check sub_category
        sub_cat = meta.get('sub_category', '').lower()
        if 'multiport' in sub_cat:
            return True

        # Check SKU for multiport pattern
        sku = (product.product_number or '').upper()
        if 'MULTIPORT' in sku:
            return True

        # Check product name for multiport indicators
        name = meta.get('name', '').lower()
        content = (product.content or '').lower()
        if 'multiport' in name or 'multi-port' in name:
            return True
        if 'multiport' in content or 'multi-port' in content:
            return True

        return False

    def _is_pcie_card(self, product) -> bool:
        """Check if this is a PCIe/computer expansion card."""
        meta = product.metadata
        category = meta.get('category', '').lower()
        if category == 'computer_card':
            return True

        # Check sub_category for card types
        sub_cat = meta.get('sub_category', '').lower()
        if 'card' in sub_cat and ('network' in sub_cat or 'usb' in sub_cat or 'serial' in sub_cat):
            return True

        # Check BUSTYPE - if it contains PCI, it's a card
        bus_type = meta.get('BUSTYPE', '') or meta.get('BusType', '')
        if bus_type and 'pci' in str(bus_type).lower():
            return True

        # Check if sub_category mentions components/accessories (often cards)
        if 'component' in sub_cat and 'accessor' in sub_cat:
            # Further check - does it have PCI-related content?
            content = (product.content or '').lower()
            name = meta.get('name', '').lower()
            if 'pci' in content or 'pci' in name or 'slot' in content or 'slot' in name:
                return True

        # Check product name/content for PCIe indicators
        name = meta.get('name', '').lower()
        content = (product.content or '').lower()
        if 'pcie' in name or 'pci express' in name or 'pci-e' in name:
            return True
        if 'expansion card' in content or 'add-on card' in content:
            return True

        return False

    def _format_pcie_card_line(self, product, position: int) -> str:
        """
        Format a PCIe/computer expansion card with card-specific information.

        Format: 1. ST1000SPEX2 - PCIe x1 Network Card
                   4-Port Gigabit Ethernet, Low Profile, Intel I350

        Shows bus type, card profile, port count, and interface type.
        """
        sku = product.product_number
        meta = product.metadata
        sub_category = meta.get('sub_category', '')
        name = meta.get('name', '') or meta.get('Name', '') or ''
        content = product.content or ''

        # Combine text sources for pattern matching
        all_text = f"{name} {sub_category} {content}".lower()

        # Extract card-specific metadata - check multiple field name variants
        bus_type = (meta.get('BUSTYPE', '') or meta.get('BusType', '') or
                    meta.get('bus_type', '') or meta.get('interface_a', '') or
                    meta.get('INTERFACEA', ''))
        card_profile = (meta.get('CARDPROFILE', '') or meta.get('CardProfile', '') or
                        meta.get('card_profile', '') or meta.get('PROFILE', ''))
        interface_a = meta.get('INTERFACEA', '') or meta.get('interface_a', '')
        interface_b = meta.get('INTERFACEB', '') or meta.get('interface_b', '')
        num_ports = meta.get('NUMBERPORTS', '') or meta.get('hub_ports', '') or meta.get('NumberPorts', '')
        chipset = meta.get('CHIPSET', '') or meta.get('CONTROLLER', '') or meta.get('Chipset', '')

        # Try to extract bus type from interface_a or text if not in dedicated field
        bus_display = ''
        if bus_type:
            bus_str = str(bus_type).strip()
            if bus_str.lower() not in ('nan', 'none', ''):
                # Extract PCIe lane info if present
                pcie_match = re.search(r'pci\s*express?\s*(x\d+)?', bus_str, re.IGNORECASE)
                if pcie_match:
                    lane = pcie_match.group(1) or ''
                    bus_display = f"PCIe {lane}".strip() if lane else "PCIe"
                elif 'pci' in bus_str.lower():
                    bus_display = bus_str

        # Fallback: extract bus type from name/content
        if not bus_display:
            pcie_match = re.search(r'pci(?:e|[\s-]*express)?\s*(x\d+)?', all_text)
            if pcie_match:
                lane = pcie_match.group(1) or ''
                bus_display = f"PCIe {lane}".strip() if lane else "PCIe"

        # Build the main line
        parts = [f"**{position}. {sku}**"]

        # Determine card type from sub_category, name, interface, or content
        card_type = ''
        card_type_source = sub_category.lower() + ' ' + name.lower()

        if 'network' in card_type_source or 'ethernet' in all_text or 'gigabit' in all_text:
            card_type = 'Network Card'
        elif 'usb card' in card_type_source or ('usb' in card_type_source and 'card' in card_type_source):
            card_type = 'USB Card'
        elif 'serial' in card_type_source or 'rs-232' in all_text or 'rs232' in all_text:
            card_type = 'Serial Card'
        elif 'sata' in card_type_source or 'storage' in card_type_source or 'raid' in all_text:
            card_type = 'Storage Controller'
        elif 'video' in card_type_source or 'display' in card_type_source or 'graphics' in all_text:
            card_type = 'Video Card'
        elif 'riser' in all_text or 'extender' in all_text or 'adapter' in all_text:
            card_type = 'Slot Adapter'
        elif interface_b:
            # Infer from interface B (output side)
            intf_lower = str(interface_b).lower()
            if 'ethernet' in intf_lower or 'rj45' in intf_lower or 'rj-45' in intf_lower:
                card_type = 'Network Card'
            elif 'usb' in intf_lower:
                card_type = 'USB Card'
            elif 'serial' in intf_lower or 'rs232' in intf_lower or 'rs-232' in intf_lower:
                card_type = 'Serial Card'
            elif 'sata' in intf_lower or 'sas' in intf_lower:
                card_type = 'Storage Controller'
            elif 'pci' in intf_lower:
                card_type = 'Slot Adapter'
            else:
                card_type = 'Expansion Card'
        else:
            card_type = 'Expansion Card'

        # Add bus type and card type
        if bus_display:
            parts.append(f" - {bus_display} {card_type}")
        else:
            parts.append(f" - {card_type}")

        line = "".join(parts)

        # Build feature bullets
        features = []

        # Port count and interface type - check dedicated field first
        port_num = 0
        if num_ports:
            try:
                port_num = int(float(num_ports))
            except (ValueError, TypeError):
                pass

        # Fallback: extract port count from name/content
        if port_num == 0:
            port_match = re.search(r'(\d+)[\s-]*port', all_text)
            if port_match:
                try:
                    port_num = int(port_match.group(1))
                except ValueError:
                    pass

        # Extract network speed from content
        network_speed = ''
        if '10 gigabit' in all_text or '10gbe' in all_text or '10g ethernet' in all_text:
            network_speed = '10 Gigabit'
        elif '2.5 gigabit' in all_text or '2.5gbe' in all_text or '2.5g ethernet' in all_text:
            network_speed = '2.5 Gigabit'
        elif 'gigabit' in all_text or '1gbe' in all_text or '1000base' in all_text:
            network_speed = 'Gigabit'
        elif '10/100' in all_text or 'fast ethernet' in all_text:
            network_speed = '10/100'

        # Build port/interface feature string
        if port_num > 0:
            if network_speed:
                features.append(f"{port_num}-Port {network_speed} Ethernet")
            elif interface_b:
                intf_clean = self._simplify_interface_name(str(interface_b))
                features.append(f"{port_num}-Port {intf_clean}")
            else:
                features.append(f"{port_num}-Port")
        elif network_speed:
            features.append(f"{network_speed} Ethernet")
        elif interface_b:
            intf_clean = self._simplify_interface_name(str(interface_b))
            if intf_clean:
                features.append(intf_clean)

        # Card profile (critical for case compatibility) - check field and content
        if card_profile:
            profile_str = str(card_profile).strip()
            if profile_str.lower() not in ('nan', 'none', ''):
                features.append(profile_str)
        else:
            # Extract profile from content
            if 'low profile' in all_text or 'low-profile' in all_text:
                if 'full height' in all_text or 'standard profile' in all_text:
                    features.append('Low Profile & Full Height')
                else:
                    features.append('Low Profile')
            elif 'full height' in all_text:
                features.append('Full Height')

        # Chipset/controller (useful for compatibility)
        if chipset:
            chipset_str = str(chipset).strip()
            if chipset_str.lower() not in ('nan', 'none', ''):
                features.append(chipset_str)

        # Add standard product features (but avoid duplicates)
        product_features = meta.get('features', [])
        for feat in product_features[:2]:
            feat_lower = feat.lower()
            # Skip if already covered
            if any(feat_lower in f.lower() for f in features):
                continue
            features.append(feat)

        # Add feature line
        if features:
            feature_str = ", ".join(features[:4])  # Limit to 4 features
            line += f"\n   {feature_str}"

        return line

    def _format_multiport_adapter_line(self, product, position: int) -> str:
        """
        Format a multiport adapter with port configuration information.

        Format: 1. DKT30CHPD3 - USB-C Multiport Adapter
                   Ports: HDMI + 2x USB-A + GbE + SD/microSD, PD, 4K

        Shows input type, output ports, power delivery, and video support.
        """
        sku = product.product_number
        meta = product.metadata
        name = meta.get('name', '') or ''
        content = product.content or ''
        sub_category = meta.get('sub_category', '')

        # Combine text sources for pattern matching
        all_text = f"{name} {sub_category} {content}".lower()

        # Extract input connection type from sub_category or content
        input_type = ''
        if 'thunderbolt' in sub_category.lower():
            if '4' in sub_category:
                input_type = 'Thunderbolt 4'
            elif '3' in sub_category:
                input_type = 'Thunderbolt 3'
            else:
                input_type = 'Thunderbolt'
        elif 'usb-c' in sub_category.lower() or 'usb c' in sub_category.lower():
            input_type = 'USB-C'
        elif 'usb-a' in sub_category.lower():
            input_type = 'USB-A'
        # Fallback: check content/name
        if not input_type:
            if 'thunderbolt 4' in all_text:
                input_type = 'Thunderbolt 4'
            elif 'thunderbolt 3' in all_text:
                input_type = 'Thunderbolt 3'
            elif 'thunderbolt' in all_text:
                input_type = 'Thunderbolt'
            elif 'usb-c' in all_text or 'usb type-c' in all_text:
                input_type = 'USB-C'

        # Build the main line
        parts = [f"**{position}. {sku}**"]
        if input_type:
            parts.append(f" - {input_type} Multiport Adapter")
        else:
            parts.append(" - Multiport Adapter")

        line = "".join(parts)

        # Parse EXTERNALPORTS field for port configuration
        ports = self._parse_external_ports(meta.get('EXTERNALPORTS', ''))

        # Build specs list - ports first (key differentiator)
        specs = []

        if ports:
            # Format port list cleanly
            port_items = []
            if ports.get('hdmi'):
                port_items.append(f"{ports['hdmi']}x HDMI" if ports['hdmi'] > 1 else "HDMI")
            if ports.get('displayport'):
                port_items.append(f"{ports['displayport']}x DP" if ports['displayport'] > 1 else "DP")
            if ports.get('vga'):
                port_items.append("VGA")
            if ports.get('usb_a'):
                port_items.append(f"{ports['usb_a']}x USB-A")
            if ports.get('usb_c'):
                port_items.append(f"{ports['usb_c']}x USB-C")
            if ports.get('ethernet'):
                port_items.append("GbE")
            if ports.get('sd'):
                if ports.get('microsd'):
                    port_items.append("SD/microSD")
                else:
                    port_items.append("SD")
            elif ports.get('microsd'):
                port_items.append("microSD")
            if ports.get('audio'):
                port_items.append("Audio")

            if port_items:
                specs.append("Ports: " + " + ".join(port_items))

        # Check for power delivery
        power_delivery = meta.get('POWERDELIVERY', '') or meta.get('power_delivery', '')
        hub_pd = meta.get('hub_power_delivery', '')
        features = meta.get('features', [])

        if power_delivery or hub_pd or 'Power Delivery' in features:
            pd_match = re.search(r'(\d+)\s*w(?:att)?', all_text)
            if pd_match:
                specs.append(f"PD {pd_match.group(1)}W")
            elif hub_pd:
                specs.append(f"PD {hub_pd}")
            else:
                specs.append("PD")

        # Check for 4K support
        max_res = meta.get('MAXRESOLUTION', '') or meta.get('max_resolution', '')
        dock_4k = meta.get('DOCK4KSUPPORT', '')
        if '4k' in all_text or '4K' in str(max_res) or '2160' in str(max_res) or dock_4k == 'Yes':
            if '60' in all_text or '60hz' in all_text.replace(' ', ''):
                specs.append("4K@60Hz")
            elif '30' in all_text or '30hz' in all_text.replace(' ', ''):
                specs.append("4K@30Hz")
            else:
                specs.append("4K")

        # Add specs line
        if specs:
            line += f"\n   {', '.join(specs)}"

        return line

    def _parse_external_ports(self, external_ports: str) -> dict:
        """
        Parse EXTERNALPORTS field into categorized port counts.

        Args:
            external_ports: String like "1 x HDMI, 2 x USB 3.2 Type-A..."

        Returns:
            Dict with port type counts: {'hdmi': 1, 'usb_a': 2, 'ethernet': 1, ...}
        """
        if not external_ports or str(external_ports).lower() == 'nan':
            return {}

        ports_str = str(external_ports).lower()
        result = {}

        # Count HDMI ports
        hdmi_matches = re.findall(r'(\d+)\s*x\s*hdmi', ports_str)
        if hdmi_matches:
            result['hdmi'] = sum(int(m) for m in hdmi_matches)
        elif 'hdmi' in ports_str:
            result['hdmi'] = 1

        # Count DisplayPort
        dp_matches = re.findall(r'(\d+)\s*x\s*(?:displayport|display port|dp\b)', ports_str)
        if dp_matches:
            result['displayport'] = sum(int(m) for m in dp_matches)
        elif 'displayport' in ports_str or 'display port' in ports_str:
            result['displayport'] = 1

        # Count VGA
        if 'vga' in ports_str:
            vga_matches = re.findall(r'(\d+)\s*x\s*vga', ports_str)
            result['vga'] = sum(int(m) for m in vga_matches) if vga_matches else 1

        # Count USB Type-A ports (exclude USB-C and power-only)
        usb_a_matches = re.findall(r'(\d+)\s*x\s*usb\s*(?:3\.\d+\s*)?type-?a', ports_str)
        if usb_a_matches:
            result['usb_a'] = sum(int(m) for m in usb_a_matches)

        # Count USB Type-C ports (exclude input/host port, count output ports only)
        # Look for USB-C that's NOT "power delivery only"
        usb_c_entries = re.findall(r'(\d+)\s*x\s*usb[^,]*type-?c[^,]*', ports_str)
        usb_c_count = 0
        for entry in usb_c_entries:
            # Skip if it's power delivery only (passthrough charging)
            if 'power delivery only' not in entry.lower():
                match = re.search(r'(\d+)', entry)
                if match:
                    usb_c_count += int(match.group(1))
        if usb_c_count > 0:
            result['usb_c'] = usb_c_count

        # Count Ethernet (RJ-45)
        if 'rj-45' in ports_str or 'rj45' in ports_str or 'ethernet' in ports_str:
            eth_matches = re.findall(r'(\d+)\s*x\s*(?:rj-?45|ethernet)', ports_str)
            result['ethernet'] = sum(int(m) for m in eth_matches) if eth_matches else 1

        # Count SD card slots
        if 'sd' in ports_str:
            # Check for SD (not microSD)
            if re.search(r'\bsd\s*/\s*mmc\b|\bsd\s+slot\b|\bsd\s+card\b', ports_str):
                result['sd'] = 1
            # Check for microSD
            if 'microsd' in ports_str or 'micro sd' in ports_str:
                result['microsd'] = 1

        # Count audio ports
        if '3.5mm' in ports_str or 'audio' in ports_str or 'headphone' in ports_str:
            result['audio'] = 1

        return result

    def _simplify_interface_name(self, interface: str) -> str:
        """Simplify interface names for cleaner display."""
        if not interface:
            return ""

        intf = interface.strip()

        # Remove quantity prefix like "1 x ", "2 x ", "4 x "
        intf = re.sub(r'^\d+\s*x\s*', '', intf, flags=re.IGNORECASE)

        # Common simplifications
        simplifications = {
            r'RJ-?45.*Gigabit.*': 'Gigabit Ethernet',
            r'RJ-?45.*10/100.*': 'Fast Ethernet',
            r'RJ-?45.*10G.*': '10 Gigabit Ethernet',
            r'RJ-?45.*2\.5G.*': '2.5 Gigabit Ethernet',
            r'Serial.*RS-?232.*': 'RS-232 Serial',
            r'USB\s*3\.2\s*Gen\s*2.*10\s*G': 'USB 3.2 Gen 2 (10Gbps)',
            r'USB\s*3\.2\s*Gen\s*1.*5\s*G': 'USB 3.2 Gen 1 (5Gbps)',
            r'USB\s*3\.0.*5\s*G': 'USB 3.0 (5Gbps)',
            r'SATA.*6\s*G': 'SATA III (6Gbps)',
            r'SAS.*12\s*G': 'SAS (12Gbps)',
        }

        for pattern, replacement in simplifications.items():
            if re.search(pattern, intf, re.IGNORECASE):
                return replacement

        # Remove pin counts and technical details
        intf = re.sub(r'\s*\(\d+\s*pin\)', '', intf, flags=re.IGNORECASE)
        intf = re.sub(r'\s*\(.*?\)', '', intf)  # Remove other parenthetical info

        return intf.strip()

    def _format_fiber_product_line(self, product, position: int) -> str:
        """
        Format a fiber optic cable product with fiber-specific information.

        Format: 1. MPO12PL10M - 32.8ft Multimode Fiber (MPO/MTP to MPO/MTP, 850nm)

        Shows length, fiber type, connector type, and wavelength.
        """
        sku = product.product_number
        length_ft = product.metadata.get('length_ft')
        length_m = product.metadata.get('length_m')
        fiber_type = product.metadata.get('fiber_type', '')
        fiber_connector = product.metadata.get('fiber_connector', '')
        fiber_duplex = product.metadata.get('fiber_duplex', '')
        fiber_wavelength = product.metadata.get('fiber_wavelength', '')
        connectors = product.metadata.get('connectors', [])

        # Build the main description
        parts = [f"**{position}. {sku}**"]

        # Add length
        if length_ft:
            if length_m and length_m >= 1:
                parts.append(f" - {length_ft}ft [{length_m}m]")
            else:
                parts.append(f" - {length_ft}ft")

        # Add fiber type (Multimode/Single-mode)
        if fiber_type:
            parts.append(f" {fiber_type}")

        parts.append(" Fiber")

        # Build specs in parentheses
        spec_parts = []

        # Connector type (prefer fiber_connector, fall back to connectors)
        if fiber_connector:
            spec_parts.append(f"{fiber_connector} to {fiber_connector}")
        elif connectors and len(connectors) >= 2:
            # Clean up connector strings
            conn_a = str(connectors[0]).replace('1 x ', '').replace('Fiber Optic ', '')
            conn_b = str(connectors[1]).replace('1 x ', '').replace('Fiber Optic ', '')
            spec_parts.append(f"{conn_a} to {conn_b}")

        # Add duplex/simplex if available
        if fiber_duplex:
            spec_parts.append(fiber_duplex)

        # Add wavelength
        if fiber_wavelength:
            spec_parts.append(fiber_wavelength)

        if spec_parts:
            parts.append(f" ({', '.join(spec_parts)})")

        return "".join(parts)

    def _format_storage_product_line(self, product, position: int) -> str:
        """
        Format a storage enclosure product with storage-specific information.

        Format: 1. SM21BMU31C3 - M.2 SATA Drive Enclosure (USB 3.2 Gen 2, Aluminum)

        Shows drive size, interface, and key features.
        """
        sku = product.product_number
        drive_size = product.metadata.get('drive_size', '')
        storage_interface = product.metadata.get('storage_interface', '')
        num_drives = product.metadata.get('num_drives')
        enclosure_material = product.metadata.get('enclosure_material', '')
        tool_free = product.metadata.get('tool_free', False)
        sub_category = product.metadata.get('sub_category', '')
        name = product.metadata.get('name', '')

        # Build the main description
        parts = [f"**{position}. {sku}**"]

        # Build description
        desc_parts = []

        # Number of drives
        if num_drives and num_drives > 1:
            desc_parts.append(f"{num_drives}-Bay")

        # Drive size
        if drive_size:
            desc_parts.append(drive_size)

        # Determine product type
        sku_lower = sku.lower()
        sub_lower = sub_category.lower()

        if 'external' in sub_lower:
            desc_parts.append("Drive Enclosure")
        elif 'adapter' in sub_lower or 'converter' in sub_lower:
            desc_parts.append("Drive Adapter")
        elif 'dock' in sub_lower or 'docking' in sub_lower:
            desc_parts.append("Drive Dock")
        elif 'bracket' in sku_lower:
            desc_parts.append("Drive Bracket")
        else:
            desc_parts.append("Drive Enclosure")

        if desc_parts:
            parts.append(f" - {' '.join(desc_parts)}")

        # Build specs in parentheses
        spec_parts = []

        # Interface
        if storage_interface:
            spec_parts.append(storage_interface)

        # Material
        if enclosure_material:
            mat_lower = enclosure_material.lower()
            if 'aluminum' in mat_lower and 'plastic' in mat_lower:
                spec_parts.append("Aluminum/Plastic")
            elif 'aluminum' in mat_lower:
                spec_parts.append("Aluminum")
            elif 'steel' in mat_lower:
                spec_parts.append("Steel")
            elif 'plastic' in mat_lower:
                spec_parts.append("Plastic")
            elif 'metal' in mat_lower:
                spec_parts.append("Metal")

        # Tool-free
        if tool_free:
            spec_parts.append("Tool-free")

        if spec_parts:
            parts.append(f" ({', '.join(spec_parts)})")

        return "".join(parts)

    def _format_privacy_screen_line(self, product, position: int) -> str:
        """
        Format a privacy screen product with privacy-specific information.

        Format: 1. 135CT-PRIVACY-SCREEN - 13.5" Laptop Privacy Filter (Magnetic, Touch)

        Shows screen size, device type, and attachment method.
        """
        sku = product.product_number
        sub_category = product.metadata.get('sub_category', '')
        name = product.metadata.get('name', '')
        sku_lower = sku.lower()
        sub_lower = sub_category.lower()
        name_lower = name.lower()

        # Build the main description
        parts = [f"**{position}. {sku}**"]

        desc_parts = []

        # Extract screen size from SKU or name
        # SKUs often have size like "135CT" = 13.5", "156W9B" = 15.6", "24MAM" = 24"
        screen_size = self._extract_privacy_screen_size(sku, name, sub_category)
        if screen_size:
            desc_parts.append(screen_size)

        # Determine device type (laptop, monitor, tablet)
        if 'laptop' in sub_lower or 'laptop' in name_lower or 'notebook' in name_lower:
            desc_parts.append("Laptop")
        elif 'monitor' in sub_lower or 'monitor' in name_lower or 'desktop' in name_lower:
            desc_parts.append("Monitor")
        elif 'tablet' in sub_lower or 'tablet' in name_lower or 'ipad' in name_lower:
            desc_parts.append("Tablet")
        elif 'macbook' in name_lower:
            desc_parts.append("MacBook")

        desc_parts.append("Privacy Filter")

        if desc_parts:
            parts.append(f" - {' '.join(desc_parts)}")

        # Build specs in parentheses
        spec_parts = []

        # Attachment method - check SKU patterns and name
        combined = f"{sku_lower} {name_lower}"
        if 'magnetic' in combined or 'mag' in sku_lower:
            spec_parts.append("Magnetic")
        elif 'adhesive' in combined or 'adh' in combined:
            spec_parts.append("Adhesive")
        elif 'slide' in combined:
            spec_parts.append("Slide-mount")
        elif 'hang' in combined or 'tab' in combined:
            spec_parts.append("Hanging tabs")

        # Touch compatible
        if 'touch' in combined:
            spec_parts.append("Touch-compatible")

        # Glossy/Matte
        if 'matte' in combined:
            spec_parts.append("Matte")
        elif 'glossy' in combined or 'gloss' in combined:
            spec_parts.append("Glossy")

        # Aspect ratio (if mentioned in name)
        if '16:9' in name or 'widescreen' in name_lower:
            spec_parts.append("16:9")
        elif '16:10' in name:
            spec_parts.append("16:10")
        elif '4:3' in name:
            spec_parts.append("4:3")

        if spec_parts:
            parts.append(f" ({', '.join(spec_parts)})")

        return "".join(parts)

    def _extract_privacy_screen_size(self, sku: str, name: str, sub_category: str) -> str:
        """
        Extract screen size from privacy screen product info.

        Common patterns:
        - SKU: 135CT = 13.5", 156W9B = 15.6", 24MAM = 24", 13MAM = 13"
        - Name: often contains "13.5 inch" or "13.5-inch"
        """
        import re

        # Try to find size in name first (most reliable)
        # Match patterns like "13.5 inch", "24-inch", "15.6""
        size_match = re.search(r'(\d+(?:\.\d+)?)\s*[-"]?\s*(?:inch|in\b|")', name.lower())
        if size_match:
            size = size_match.group(1)
            return f'{size}"'

        # Try to extract from SKU
        # Pattern: leading digits might indicate size
        # 135CT -> 13.5, 156W -> 15.6, 24MAM -> 24, 13MAM -> 13
        sku_upper = sku.upper()

        # Check for 3-digit patterns that represent X.X sizes
        match = re.match(r'^(\d{3})', sku_upper)
        if match:
            digits = match.group(1)
            # 135 -> 13.5, 156 -> 15.6, etc.
            size = f"{digits[0:2]}.{digits[2]}"
            return f'{size}"'

        # Check for 2-digit patterns
        match = re.match(r'^(\d{2})', sku_upper)
        if match:
            size = match.group(1)
            return f'{size}"'

        return ""

    def _format_generic_product_line(self, product, position: int) -> str:
        """
        Smart fallback formatter for unknown product categories.

        Format: 1. SKU - Length, Sub Category (Feature1, Feature2)

        Uses sub_category as the primary description, with features in parentheses.
        This ensures any new product category shows useful info without a custom formatter.

        When features are empty, shows useful alternative metadata like color, material,
        rack units, etc.
        """
        sku = product.product_number
        sub_category = product.metadata.get('sub_category', '')
        features = product.metadata.get('features', [])
        category = product.metadata.get('category', '')
        metadata = product.metadata

        # Build main line
        parts = [f"**{position}. {sku}**"]

        # Add length if available (cables, some adapters, etc.)
        length_ft = metadata.get('length_ft')
        length_display = metadata.get('length_display')
        if length_display:
            parts.append(f" - {length_display}")
        elif length_ft:
            parts.append(f" - {length_ft}ft")

        # Use sub_category as description (it's usually descriptive)
        if sub_category:
            # Clean up sub_category - remove redundant "StarTech.com" prefix if present
            desc = sub_category
            if desc.lower().startswith('startech'):
                desc = desc.split(' ', 1)[-1] if ' ' in desc else desc
            parts.append(f" - {desc}")
        elif category:
            # Fallback to category if no sub_category
            parts.append(f" - {category.replace('_', ' ').title()}")

        # Add features in parentheses if available
        if features:
            # Limit to first 3 features to keep it concise
            feature_str = ", ".join(features[:3])
            parts.append(f" ({feature_str})")
        else:
            # No features - show useful alternative metadata
            alt_info = []

            # Rack-specific: rack units (UHEIGHT contains "10U", "14U", etc.)
            rack_height = metadata.get('UHEIGHT')
            if rack_height and str(rack_height).lower() not in ('nan', 'none', ''):
                alt_info.append(str(rack_height))

            # Rack type (2-Post, 4-Post, etc.)
            rack_type = metadata.get('RACKTYPE')
            if rack_type and str(rack_type).lower() not in ('nan', 'none', ''):
                alt_info.append(str(rack_type))

            # Material (for racks, mounts, etc.) - CONSTMATERIAL is the column name
            material = metadata.get('CONSTMATERIAL') or metadata.get('mount_material')
            if material and str(material).lower() not in ('nan', 'none', ''):
                alt_info.append(str(material))

            # Computer card specific: Bus type (PCI Express, USB 3.2, etc.)
            bus_type = metadata.get('BUSTYPE')
            if bus_type and str(bus_type).lower() not in ('nan', 'none', ''):
                alt_info.append(str(bus_type))

            # Card profile (Low Profile, Full Height) - critical for case compatibility
            card_profile = metadata.get('CARDPROFILE')
            if card_profile and str(card_profile).lower() not in ('nan', 'none', ''):
                alt_info.append(str(card_profile))

            # Number of ports (for cards, hubs)
            ports = metadata.get('NUMBERPORTS') or metadata.get('hub_ports')
            if ports and str(ports).lower() not in ('nan', 'none', '0', ''):
                try:
                    port_num = int(float(ports))
                    if port_num > 0:
                        alt_info.append(f"{port_num}-port")
                except (ValueError, TypeError):
                    pass

            # Color as last resort (common across many product types)
            color = metadata.get('color') or metadata.get('COLOR')
            if color and str(color).lower() not in ('nan', 'none', ''):
                alt_info.append(str(color))

            if alt_info:
                parts.append(f" ({', '.join(alt_info[:3])})")

        return "".join(parts)

    def _format_product_line(self, product, position: int, show_color: bool = False, query: str = None) -> str:
        """
        Format a single product as a concise line item.

        Format: 1. SKU - Length, Connector Type
                - Feature 1, Feature 2

        Special handling:
        - Hubs: Shows port count, USB version, power status
        - Ethernet switches: Shows port count, speed, features
        - KVM switches: Shows port count, video type, resolution, audio/USB
        - Network cables: Shows Cat rating (Cat5e, Cat6, Cat6a) prominently

        Args:
            product: Product to format
            position: Position number (1, 2, 3)
            show_color: Whether to include color in the display (when user asked for specific color)
            query: Original user query (for contextual formatting)
        """
        # Delegate to specialized formatters for non-cable products
        if self._is_hub_product(product):
            return self._format_hub_product_line(product, position)

        if self._is_dock_product(product):
            return self._format_dock_product_line(product, position, query=query)

        if self._is_ethernet_switch(product):
            return self._format_ethernet_switch_line(product, position)

        if self._is_kvm_switch(product):
            return self._format_kvm_switch_line(product, position)

        if self._is_mount_product(product):
            return self._format_mount_product_line(product, position)

        if self._is_video_splitter(product):
            return self._format_video_splitter_line(product, position)

        if self._is_fiber_cable(product):
            return self._format_fiber_product_line(product, position)

        if self._is_storage_enclosure(product):
            return self._format_storage_product_line(product, position)

        if self._is_privacy_screen(product):
            return self._format_privacy_screen_line(product, position)

        if self._is_multiport_adapter(product):
            return self._format_multiport_adapter_line(product, position)

        if self._is_pcie_card(product):
            return self._format_pcie_card_line(product, position)

        # Smart fallback for unknown categories - show sub_category info
        # This catches any product type we haven't built a custom formatter for
        category = product.metadata.get('category', '').lower()
        cable_like_categories = {'cable', 'adapter', 'dock', 'power', 'network', 'other', ''}
        if category not in cable_like_categories:
            return self._format_generic_product_line(product, position)

        sku = product.product_number
        length_ft = product.metadata.get('length_ft')
        connectors = product.metadata.get('connectors', [])
        network_rating = product.metadata.get('network_rating')  # e.g., "Cat6a"
        network_speed = product.metadata.get('network_max_speed')  # e.g., "10 Gigabit"
        color = product.metadata.get('color')

        # Build the main line
        parts = [f"**{position}. {sku}**"]

        # Add length
        if length_ft:
            parts.append(f" - {length_ft}ft")

        # Check if this is a network cable (has Cat rating or RJ-45 connectors)
        is_network_cable = network_rating or (
            connectors and any('rj-45' in str(c).lower() for c in connectors)
        )

        if is_network_cable and network_rating:
            # Network cable with rating - show "Cat6a Ethernet" instead of "RJ-45"
            parts.append(f" {network_rating} Ethernet")
        elif connectors and len(connectors) >= 2:
            # Non-network cable - show connector types
            source = self._simplify_connector_name(str(connectors[0]))
            target = self._simplify_connector_name(str(connectors[1]))
            if source != target:
                parts.append(f" {source} to {target}")
            else:
                parts.append(f" {source}")

        line = "".join(parts)

        # Add feature bullets on same line in parentheses for cleaner Streamlit rendering
        features = self._get_feature_bullets(product)

        # For network cables, add speed rating as first feature if available
        if is_network_cable and network_speed and network_speed not in features:
            features = [network_speed] + features

        # Add color when requested (typically when color filter was dropped)
        if show_color and color:
            features = [color] + features

        if features:
            line += f" ({', '.join(features)})"

        return line

    def _normalize_connector_name(self, connector: str) -> str:
        """
        Normalize connector name for display.

        Args:
            connector: Raw connector name from filters

        Returns:
            Human-readable connector name
        """
        if not connector:
            return ""

        connector_lower = connector.lower()

        # Map to display names
        if 'usb-c' in connector_lower or 'usb c' in connector_lower or 'type-c' in connector_lower:
            return "USB-C"
        elif 'usb-a' in connector_lower or 'usb a' in connector_lower or 'type-a' in connector_lower:
            return "USB-A"
        elif 'displayport' in connector_lower or connector_lower == 'dp':
            return "DisplayPort"
        elif 'hdmi' in connector_lower:
            return "HDMI"
        elif 'thunderbolt' in connector_lower:
            return "Thunderbolt"
        elif 'vga' in connector_lower:
            return "VGA"
        elif 'dvi' in connector_lower:
            return "DVI"
        elif 'ethernet' in connector_lower or 'rj45' in connector_lower:
            return "Ethernet"
        else:
            return connector  # Return as-is if unknown

    def _get_product_type(
        self,
        query: str,
        connector_from: Optional[str] = None,
        connector_to: Optional[str] = None
    ) -> str:
        """
        Extract product type from query and/or actual connectors.

        Args:
            query: User's query text
            connector_from: Actual source connector (from filters, after device inference)
            connector_to: Actual target connector (from filters)

        Returns:
            Human-readable product type description

        IMPORTANT: When connector_from and connector_to are provided, use those
        rather than inferring from query text. Query text might say "HDMI or
        DisplayPort?" but the actual products are USB-C to HDMI.
        """
        query_lower = query.lower()

        # Non-cable product types (always infer from query)
        if 'hub' in query_lower:
            return "USB hubs"
        elif 'dock' in query_lower:
            return "docking stations"
        elif 'kvm' in query_lower:
            return "KVM switches"
        elif 'mount' in query_lower:
            if 'wall' in query_lower:
                return "wall mounts"
            elif 'desk' in query_lower:
                return "desk mounts"
            elif 'tv' in query_lower:
                return "TV mounts"
            else:
                return "monitor mounts"

        # Cable types - prefer actual connectors over query inference
        if connector_from and connector_to:
            # Use actual connector info from filters (after device inference)
            from_normalized = self._normalize_connector_name(connector_from)
            to_normalized = self._normalize_connector_name(connector_to)

            if from_normalized == to_normalized:
                return f"{from_normalized} cables"
            else:
                return f"{from_normalized} to {to_normalized} cables"

        # Fallback to query text inference
        if 'usb-c' in query_lower and 'hdmi' in query_lower:
            return "USB-C to HDMI cables"
        elif 'usb-c' in query_lower and 'displayport' in query_lower:
            return "USB-C to DisplayPort cables"
        elif 'displayport' in query_lower and 'hdmi' in query_lower:
            return "DisplayPort to HDMI cables"
        elif 'displayport' in query_lower:
            return "DisplayPort cables"
        elif 'hdmi' in query_lower:
            return "HDMI cables"
        elif 'thunderbolt' in query_lower:
            return "Thunderbolt cables"
        elif 'usb-c' in query_lower:
            return "USB-C cables"
        else:
            return "options"

    def _extract_requested_length(self, query: str) -> Optional[float]:
        """Extract the length user requested from query."""
        import re
        query_lower = query.lower()

        # Match patterns like "6ft", "6 ft", "6 foot", "10ft"
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:ft|foot|feet)', query_lower)
        if match:
            return float(match.group(1))

        # Word-based numbers
        word_to_num = {'six': 6, 'three': 3, 'ten': 10, 'five': 5}
        for word, num in word_to_num.items():
            if f'{word} foot' in query_lower or f'{word} ft' in query_lower:
                return float(num)

        return None

    def _user_is_flexible_on_length(self, query: str) -> bool:
        """Check if user indicated flexibility on length."""
        query_lower = query.lower()
        flexible_phrases = [
            'shorter is fine', 'shorter is ok', 'shorter works',
            'shorter would work', 'shorter is okay',
            'around', 'about', 'approximately', 'roughly'
        ]
        return any(phrase in query_lower for phrase in flexible_phrases)

    # Expert context for different product categories
    # Provides educational value like a knowledgeable sales rep would
    EXPERT_CONTEXT = {
        'fiber': (
            "Fiber optic cables offer much higher bandwidth and longer distances than copper - "
            "ideal for data centers, high-speed networks, or long runs where signal quality matters."
        ),
        'privacy': (
            "Privacy screens limit viewing angles so only you can see your display - "
            "great for working with sensitive data in open offices or public spaces."
        ),
        'multiport': (
            "Multiport adapters turn one USB-C port into multiple connections - "
            "perfect for laptops with limited ports that need video, USB, and more."
        ),
        'kvm': (
            "KVM switches let you control multiple computers from one keyboard, mouse, and monitor - "
            "saves desk space and makes switching between systems seamless."
        ),
        'mount': (
            "Monitor mounts free up desk space and let you position screens at the perfect height - "
            "better ergonomics and a cleaner setup."
        ),
        'enclosure': (
            "Drive enclosures turn internal drives into portable external storage - "
            "great for backups, data migration, or repurposing old drives."
        ),
        'hub': (
            "USB hubs expand your port options - "
            "connect more devices without constantly swapping cables."
        ),
        'dock': (
            "Docking stations turn your laptop into a desktop setup with one cable - "
            "multiple monitors, USB devices, and charging all at once."
        ),
        'switch': (
            "Network switches expand your wired connections - "
            "more reliable and faster than WiFi for devices that need consistent connectivity."
        ),
        'splitter': (
            "Video splitters send one source to multiple displays - "
            "great for presentations, digital signage, or mirroring content."
        ),
        'rack': (
            "Server racks organize and secure your equipment - "
            "proper mounting improves airflow and makes maintenance easier."
        ),
        'card': (
            "Expansion cards add capabilities your motherboard doesn't have built-in - "
            "extra ports, faster networking, or specialized functions."
        ),
    }

    def _get_expert_context(self, query: str) -> Optional[str]:
        """
        Get expert context based on the product type in the query.

        Returns a 1-2 sentence educational explanation of why someone
        might want this product, like a knowledgeable sales rep would provide.
        """
        query_lower = query.lower()

        # Check for specific product types
        if 'fiber' in query_lower or 'optic' in query_lower:
            return self.EXPERT_CONTEXT['fiber']
        elif 'privacy' in query_lower and ('screen' in query_lower or 'filter' in query_lower):
            return self.EXPERT_CONTEXT['privacy']
        elif 'multiport' in query_lower or 'multi-port' in query_lower or 'multi port' in query_lower:
            return self.EXPERT_CONTEXT['multiport']
        elif 'kvm' in query_lower:
            return self.EXPERT_CONTEXT['kvm']
        elif 'mount' in query_lower or 'arm' in query_lower:
            return self.EXPERT_CONTEXT['mount']
        elif 'enclosure' in query_lower or ('drive' in query_lower and 'external' in query_lower):
            return self.EXPERT_CONTEXT['enclosure']
        elif 'hub' in query_lower and 'usb' in query_lower:
            return self.EXPERT_CONTEXT['hub']
        elif 'dock' in query_lower:
            return self.EXPERT_CONTEXT['dock']
        elif 'switch' in query_lower and ('ethernet' in query_lower or 'network' in query_lower):
            return self.EXPERT_CONTEXT['switch']
        elif 'splitter' in query_lower:
            return self.EXPERT_CONTEXT['splitter']
        elif 'rack' in query_lower:
            return self.EXPERT_CONTEXT['rack']
        elif 'card' in query_lower and ('pci' in query_lower or 'expansion' in query_lower):
            return self.EXPERT_CONTEXT['card']

        return None

    def _generate_intro(
        self,
        products: List,
        query: str,
        connector_from: Optional[str] = None,
        connector_to: Optional[str] = None
    ) -> str:
        """
        Generate a warm, helpful intro like a real CSR would.

        Acknowledges the customer first, provides expert context about
        the product type, then introduces the results.
        """
        query_lower = query.lower()
        product_type = self._get_product_type(query, connector_from, connector_to)
        requested_length = self._extract_requested_length(query)
        is_flexible = self._user_is_flexible_on_length(query)

        # Get actual lengths in results
        lengths = [p.metadata.get('length_ft') for p in products if p.metadata.get('length_ft')]
        has_exact_length = requested_length and any(
            abs(l - requested_length) < 0.5 for l in lengths
        ) if lengths else False

        # Get expert context for this product type
        expert_context = self._get_expert_context(query)

        # Build the intro
        if requested_length and not has_exact_length and is_flexible:
            # User asked for specific length, we don't have it, but they're flexible
            base = (
                f"I can help with that! We don't have {product_type} at exactly "
                f"{int(requested_length)}ft, but since you're flexible on length, "
                f"here are {len(products)} great options:"
            )
        elif requested_length and not has_exact_length:
            # User asked for specific length, we don't have it
            base = (
                f"I can help with that! We don't have {product_type} at exactly "
                f"{int(requested_length)}ft, but here are {len(products)} close alternatives:"
            )
        elif requested_length and has_exact_length:
            # We have what they asked for
            base = (
                f"I can help with that! Here are {len(products)} {product_type} "
                f"that match what you're looking for:"
            )
        elif expert_context:
            # General search with expert context - add the educational value
            return f"{expert_context}\n\nHere are {len(products)} options:"
        else:
            # General search, no specific length, no expert context
            base = f"I can help with that! Here are {len(products)} {product_type}:"

        return base

    def build_response(
        self,
        ranked_products: List,  # List of RankedProduct
        query: str,
        intro_text: Optional[str] = None,
        dropped_filters: Optional[List] = None,
        original_filters: Optional[Dict] = None,
    ) -> str:
        """
        Build a conversational response with product recommendations.

        Structure:
        - Positive opening acknowledging what we found
        - Combined constraints explanation (if multiple filters dropped)
        - Numbered product list with key specs
        - Helpful closing question with alternatives

        Args:
            ranked_products: List of RankedProduct objects
            query: Original user query
            intro_text: Optional intro from domain rules
            dropped_filters: List of DroppedFilter objects from search
            original_filters: Original SearchFilters as dict (for context)
        """
        products = [rp.product for rp in ranked_products[:3]]

        # Build response
        response_parts = []

        # Collect individual constraint info (structured, not formatted)
        constraints = self._collect_constraints(
            dropped_filters, original_filters, products, query
        )

        # Determine if we should show color
        show_color = constraints.get('color') is not None

        # Extract actual connectors from filters (after device inference)
        connector_from = original_filters.get('connector_from') if original_filters else None
        connector_to = original_filters.get('connector_to') if original_filters else None

        # Build the opening and constraints section
        if constraints:
            # Positive opening based on what we DID find
            opening = self._build_positive_opening(
                products, query, constraints, connector_from, connector_to
            )
            response_parts.append(opening)
            response_parts.append("")

            # Combined constraints explanation
            combined = self._build_combined_constraints(constraints, query)
            if combined:
                response_parts.append(combined)
                response_parts.append("")
        else:
            # No constraints - use standard intro
            intro = self._generate_intro(products, query, connector_from, connector_to)
            response_parts.append(intro)
            response_parts.append("")

        # Add domain rule intro if provided (e.g., technical tip or note)
        # But skip if it duplicates what we already said in constraints
        if intro_text and not self._intro_duplicates_constraints(intro_text, constraints):
            intro_clean = intro_text.strip()
            # If it's already a Note (from fallback search), don't wrap with Tip
            if intro_clean.startswith("**Note:**") or intro_clean.startswith("Note:"):
                response_parts.append(intro_clean)
            else:
                response_parts.append(f"**Tip:** {intro_clean}")
            response_parts.append("")

        # Product list
        for i, rp in enumerate(ranked_products[:3], 1):
            product_line = self._format_product_line(rp.product, i, show_color=show_color, query=query)
            response_parts.append(product_line)

            # Add differentiating note for products 2 and 3
            note = self._get_differentiating_note(rp, i, ranked_products, query)
            if note:
                response_parts.append(f"   {note}")

            response_parts.append("")  # Blank line between products

        # Closing - context-aware based on what constraints were hit
        closing = self._build_contextual_closing(constraints, query)
        response_parts.append(closing)

        return "\n".join(response_parts)

    def _collect_constraints(
        self,
        dropped_filters: Optional[List],
        original_filters: Optional[Dict],
        products: List,
        query: str
    ) -> Dict[str, dict]:
        """
        Collect all constraint information into a structured dict.

        Returns dict with keys like 'color', 'features', 'length' containing
        the constraint details (not formatted text).
        """
        constraints = {}

        if not dropped_filters or not original_filters:
            return constraints

        for df in dropped_filters:
            if not hasattr(df, 'filter_name'):
                continue

            if df.filter_name == 'color':
                actual_colors = set()
                for prod in products:
                    color = prod.metadata.get('color')
                    if color:
                        actual_colors.add(color)
                constraints['color'] = {
                    'requested': original_filters.get('color'),
                    'available': sorted(actual_colors)
                }

            elif df.filter_name == 'features':
                requested_features = df.requested_value if isinstance(df.requested_value, list) else [df.requested_value]

                # CRITICAL: Verify products actually DON'T have the features
                # The cascading search may have dropped the filter, but the
                # results might still have products with those features.
                actually_missing = []
                for feat in requested_features:
                    feat_lower = feat.lower()
                    # Check if ANY product has this feature
                    has_feature = False
                    for prod in products:
                        if feat_lower == '4k':
                            # Use unified 4K detection method
                            has_feature = prod.supports_4k()
                        else:
                            prod_features = prod.metadata.get('features', [])
                            has_feature = any(feat_lower in f.lower() for f in prod_features)
                        if has_feature:
                            break
                    if not has_feature:
                        actually_missing.append(feat)

                # Only record as constraint if features are ACTUALLY missing
                if actually_missing:
                    query_lower = query.lower()
                    video_outputs = ['hdmi', 'displayport', 'vga', 'dvi']
                    has_video = any(v in query_lower for v in video_outputs)
                    wants_pd = 'Power Delivery' in actually_missing

                    constraints['features'] = {
                        'requested': actually_missing,
                        'impossible_pd': has_video and wants_pd,
                        'other_features': [f for f in actually_missing if f != 'Power Delivery'] if wants_pd else actually_missing
                    }

            elif df.filter_name == 'length':
                actual_lengths = []
                for prod in products:
                    length_ft = prod.metadata.get('length_ft')
                    if length_ft and length_ft not in actual_lengths:
                        actual_lengths.append(length_ft)
                actual_lengths.sort()

                requested_length = original_filters.get('length')

                # Check if any available length is "close enough" to requested
                # (within 15% or 1ft, whichever is greater)
                # 6ft request with 6.6ft result = 10% diff = close enough!
                has_close_match = False
                if requested_length and actual_lengths:
                    tolerance = max(requested_length * 0.15, 1.0)  # 15% or 1ft
                    has_close_match = any(
                        abs(avail - requested_length) <= tolerance
                        for avail in actual_lengths
                    )

                # Only record constraint if NO close match found
                # If 6.6ft exists for 6ft request, that's close enough - don't complain
                if not has_close_match:
                    constraints['length'] = {
                        'requested': requested_length,
                        'available': actual_lengths
                    }

        return constraints

    def _build_positive_opening(
        self,
        products: List,
        query: str,
        constraints: Dict,
        connector_from: Optional[str] = None,
        connector_to: Optional[str] = None
    ) -> str:
        """Build a positive opening that acknowledges what we DID find."""
        product_type = self._get_product_type(query, connector_from, connector_to)

        # Check what features we DO have - use unified method for consistency
        has_4k = any(p.supports_4k() for p in products)

        if has_4k and 'features' in constraints:
            return f"I found {product_type} with 4K support! A couple things to know:"
        elif constraints:
            return f"I found some {product_type} that might work. A couple things to know:"
        else:
            return f"I found {len(products)} {product_type}:"

    def _build_combined_constraints(
        self,
        constraints: Dict,
        query: str
    ) -> Optional[str]:
        """
        Build a combined, conversational explanation of all constraints.

        Instead of multiple "Note:" lines, creates one cohesive block.
        """
        if not constraints:
            return None

        sections = []

        # Power Delivery explanation (most important domain knowledge)
        if 'features' in constraints and constraints['features'].get('impossible_pd'):
            sections.append(
                "**About Power Delivery:** USB-C to video cables can't carry charging power - "
                "HDMI/DisplayPort/VGA don't support it. For charging while connected, "
                "consider a USB-C dock with pass-through power."
            )

        # Length explanation
        if 'length' in constraints:
            requested = constraints['length']['requested']
            available = constraints['length']['available']

            if available:
                if len(available) == 1:
                    length_str = f"{available[0]}ft"
                elif len(available) == 2:
                    length_str = f"{available[0]}ft and {available[1]}ft"
                else:
                    length_str = f"{available[0]}ft to {available[-1]}ft"

                sections.append(
                    f"**About length:** We don't have {int(requested)}ft options for this cable type. "
                    f"Available: {length_str}."
                )

        # Color explanation
        if 'color' in constraints:
            requested = constraints['color']['requested']
            available = constraints['color']['available']

            if available:
                color_str = ", ".join(available) if len(available) > 1 else available[0]
                sections.append(
                    f"**About color:** We don't have {requested.lower()} options. "
                    f"Showing {color_str} instead."
                )

        # Other dropped features (not Power Delivery)
        if 'features' in constraints:
            other = constraints['features'].get('other_features', [])
            # Only mention if there are features we couldn't match AND it's not just PD
            if other and not constraints['features'].get('impossible_pd'):
                feature_str = ", ".join(other)
                sections.append(
                    f"**About features:** Couldn't find products with {feature_str}. "
                    f"Showing closest matches."
                )

        if not sections:
            return None

        return "\n\n".join(sections)

    def _intro_duplicates_constraints(
        self,
        intro_text: str,
        constraints: Dict
    ) -> bool:
        """Check if intro_text duplicates info already in constraints."""
        if not intro_text:
            return False

        intro_lower = intro_text.lower()

        # Skip if it's about 4K and we already explained features
        if '4k' in intro_lower and 'features' in constraints:
            return True

        # Skip if it's about power delivery and we already explained
        if 'power delivery' in intro_lower and constraints.get('features', {}).get('impossible_pd'):
            return True

        return False

    def _build_contextual_closing(
        self,
        constraints: Dict,
        query: str
    ) -> str:
        """Build a closing question based on what constraints were hit."""

        # If Power Delivery was requested but impossible, offer dock alternative
        if constraints.get('features', {}).get('impossible_pd'):
            return "Would you like specs on any of these, or should I find a dock with video output + charging?"

        # If color was dropped
        if 'color' in constraints:
            return "Would you like me to search for a different color?"

        # If length was dropped
        if 'length' in constraints:
            return "Would you like specs on any of these, or should I check for longer options in adapters?"

        # Default - use refinement offer or generic
        refinement_offer = self._generate_refinement_offer(query)
        if refinement_offer:
            return refinement_offer
        else:
            return "Would you like specs on any of these?"

    def _build_dropped_color_notice(
        self,
        dropped_filters: Optional[List],
        original_filters: Optional[Dict],
        products: List
    ) -> Optional[str]:
        """
        Build a notice when color filter was dropped because no matching products exist.

        Args:
            dropped_filters: List of DroppedFilter objects
            original_filters: Original filter dict with 'color' key
            products: List of products being shown

        Returns:
            Notice string or None if color wasn't dropped
        """
        if not dropped_filters or not original_filters:
            return None

        # Check if color was dropped
        color_dropped = None
        for df in dropped_filters:
            if hasattr(df, 'filter_name') and df.filter_name == 'color':
                color_dropped = df
                break

        if not color_dropped:
            return None

        # Get the requested color
        requested_color = original_filters.get('color')
        if not requested_color:
            return None

        # Get actual colors of products being shown
        actual_colors = set()
        for prod in products:
            color = prod.metadata.get('color')
            if color:
                actual_colors.add(color)

        # Build the notice
        if len(actual_colors) == 1:
            actual_color_str = list(actual_colors)[0]
            notice = f"I couldn't find **{requested_color.lower()}** products matching your criteria. Here are some in **{actual_color_str}** instead:"
        elif len(actual_colors) > 1:
            color_list = ", ".join(sorted(actual_colors))
            notice = f"I couldn't find **{requested_color.lower()}** products matching your criteria. Here are alternatives in {color_list}:"
        else:
            notice = f"I couldn't find **{requested_color.lower()}** products matching your criteria. Here are some alternatives:"

        return notice

    def _build_dropped_features_notice(
        self,
        dropped_filters: Optional[List],
        original_filters: Optional[Dict],
        query: str
    ) -> Optional[str]:
        """
        Build a notice when features were dropped because no matching products exist.

        Also provides domain knowledge for impossible combinations like:
        - USB-C to HDMI + Power Delivery (HDMI can't carry power)

        Args:
            dropped_filters: List of DroppedFilter objects
            original_filters: Original filter dict
            query: Original query for context

        Returns:
            Notice string or None if features weren't dropped
        """
        if not dropped_filters:
            return None

        # Check if features were dropped
        features_dropped = None
        for df in dropped_filters:
            if hasattr(df, 'filter_name') and df.filter_name == 'features':
                features_dropped = df
                break

        if not features_dropped:
            return None

        # Get dropped features
        dropped_feature_list = features_dropped.requested_value
        if not dropped_feature_list or not isinstance(dropped_feature_list, list):
            return None

        query_lower = query.lower()

        # Domain knowledge: Explain impossible combinations
        # USB-C/Thunderbolt to HDMI/DisplayPort/VGA/DVI + Power Delivery is impossible
        # because non-USB video interfaces can't carry power
        video_outputs = ['hdmi', 'displayport', 'vga', 'dvi']
        has_video_output = any(v in query_lower for v in video_outputs)
        wants_power_delivery = 'Power Delivery' in dropped_feature_list

        if has_video_output and wants_power_delivery:
            # Explain why PD isn't possible with video adapters
            non_pd_features = [f for f in dropped_feature_list if f != 'Power Delivery']
            if non_pd_features:
                return (
                    f"**Note:** Power Delivery isn't available on USB-C to video cables - "
                    f"HDMI/DisplayPort/VGA can't carry charging power. "
                    f"For charging, you'd need a USB-C dock or hub with pass-through power.\n"
                    f"Showing cables with {', '.join(non_pd_features)}:"
                )
            else:
                return (
                    f"**Note:** Power Delivery isn't available on USB-C to video cables - "
                    f"HDMI/DisplayPort/VGA can't carry charging power. "
                    f"For charging while using video output, consider a USB-C dock with pass-through power."
                )

        # Generic message for other dropped features
        if len(dropped_feature_list) == 1:
            return f"**Note:** Couldn't find products with {dropped_feature_list[0]}. Showing alternatives:"
        else:
            return f"**Note:** No products found with all features: {', '.join(dropped_feature_list)}. Showing closest matches:"

    def _build_dropped_length_notice(
        self,
        dropped_filters: Optional[List],
        original_filters: Optional[Dict],
        products: List
    ) -> Optional[str]:
        """
        Build a notice when length filter was dropped because no matching products exist.

        Args:
            dropped_filters: List of DroppedFilter objects
            original_filters: Original filter dict with 'length' key
            products: List of products being shown (to show actual lengths available)

        Returns:
            Notice string or None if length wasn't dropped
        """
        if not dropped_filters or not original_filters:
            return None

        # Check if length was dropped
        length_dropped = None
        for df in dropped_filters:
            if hasattr(df, 'filter_name') and df.filter_name == 'length':
                length_dropped = df
                break

        if not length_dropped:
            return None

        # Get the requested length
        requested_length = original_filters.get('length')
        if not requested_length:
            return None

        # Get actual lengths of products being shown
        actual_lengths = []
        for prod in products:
            length_ft = prod.metadata.get('length_ft')
            if length_ft and length_ft not in actual_lengths:
                actual_lengths.append(length_ft)

        actual_lengths.sort()

        # Build the notice
        if actual_lengths:
            if len(actual_lengths) == 1:
                length_str = f"{actual_lengths[0]}ft"
            elif len(actual_lengths) == 2:
                length_str = f"{actual_lengths[0]}ft and {actual_lengths[1]}ft"
            else:
                length_str = f"{actual_lengths[0]}ft to {actual_lengths[-1]}ft"

            return (
                f"**Note:** We don't have **{int(requested_length)}ft** options for this cable type. "
                f"Available lengths: {length_str}. Here are the closest matches:"
            )
        else:
            return (
                f"**Note:** We don't have **{int(requested_length)}ft** options for this cable type. "
                f"Here are the available alternatives:"
            )

    def _get_differentiating_note(
        self,
        ranked_product,
        position: int,
        all_ranked: List,
        query: str
    ) -> Optional[str]:
        """
        Get a note that differentiates this product from others.
        Only adds a note if there's something meaningful to say.
        Returns None to avoid repetitive/filler text.
        """
        product = ranked_product.product
        length_ft = product.metadata.get('length_ft')
        features = product.metadata.get('features', [])
        query_lower = query.lower()

        # For position 1, no note needed - it's the best match
        if position == 1:
            return None

        # For position 2, explain why it's different
        if position == 2:
            first_product = all_ranked[0].product
            first_length = first_product.metadata.get('length_ft')

            # If this is the shorter option
            if length_ft and first_length and length_ft < first_length - 1:
                return f"^ Shorter option at {length_ft}ft"

            # If this is longer
            if length_ft and first_length and length_ft > first_length + 1:
                return f"^ Longer reach at {length_ft}ft"

            # If different resolution - use unified method for consistency
            if first_product.supports_4k() and not product.supports_4k():
                if product.supports_resolution('1080p'):
                    return "^ 1080p version (more affordable)"

        # For position 3, only add note if truly different
        if position == 3:
            # Check if it has unique features - use unified method for consistency
            if not product.supports_4k() and any(all_ranked[j].product.supports_4k() for j in range(2)):
                if product.supports_resolution('1080p'):
                    return "^ Budget-friendly 1080p option"

            # If it's a different length than both others
            other_lengths = [all_ranked[j].product.metadata.get('length_ft') for j in range(2)]
            valid_other_lengths = [ol for ol in other_lengths if ol]

            # Only compare if we have valid lengths to compare against
            if length_ft and valid_other_lengths:
                if all(length_ft != ol for ol in valid_other_lengths):
                    if length_ft > max(valid_other_lengths):
                        return f"^ Longest option at {length_ft}ft"

        return None

    def build_explanation(
        self,
        ranked_product,
        position: int,
        query: str
    ) -> ProductExplanation:
        """
        Build explanation for a single ranked product.
        Kept for backwards compatibility.
        """
        product = ranked_product.product
        formatted = self._format_product_line(product, position)

        return ProductExplanation(
            product_number=product.product_number,
            name=product.metadata.get('name', ''),
            formatted_text=formatted
        )
