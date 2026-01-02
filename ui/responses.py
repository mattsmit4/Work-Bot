"""
Response formatting for ST-Bot UI.

Handles all response formatting including products, conversations,
errors, and educational content.
"""

import re
from typing import List, Optional
from core.context import Product, SearchResult, DroppedFilter
from llm.prompts import get_system_prompts


class ResponseFormatter:
    """
    Formats chatbot responses for display.
    
    Features:
    - Product result formatting
    - Conversation formatting (greetings, farewells)
    - Error formatting
    - Context notes and educational tips
    - Markdown formatting for rich display
    
    Example:
        formatter = ResponseFormatter()
        response = formatter.format_product_response(
            products=search_results,
            query="USB-C cable",
            context_note="💡 Tip: For 4K support..."
        )
    """
    
    def __init__(self):
        """Initialize response formatter."""
        self.prompts = get_system_prompts()
    
    def format_product_response(
        self,
        products: List[Product],
        query: str,
        context_note: Optional[str] = None,
        tier: Optional[str] = None,
        search_result: Optional[SearchResult] = None
    ) -> str:
        """
        Format a product search response.

        Args:
            products: List of products found
            query: Original search query
            context_note: Optional educational note to add
            tier: Search tier used (tier1, tier2, tier3)
            search_result: Optional SearchResult with filter relaxation info

        Returns:
            Formatted response string

        Example:
            >>> response = formatter.format_product_response(
            ...     products=[product1, product2],
            ...     query="HDMI cable",
            ...     context_note="Tip: For 4K support..."
            ... )
        """
        response = ""

        # Add transparency message if filters were relaxed
        if search_result and search_result.had_filter_relaxation():
            transparency_msg = self._format_filter_relaxation_message(search_result)
            if transparency_msg:
                response += f"{transparency_msg}\n\n"

        # Summary
        summary = self.prompts.format_product_summary(len(products), query)
        response += f"{summary}\n\n"

        # Products
        if products:
            for i, product in enumerate(products, 1):
                response += self._format_single_product(product, i)
                response += "\n"

        # Context note
        if context_note:
            response += f"\n{context_note}"

        # Search tier info (for debugging/transparency)
        if tier:
            response += f"\n\n_Search tier: {tier}_"

        return response.strip()

    def _format_filter_relaxation_message(self, search_result: SearchResult) -> str:
        """
        Format a transparency message explaining why filters were relaxed.

        Args:
            search_result: SearchResult with dropped filter info

        Returns:
            Formatted transparency message
        """
        messages = []

        for dropped in search_result.dropped_filters:
            if dropped.filter_name == "length":
                msg = self._format_length_relaxation(dropped)
                if msg:
                    messages.append(msg)
            elif dropped.filter_name == "features":
                msg = self._format_features_relaxation(dropped)
                if msg:
                    messages.append(msg)

        return "\n".join(messages)

    def _format_length_relaxation(self, dropped: DroppedFilter) -> str:
        """
        Format message for length filter relaxation.

        Args:
            dropped: DroppedFilter for length

        Returns:
            User-friendly message about length unavailability
        """
        requested = dropped.requested_value

        if dropped.alternatives:
            alt_str = ", ".join(dropped.alternatives)
            return (
                f"**Note:** We don't have cables at exactly {requested}.\n"
                f"Available lengths: {alt_str}\n"
                f"Showing the closest options:"
            )
        else:
            return (
                f"**Note:** We don't have cables at exactly {requested}. "
                f"Showing the closest available options:"
            )

    def _format_features_relaxation(self, dropped: DroppedFilter) -> str:
        """
        Format message for features filter relaxation.

        Args:
            dropped: DroppedFilter for features

        Returns:
            User-friendly message about feature unavailability
        """
        features = dropped.requested_value
        if isinstance(features, list) and features:
            feature_str = ", ".join(features)
            return f"**Note:** No products found with all features: {feature_str}. Showing closest matches:"
        return ""
    
    def _format_single_product(self, product: Product, index: int) -> str:
        """
        Format a single product for display.

        Args:
            product: Product to format
            index: Product number in list

        Returns:
            Formatted product string
        """
        # Route to specialized formatters based on product type
        if self._is_multiport_adapter(product):
            return self._format_multiport_adapter(product, index)

        if self._is_pcie_card(product):
            return self._format_pcie_card(product, index)

        # Standard cable/adapter formatting
        name = product.metadata.get('name', 'Unknown Product')
        sku = product.product_number

        # Build product display
        result = f"**{index}. {name}**\n"
        result += f"   SKU: {sku}\n"

        # Add key details
        # Prefer formatted length display (e.g., "6.0 ft [1.8 m]")
        length_display = product.metadata.get('length_display')
        if length_display:
            result += f"   Length: {length_display}\n"
        else:
            # Fallback to old format
            length = product.metadata.get('length')
            length_unit = product.metadata.get('length_unit')
            if length and length_unit:
                result += f"   Length: {length}{length_unit}\n"

        features = product.metadata.get('features', [])
        if features:
            result += f"   Features: {', '.join(features)}\n"

        connectors = product.metadata.get('connectors', [])
        if connectors and len(connectors) >= 2:
            result += f"   Connectors: {connectors[0]} → {connectors[1]}\n"

        return result

    def _is_multiport_adapter(self, product: Product) -> bool:
        """
        Check if this is a multiport adapter (USB-C hub, travel dock, etc.).

        Args:
            product: Product to check

        Returns:
            True if this is a multiport adapter, False otherwise
        """
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

    def _format_multiport_adapter(self, product: Product, index: int) -> str:
        """
        Format a multiport adapter with port configuration information.

        Shows input type, output ports, power delivery, and video support
        instead of cable-specific info like length.

        Args:
            product: Product to format
            index: Product number in list

        Returns:
            Formatted multiport adapter string
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
        if input_type:
            result = f"**{index}. {sku}** - {input_type} Multiport Adapter\n"
        else:
            result = f"**{index}. {sku}** - Multiport Adapter\n"

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
            # Extract wattage if available
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

        # Add specs line if we have any
        if specs:
            result += f"   {', '.join(specs)}\n"

        return result

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

    def _is_pcie_card(self, product: Product) -> bool:
        """
        Check if this is a PCIe/computer expansion card.

        Args:
            product: Product to check

        Returns:
            True if this is an expansion card, False otherwise
        """
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

        return False

    def _format_pcie_card(self, product: Product, index: int) -> str:
        """
        Format a PCIe/computer expansion card with card-specific information.

        Shows bus type, card profile, port count, and interface type
        instead of cable-specific info like length and connectors.

        Args:
            product: Product to format
            index: Product number in list

        Returns:
            Formatted card string
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
        elif 'riser' in all_text or 'extender' in all_text or 'adapter' in all_text or 'slot' in all_text:
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

        # Build the main line
        if bus_display:
            result = f"**{index}. {sku}** - {bus_display} {card_type}\n"
        else:
            result = f"**{index}. {sku}** - {card_type}\n"

        # Build feature list
        features = []

        # Port count and interface type
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

        # Card profile (critical for case compatibility)
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

        # Add feature line if we have features
        if features:
            feature_str = ", ".join(features[:4])  # Limit to 4 features
            result += f"   {feature_str}\n"

        return result

    def _simplify_interface_name(self, interface: str) -> str:
        """
        Simplify interface names for cleaner display.

        Args:
            interface: Raw interface string (e.g., "4 x RJ-45 (Gigabit Ethernet)")

        Returns:
            Simplified interface name (e.g., "RJ-45")
        """
        if not interface:
            return ""

        intf = interface.strip()

        # Remove count prefix like "4 x " or "1x "
        intf = re.sub(r'^\d+\s*x\s*', '', intf, flags=re.IGNORECASE)

        # Extract just the connector type from parentheses patterns
        # "RJ-45 (Gigabit Ethernet)" -> "RJ-45"
        paren_match = re.match(r'^([^(]+)\s*\(', intf)
        if paren_match:
            intf = paren_match.group(1).strip()

        return intf
    
    def format_greeting(self) -> str:
        """
        Format a greeting response.
        
        Returns:
            Formatted greeting
            
        Example:
            >>> greeting = formatter.format_greeting()
        """
        return self.prompts.format_greeting_response()
    
    def format_farewell(self) -> str:
        """
        Format a farewell response.
        
        Returns:
            Formatted farewell
            
        Example:
            >>> farewell = formatter.format_farewell()
        """
        return self.prompts.format_farewell_response()
    
    def format_blocked_request(
        self,
        reason: str,
        alternatives: Optional[List[str]] = None
    ) -> str:
        """
        Format a blocked request response.
        
        Args:
            reason: Why request was blocked
            alternatives: Alternative suggestions
            
        Returns:
            Formatted blocked request message
            
        Example:
            >>> response = formatter.format_blocked_request(
            ...     reason="Daisy-chaining not supported",
            ...     alternatives=["Use docking station", "Individual cables"]
            ... )
        """
        return self.prompts.format_blocked_request(reason, alternatives)
    
    def format_no_results(
        self,
        query: str,
        suggestions: Optional[List[str]] = None
    ) -> str:
        """
        Format a no results response.
        
        Args:
            query: Original query
            suggestions: Suggestions for user
            
        Returns:
            Formatted no results message
            
        Example:
            >>> response = formatter.format_no_results(
            ...     query="50ft cable",
            ...     suggestions=["Try shorter length", "Check spelling"]
            ... )
        """
        return self.prompts.format_no_results_response(query, suggestions)
    
    def format_error(self, error_type: str) -> str:
        """
        Format an error response.
        
        Args:
            error_type: Type of error
            
        Returns:
            Formatted error message
            
        Example:
            >>> error = formatter.format_error("search_failed")
        """
        return self.prompts.format_error_response(error_type)
    
    def format_ambiguous_query(self) -> str:
        """
        Format an ambiguous query response.

        Returns:
            Message asking for clarification

        Example:
            >>> response = formatter.format_ambiguous_query()
        """
        return self.prompts.format_ambiguous_query_response()

    def format_setup_guidance(self, setup_type: str, meta_info: dict) -> str:
        """
        Format a setup guidance response with diagnostic questions.

        Args:
            setup_type: Type of setup (e.g., 'multi_monitor')
            meta_info: Additional info extracted from query

        Returns:
            Formatted guidance response with questions

        Example:
            >>> response = formatter.format_setup_guidance(
            ...     'multi_monitor',
            ...     {'monitor_count': 3}
            ... )
        """
        if setup_type == 'multi_monitor':
            return self._format_multi_monitor_guidance(meta_info)

        if setup_type == 'single_monitor':
            return self._format_single_monitor_guidance(meta_info)

        if setup_type == 'dock_selection':
            return self._format_dock_selection_guidance(meta_info)

        if setup_type == 'kvm_selection':
            return self._format_kvm_selection_guidance(meta_info)

        # Fallback for unknown setup types
        return (
            "I'd like to help you find the right products for your setup. "
            "Could you tell me more about what you're trying to connect?"
        )

    def _format_multi_monitor_guidance(self, meta_info: dict) -> str:
        """
        Format multi-monitor setup guidance.

        Args:
            meta_info: Info including monitor_count

        Returns:
            Diagnostic questions for multi-monitor setup
        """
        monitor_count = meta_info.get('monitor_count')

        # Build personalized intro based on monitor count
        if monitor_count:
            intro = f"Setting up {monitor_count} monitors"
        else:
            intro = "Setting up multiple monitors"

        return f"""{intro} - I can help with that!

To recommend the right solution, I need to know about your setup. Please answer these three questions:

**Your computer's video outputs:**
(USB-C, Thunderbolt, HDMI, DisplayPort, etc.)

**Your monitors' inputs:**
(List what each monitor has - e.g., "HDMI, DisplayPort, VGA")

**Your preference:**
Individual cables for each monitor, or a docking station?

---

**Example response:**
"USB-C and HDMI ports on my laptop.
Monitor 1: HDMI, Monitor 2: DisplayPort, Monitor 3: VGA.
Individual cables please."

Once I know your setup, I'll recommend the exact cables or adapters you need!"""

    def _format_single_monitor_guidance(self, meta_info: dict) -> str:
        """
        Format single monitor connection guidance.

        Args:
            meta_info: Any info extracted from query

        Returns:
            Diagnostic questions for single monitor setup
        """
        return """I can help you connect your monitor!

To recommend the right cable or adapter, I just need a few quick details:

**1. What port does your computer have?**
(USB-C, HDMI, DisplayPort, VGA, DVI, Thunderbolt, or not sure)

**2. What port does your monitor have?**
(HDMI, DisplayPort, VGA, DVI, or not sure)

**3. How far apart are they?**
(e.g., "3 feet", "across the room", "6 meters")

---

**Example response:**
"USB-C on my laptop, HDMI on my monitor, about 6 feet apart"

Once I know your ports, I'll recommend the exact cable you need!"""

    def _format_dock_selection_guidance(self, meta_info: dict) -> str:
        """
        Format docking station selection guidance.

        Args:
            meta_info: Any info extracted from query

        Returns:
            Diagnostic questions for dock selection
        """
        return """I can help you find the right docking station!

Docks vary a lot - some support multiple monitors, some charge your laptop, some have lots of extra ports. To recommend the best one for you, please tell me:

**1. What do you need the dock for?**
(e.g., "connect 2 monitors", "charge my laptop", "add more USB ports", "all of the above")

**2. What port does your laptop have?**
(USB-C, Thunderbolt 3/4, USB-A, or not sure)

**3. How many monitors do you want to connect?**
(1, 2, 3, or more)

**4. Any must-have features?**
(e.g., "needs to charge my laptop", "must have ethernet", "need SD card reader")

---

**Example response:**
"I need to connect 2 monitors and charge my MacBook Pro.
It has Thunderbolt 4 ports.
Must have ethernet and at least 60W charging."

Once I understand your needs, I'll recommend 2-3 docks that actually fit your setup!"""

    def _format_kvm_selection_guidance(self, meta_info: dict) -> str:
        """
        Format KVM switch selection guidance.

        Args:
            meta_info: Any info extracted from query (e.g., port_count)

        Returns:
            Diagnostic questions for KVM selection
        """
        port_count = meta_info.get('port_count')

        # Build personalized intro if we already know port count
        if port_count:
            intro = f"Setting up a KVM switch for {port_count} computers"
        else:
            intro = "Setting up a KVM switch"

        return f"""{intro} - I can help with that!

KVM switches let you control multiple computers from one keyboard, mouse, and monitor. To recommend the right one, I need to know:

**1. How many computers do you want to control?**
(2, 4, 8, or more?)

**2. What video output does your monitor have?**
(HDMI, DisplayPort, VGA, or DVI?)

**3. Do you need USB device switching?**
(keyboard, mouse, USB drives - most people want this)

---

**Example response:**
"2 computers, HDMI monitor, yes I need USB switching"

Or just answer each question - I'll figure it out!"""

    def format_with_context_note(
        self,
        main_response: str,
        context_type: str,
        details: Optional[str] = None
    ) -> str:
        """
        Add a context note to a response.
        
        Args:
            main_response: Main response text
            context_type: Type of context note
            details: Additional details
            
        Returns:
            Response with context note appended
            
        Example:
            >>> response = formatter.format_with_context_note(
            ...     main_response="Here are your results...",
            ...     context_type="4k",
            ...     details="Check cable certification"
            ... )
        """
        note = self.prompts.format_context_note(context_type, details)
        
        if note:
            return f"{main_response}\n\n{note}"
        
        return main_response
    
    def format_connector_info(self, connector_type: str) -> str:
        """
        Format connector information.
        
        Args:
            connector_type: Type of connector
            
        Returns:
            Connector explanation
            
        Example:
            >>> info = formatter.format_connector_info("USB-C")
        """
        from llm.prompts import get_response_templates
        templates = get_response_templates()
        return templates.format_connector_explanation(connector_type)
    
    def format_feature_info(self, feature: str) -> str:
        """
        Format feature information.
        
        Args:
            feature: Feature name
            
        Returns:
            Feature explanation
            
        Example:
            >>> info = formatter.format_feature_info("4K")
        """
        from llm.prompts import get_response_templates
        templates = get_response_templates()
        return templates.format_feature_explanation(feature)
    
    def format_multi_line(self, text: str, indent: int = 0) -> str:
        """
        Format multi-line text with indentation.
        
        Args:
            text: Text to format
            indent: Spaces to indent
            
        Returns:
            Formatted text
            
        Example:
            >>> formatted = formatter.format_multi_line(
            ...     "Line 1\\nLine 2",
            ...     indent=2
            ... )
        """
        if indent == 0:
            return text
        
        indent_str = " " * indent
        lines = text.split('\n')
        return '\n'.join(indent_str + line for line in lines)
    
    def truncate_text(self, text: str, max_length: int = 100) -> str:
        """
        Truncate text to maximum length.
        
        Args:
            text: Text to truncate
            max_length: Maximum length
            
        Returns:
            Truncated text with ellipsis if needed
            
        Example:
            >>> truncated = formatter.truncate_text(
            ...     "Very long text...",
            ...     max_length=20
            ... )
        """
        if len(text) <= max_length:
            return text
        
        return text[:max_length-3] + "..."


# Singleton instance
_response_formatter = ResponseFormatter()


def get_response_formatter() -> ResponseFormatter:
    """
    Get the response formatter instance.

    Returns:
        ResponseFormatter instance

    Example:
        >>> formatter = get_response_formatter()
        >>> response = formatter.format_greeting()
    """
    return _response_formatter


# =============================================================================
# STANDALONE FORMATTING FUNCTIONS
# =============================================================================
# These functions are used by intent handlers for specialized formatting.


def format_detailed_product_specs(prod: Product) -> str:
    """
    Format detailed product specs in a structured, scannable layout.

    Used for explicit_sku views when user asks about a specific product.
    NOT for conversational responses or product lists.

    Structure:
    - Product name header
    - Basic Specs section (essential info)
    - Technical Details section (extended specs, if available)
    - Closing question

    Args:
        prod: Product to format

    Returns:
        Formatted product specs string
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
        tech_section = ["**Technical Details**"] + tech_details
        lines.append("  \n".join(tech_section))

    # Closing
    lines.append("")  # Blank line before closing
    lines.append("Anything else you'd like to know about this product?")

    return "\n\n".join(lines)


def format_dock_specs(dock: Product) -> List[str]:
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