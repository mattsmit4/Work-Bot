"""
Custom Excel Loader for StarTech.com Product Data

Loads Main_Data_AI_Bot.xlsx with 4,178 products and 132 columns.
Maps to ST-Bot's Product model.

Architecture: Load ALL columns, normalize display-critical fields.
- All 132 columns are stored in metadata (for answering any question)
- Display-critical fields get normalized aliases (for clean customer display)
- Empty/NaN values are not stored (keeps memory efficient)
"""

import pandas as pd
import re
from typing import List, Optional, Any
from core.context import Product


# =============================================================================
# COLUMN NORMALIZATION MAPPINGS
# =============================================================================
# Maps Excel column names to clean, normalized names for display
# Only display-critical fields need aliases; others keep original names

COLUMN_ALIASES = {
    # Core identification
    'Product Number': 'sku',
    'Category': 'excel_category',  # Keep original, we compute normalized 'category'
    'Sub Category': 'sub_category',

    # Length fields
    'CABLELENGTH': 'cable_length_raw',

    # Connector fields
    'INTERFACEA': 'interface_a',
    'INTERFACEB': 'interface_b',

    # Network cable fields
    'RATING': 'network_rating_raw',
    'NETWORKSPEED': 'network_speed',

    # Hub-specific fields
    'NUMBERPORTS': 'hub_ports_raw',
    'USBTYPE': 'usb_type',
    'POWERADAPTER': 'power_adapter',
    'POWERDELIVERY': 'power_delivery',

    # KVM-specific fields
    'KVMPORTS': 'kvm_ports_raw',
    'KVMINTERFACE': 'kvm_interface',
    'KVMAUDIO': 'kvm_audio',
    'KVMPCVIDEO': 'kvm_video_type',

    # Mount-specific fields
    'MAXDISPLAYSIZE': 'mount_max_display_size',
    'MINDISPLAYSIZE': 'mount_min_display_size',
    'VESAPATTERN': 'mount_vesa_pattern',
    'NUMOFDISPLAY': 'mount_num_displays',
    'MOUNTOPTIONS': 'mount_options',
    'CONSTMATERIAL': 'mount_material',
    'CURVEDTV': 'mount_curved_tv',

    # Fiber cable-specific fields
    'FIBERTYPE': 'fiber_type_raw',
    'FIBERDUPLEX': 'fiber_duplex_raw',
    'WAVELENGTH': 'fiber_wavelength_raw',

    # Storage enclosure-specific fields
    'ENCLOSURETYPE': 'enclosure_type_raw',
    'DRIVECONNECTOR': 'drive_connector_raw',
    'DRIVESIZE': 'drive_size_raw',
    'NUMHARDDRIVE': 'num_drives_raw',
    'IOINTERFACE': 'io_interface_raw',
    'HARDDRIVECOM': 'drive_compatibility_raw',

    # Extended specs (commonly asked about)
    'WIREGUAGE': 'wire_gauge',  # Note: Excel has typo "GUAGE"
    'CONNPLATING': 'connector_plating',
    'MAXRESOLUTION': 'max_resolution',
    'MAXDVIRESOLUTION': 'max_dvi_resolution',
    'SUPRESOLUTIONS': 'supported_resolutions',
    'JACKETTYPE': 'jacket_type',
    'CONNSTYLE': 'connector_style',
    'COLOR': 'color',
    'SHIELDTYPE': 'shield_type',
    'CONDUCTORTYPE': 'conductor_type',
    'FIRERATING': 'fire_rating',
    'WARRANTY': 'warranty',
    'STANDARDS': 'standards',

    # Physical dimensions
    'Product Height': 'product_height',
    'Product Length': 'product_length',
    'Product Width': 'product_width',
    'Weight of Product': 'product_weight',
    'Package Height': 'package_height',
    'Package Length': 'package_length',
    'Package Width': 'package_width',
    'Shipping (Package) Weight': 'package_weight',

    # Environmental specs
    'OPERATINGTEMP': 'operating_temp',
    'STORETEMP': 'storage_temp',
    'HUMIDITY': 'humidity',
}


# =============================================================================
# PARSING HELPERS
# =============================================================================

def parse_cable_length(length_str) -> tuple:
    """
    Parse cable length from various formats.
    Returns tuple: (feet, meters) for display like "6.0 ft [1.8 m]"

    StarTech Excel stores lengths in millimeters as bare numbers.

    Examples:
    - "6ft" -> (6.0, 1.8)
    - "10 feet" -> (10.0, 3.0)
    - "1.8m" -> (5.9, 1.8)
    - "1828.8" (mm) -> (6.0, 1.8)
    - "152.4" (mm) -> (0.5, 0.15)
    """
    if pd.isna(length_str):
        return None, None

    try:
        length_str = str(length_str).lower().strip()
        length_str = length_str.replace('approx', '').replace('approximately', '').strip()

        # Check for millimeters explicitly
        if 'mm' in length_str:
            match = re.search(r'([\d.]+)\s*mm', length_str)
            if match:
                mm = float(match.group(1))
                feet = round(mm / 304.8, 1)
                meters = round(mm / 1000.0, 1)
                return feet, meters

        # Check for meters (convert to feet)
        if 'm' in length_str:
            match = re.search(r'([\d.]+)\s*m', length_str)
            if match:
                meters = float(match.group(1))
                feet = round(meters * 3.28084, 1)
                return feet, meters

        # Check for inches (convert to feet and meters)
        if 'in' in length_str or '"' in length_str:
            match = re.search(r'([\d.]+)\s*(?:in|")', length_str)
            if match:
                inches = float(match.group(1))
                feet = round(inches / 12.0, 1)
                meters = round(inches * 0.0254, 1)
                return feet, meters

        # Check for feet explicitly
        if 'ft' in length_str or 'foot' in length_str or 'feet' in length_str or "'" in length_str:
            match = re.search(r'([\d.]+)', length_str)
            if match:
                feet = float(match.group(1))
                meters = round(feet / 3.28084, 1)
                return feet, meters

        # Bare number - assume millimeters (StarTech.com Excel format)
        match = re.search(r'([\d.]+)', length_str)
        if match:
            mm = float(match.group(1))
            if mm > 10:  # Likely in mm
                feet = round(mm / 304.8, 1)
                meters = round(mm / 1000.0, 1)
                return feet, meters
            else:  # Small number, might already be in feet
                feet = mm
                meters = round(feet / 3.28084, 1)
                return feet, meters

        return None, None

    except Exception:
        return None, None


def extract_features(row: pd.Series) -> List[str]:
    """
    Extract searchable/filterable features from product data.
    These are used for search matching and display.
    """
    features = []

    # Resolution support
    max_res = row.get('MAXRESOLUTION') or row.get('MAXDVIRESOLUTION')
    sup_res = row.get('SUPRESOLUTIONS')
    res_str = ''
    if pd.notna(max_res):
        res_str += str(max_res).upper()
    if pd.notna(sup_res):
        res_str += ' ' + str(sup_res).upper()

    if res_str:
        if '3840' in res_str or '4K' in res_str or '2160' in res_str or 'ULTRA HD' in res_str:
            features.append('4K')
        elif '7680' in res_str or '8K' in res_str or '4320' in res_str:
            features.append('8K')
        elif '2560' in res_str or '1440' in res_str:
            features.append('1440p')
        elif '1920' in res_str or '1080' in res_str:
            features.append('1080p')

    # Power Delivery
    if row.get('POWERDELIVERY') == 'Yes' or row.get('DOCKFASTCHARGE') == 'Yes':
        features.append('Power Delivery')

    # 4K Support (for docks)
    if row.get('DOCK4KSUPPORT') == 'Yes':
        features.append('4K')

    # HDR
    if pd.notna(sup_res) and 'HDR' in str(sup_res).upper():
        features.append('HDR')

    # PoE (Power over Ethernet)
    if row.get('POE_YN') == 'Yes':
        features.append('PoE')

    # Network Speed
    network_speed = row.get('NETWORKSPEED')
    if pd.notna(network_speed):
        speed_str = str(network_speed)
        if 'Gigabit' in speed_str or '1000' in speed_str or '1Gbps' in speed_str:
            features.append('Gigabit')
        elif '10G' in speed_str or '10 G' in speed_str:
            features.append('10 Gigabit')

    # Thunderbolt
    if pd.notna(row.get('HOSTCONNECTOR')):
        host = str(row.get('HOSTCONNECTOR')).lower()
        if 'thunderbolt' in host:
            features.append('Thunderbolt')

    # USB-C
    if pd.notna(row.get('USBTYPE')):
        usb = str(row.get('USBTYPE')).lower()
        if 'usb-c' in usb or 'usb c' in usb or 'type-c' in usb or 'type c' in usb:
            features.append('USB-C')

    # Active cables
    if pd.notna(row.get('POWERADAPTER')):
        if row.get('POWERADAPTER') == 'Yes':
            features.append('Active')

    # Shielded
    if pd.notna(row.get('SHIELDTYPE')):
        shield = str(row.get('SHIELDTYPE'))
        if shield and shield != 'nan' and shield != 'Unshielded':
            features.append('Shielded')

    # HDCP support
    if pd.notna(row.get('STANDARDS')):
        standards = str(row.get('STANDARDS')).upper()
        if 'HDCP' in standards:
            features.append('HDCP')

    # Audio support
    if row.get('KVMAUDIO') == 'Yes':
        features.append('Audio')
    if pd.notna(row.get('AUDIOPORTS')):
        audio_ports = str(row.get('AUDIOPORTS'))
        if audio_ports and audio_ports != 'nan' and audio_ports != '0':
            features.append('Audio')

    return list(set(features))


def extract_connectors(row: pd.Series) -> Optional[list]:
    """
    Extract connector information from INTERFACEA and INTERFACEB.

    Falls back to HOSTCONNECTOR for multiport adapters and docks which
    store their input connector in a different column.
    """
    interface_a = row.get('INTERFACEA')
    interface_b = row.get('INTERFACEB')

    # Fallback: HOSTCONNECTOR is used by multiport adapters and docks
    # for their input/host connector (e.g., "1 x USB 3.2 Type-C (24 pin)")
    host_connector = row.get('HOSTCONNECTOR')

    # If both INTERFACE columns are empty, try HOSTCONNECTOR
    if pd.isna(interface_a) and pd.isna(interface_b):
        if pd.notna(host_connector):
            host_str = str(host_connector).strip()
            if host_str and host_str.lower() != 'nan':
                # HOSTCONNECTOR is the input/source connector
                # Multiport adapters have multiple outputs, so we only return the source
                return [host_str]
        return None

    connectors = []
    if pd.notna(interface_a):
        source = str(interface_a).strip()
        if source and source != 'nan':
            connectors.append(source)

    if pd.notna(interface_b):
        target = str(interface_b).strip()
        if target and target != 'nan':
            connectors.append(target)

    if len(connectors) == 0:
        return None
    elif len(connectors) == 1:
        connectors.append(connectors[0])

    return connectors


def extract_network_rating(row: pd.Series) -> Optional[dict]:
    """
    Extract network cable rating (Cat5e, Cat6, Cat6a, etc.) from RATING column.
    Essential for network cables - determines compatibility.
    """
    rating_raw = row.get('RATING')
    if pd.isna(rating_raw):
        return None

    rating_str = str(rating_raw).strip()
    if not rating_str or rating_str == 'nan':
        return None

    result = {
        'rating': None,
        'rating_full': rating_str,
        'max_speed': None
    }

    rating_upper = rating_str.upper()

    # Extract short rating
    if 'CAT6A' in rating_upper:
        result['rating'] = 'Cat6a'
        result['max_speed'] = '10 Gigabit'
    elif 'CAT6' in rating_upper:
        result['rating'] = 'Cat6'
        result['max_speed'] = 'Gigabit'
    elif 'CAT5E' in rating_upper:
        result['rating'] = 'Cat5e'
        result['max_speed'] = 'Gigabit'
    elif 'CAT5' in rating_upper:
        result['rating'] = 'Cat5'
        result['max_speed'] = '100 Mbps'
    elif 'CAT7' in rating_upper:
        result['rating'] = 'Cat7'
        result['max_speed'] = '10 Gigabit'

    return result if result['rating'] else None


def determine_category(row: pd.Series) -> str:
    """
    Determine normalized product category from Category and Sub Category columns.
    Maps to standard categories: cable, adapter, dock, hub, switch, etc.
    """
    category = row.get('Category')
    sub_category = row.get('Sub Category')

    cat_str = ''
    if pd.notna(category):
        cat_str += str(category).lower()
    if pd.notna(sub_category):
        cat_str += ' ' + str(sub_category).lower()

    # Check connectors for special case handling
    interface_a = str(row.get('INTERFACEA', '')).lower()
    interface_b = str(row.get('INTERFACEB', '')).lower()

    # USB-C to video output are cables, not adapters
    if ('usb' in interface_a and 'type-c' in interface_a) or ('usb-c' in interface_a):
        if any(video in interface_b for video in ['hdmi', 'displayport', 'display port']):
            return 'cable'

    # Map to standard categories
    # Order matters: more specific categories first

    # Fiber cables (before generic 'cable' check)
    if 'fiber' in cat_str:
        return 'fiber_cable'

    # Storage/drive enclosures (before generic 'enclosure' check)
    if 'external drive' in cat_str or 'drive enclosure' in cat_str:
        return 'storage_enclosure'
    if 'data storage' in cat_str and ('enclosure' in cat_str or 'drive' in cat_str):
        return 'storage_enclosure'

    if 'cable' in cat_str:
        return 'cable'
    # Racks must be checked BEFORE enclosures ("Racks and Enclosures" contains both)
    elif 'rack' in cat_str:
        return 'rack'
    # Computer cards must be checked BEFORE adapters ("Computer Cards and Adapters" contains both)
    elif 'computer card' in cat_str or ('card' in cat_str and 'adapter' in cat_str):
        return 'computer_card'
    # Multiport adapters must be checked BEFORE generic adapters
    elif 'multiport' in cat_str:
        return 'multiport_adapter'
    elif 'adapter' in cat_str or 'converter' in cat_str:
        return 'adapter'
    elif 'dock' in cat_str or 'docking' in cat_str:
        return 'dock'
    elif 'hub' in cat_str:
        return 'hub'
    elif 'ethernet switch' in cat_str:
        # Ethernet switches (networking devices) - distinct from KVM/video switches
        return 'ethernet_switch'
    elif 'kvm' in cat_str:
        # KVM switches
        return 'kvm_switch'
    elif 'video switch' in cat_str:
        # Video/HDMI/DisplayPort switches
        return 'video_switch'
    elif 'switch' in cat_str:
        # Generic switch fallback
        return 'switch'
    elif 'enclosure' in cat_str:
        return 'enclosure'
    elif 'mount' in cat_str:
        return 'mount'
    elif 'privacy' in cat_str or 'screen filter' in cat_str:
        return 'privacy_screen'
    elif 'video splitter' in cat_str or ('splitter' in cat_str and 'video' in cat_str):
        return 'video_splitter'
    elif 'power' in cat_str:
        return 'power'
    elif 'network' in cat_str:
        return 'network'
    else:
        return 'other'


# =============================================================================
# DERIVED FIELD COMPUTATION
# =============================================================================

def compute_derived_fields(row: pd.Series, metadata: dict) -> None:
    """
    Compute derived/normalized fields from raw data.
    Modifies metadata in place.
    """
    # --- Length fields ---
    length_raw = row.get('CABLELENGTH')
    length_ft, length_m = parse_cable_length(length_raw)

    if length_ft is not None:
        metadata['length'] = length_ft
        metadata['length_ft'] = length_ft
        metadata['length_m'] = length_m
        metadata['length_unit'] = 'ft'

        # Format length for display
        if length_m and length_m >= 1:
            metadata['length_display'] = f"{length_ft} ft [{length_m} m]"
        else:
            metadata['length_display'] = f"{length_ft} ft"

    # --- Category ---
    metadata['category'] = determine_category(row)

    # --- Connectors ---
    connectors = extract_connectors(row)
    if connectors:
        metadata['connectors'] = connectors

    # --- Features ---
    features = extract_features(row)
    metadata['features'] = features

    # --- Network rating ---
    network_info = extract_network_rating(row)
    if network_info:
        metadata['network_rating'] = network_info['rating']
        metadata['network_rating_full'] = network_info['rating_full']
        metadata['network_max_speed'] = network_info['max_speed']

    # --- Hub-specific derived fields ---
    num_ports = row.get('NUMBERPORTS')
    if pd.notna(num_ports):
        try:
            metadata['hub_ports'] = int(float(num_ports))
        except (ValueError, TypeError):
            pass

    usb_type = row.get('USBTYPE')
    if pd.notna(usb_type):
        usb_str = str(usb_type).strip()
        if usb_str and usb_str != 'nan':
            metadata['hub_usb_type'] = usb_str
            # Simplified version for display
            usb_lower = usb_str.lower()
            if 'usb 3.2 gen 2' in usb_lower or '10 gbit' in usb_lower:
                metadata['hub_usb_version'] = 'USB 3.2 Gen 2 (10Gbps)'
            elif 'usb 3.2 gen 1' in usb_lower or 'usb 3.1' in usb_lower or 'usb 3.0' in usb_lower or '5 gbit' in usb_lower:
                metadata['hub_usb_version'] = 'USB 3.0 (5Gbps)'
            elif 'usb 2.0' in usb_lower:
                metadata['hub_usb_version'] = 'USB 2.0'
            else:
                metadata['hub_usb_version'] = usb_str

    power_adapter = row.get('POWERADAPTER')
    if pd.notna(power_adapter):
        power_str = str(power_adapter).strip()
        if power_str and power_str != 'nan':
            metadata['hub_power_type'] = power_str
            power_lower = power_str.lower()
            if 'ac adapter' in power_lower or 'included' in power_lower:
                metadata['hub_powered'] = True
            elif 'usb-powered' in power_lower or 'bus-powered' in power_lower or 'bus powered' in power_lower:
                metadata['hub_powered'] = False

    power_delivery = row.get('POWERDELIVERY')
    if pd.notna(power_delivery):
        pd_str = str(power_delivery).strip()
        if pd_str and pd_str != 'nan' and pd_str.lower() != 'no':
            metadata['hub_power_delivery'] = pd_str

    # --- KVM-specific derived fields ---
    kvm_ports = row.get('KVMPORTS')
    if pd.notna(kvm_ports):
        try:
            metadata['kvm_ports'] = int(float(kvm_ports))
        except (ValueError, TypeError):
            pass

    kvm_audio = row.get('KVMAUDIO')
    if pd.notna(kvm_audio):
        metadata['kvm_audio'] = str(kvm_audio).strip() == 'Yes'

    # --- Mount-specific derived fields ---
    # Display size range - values are strings like "100"", "30in", "12.9""
    max_display = row.get('MAXDISPLAYSIZE')
    min_display = row.get('MINDISPLAYSIZE')
    if pd.notna(max_display):
        try:
            # Extract numeric value from strings like "100"", "30in", "12.9""
            max_str = str(max_display).strip()
            max_match = re.search(r'([\d.]+)', max_str)
            if max_match:
                max_size = float(max_match.group(1))
                metadata['mount_max_display'] = max_size

                # Parse min display size similarly
                if pd.notna(min_display):
                    min_str = str(min_display).strip()
                    min_match = re.search(r'([\d.]+)', min_str)
                    if min_match:
                        min_size = float(min_match.group(1))
                        metadata['mount_min_display'] = min_size
                        metadata['mount_display_range'] = f'{int(min_size)}-{int(max_size)}"'
                    else:
                        metadata['mount_display_range'] = f'Up to {int(max_size)}"'
                else:
                    metadata['mount_display_range'] = f'Up to {int(max_size)}"'
        except (ValueError, TypeError):
            pass

    # VESA pattern - simplify for display (e.g., "100x100 mm, 75x75 mm" -> "75x75, 100x100")
    vesa = row.get('VESAPATTERN')
    if pd.notna(vesa):
        vesa_str = str(vesa).strip()
        if vesa_str and vesa_str != 'nan':
            # Extract unique VESA patterns and simplify
            vesa_patterns = set()
            for pattern in re.findall(r'(\d+x\d+)', vesa_str):
                vesa_patterns.add(pattern)
            if vesa_patterns:
                # Sort by size (smaller first)
                sorted_patterns = sorted(vesa_patterns, key=lambda x: int(x.split('x')[0]))
                metadata['mount_vesa'] = ', '.join(sorted_patterns)

    # Number of displays
    num_displays = row.get('NUMOFDISPLAY')
    if pd.notna(num_displays):
        try:
            metadata['mount_num_displays'] = int(float(num_displays))
        except (ValueError, TypeError):
            pass

    # Mount type (wall, desk, pole, etc.) - decode HTML entities
    mount_options = row.get('MOUNTOPTIONS')
    if pd.notna(mount_options):
        mount_str = str(mount_options).strip()
        if mount_str and mount_str != 'nan':
            # Decode HTML entities like &amp; -> &
            mount_str = mount_str.replace('&amp;', '&')
            metadata['mount_type'] = mount_str

    # Curved TV support
    curved_tv = row.get('CURVEDTV')
    if pd.notna(curved_tv) and str(curved_tv).strip() == 'Yes':
        metadata['mount_curved_support'] = True

    # Material
    material = row.get('CONSTMATERIAL')
    if pd.notna(material):
        mat_str = str(material).strip()
        if mat_str and mat_str != 'nan':
            metadata['mount_material'] = mat_str

    # --- Wire gauge normalization ---
    wire_gauge = row.get('WIREGUAGE')
    if pd.notna(wire_gauge):
        gauge_str = str(wire_gauge).strip()
        if gauge_str and gauge_str != 'nan':
            if 'AWG' not in gauge_str.upper():
                gauge_str = f"{gauge_str} AWG"
            metadata['wire_gauge'] = gauge_str

    # --- Fiber cable-specific derived fields ---
    fiber_type = row.get('FIBERTYPE')
    if pd.notna(fiber_type):
        ft_str = str(fiber_type).strip()
        if ft_str and ft_str != 'nan':
            # Normalize: "Multi Mode" -> "Multimode", "Single Mode" -> "Single-mode"
            if 'multi' in ft_str.lower():
                metadata['fiber_type'] = 'Multimode'
            elif 'single' in ft_str.lower():
                metadata['fiber_type'] = 'Single-mode'
            else:
                metadata['fiber_type'] = ft_str

    wavelength = row.get('WAVELENGTH')
    if pd.notna(wavelength):
        wl_str = str(wavelength).strip()
        if wl_str and wl_str != 'nan':
            # Extract primary wavelength (first one if multiple)
            # e.g., "1300nm, 850nm" -> "850nm"
            wl_parts = wl_str.split(',')
            if wl_parts:
                # Get the shortest wavelength (usually primary)
                wavelengths = []
                for part in wl_parts:
                    match = re.search(r'(\d+)\s*nm', part)
                    if match:
                        wavelengths.append(int(match.group(1)))
                if wavelengths:
                    primary_wl = min(wavelengths)
                    metadata['fiber_wavelength'] = f"{primary_wl}nm"

    # Extract fiber connector type from INTERFACEA (e.g., "1 x Fiber Optic LC Duplex" -> "LC")
    interface_a = row.get('INTERFACEA')
    if pd.notna(interface_a):
        ia_str = str(interface_a).strip().lower()
        if 'fiber' in ia_str or 'mpo' in ia_str or 'mtp' in ia_str:
            # This is a fiber cable - extract connector type
            if 'mpo' in ia_str or 'mtp' in ia_str:
                metadata['fiber_connector'] = 'MPO/MTP'
            elif 'lc' in ia_str:
                metadata['fiber_connector'] = 'LC'
            elif 'sc' in ia_str:
                metadata['fiber_connector'] = 'SC'
            elif 'st' in ia_str:
                metadata['fiber_connector'] = 'ST'

            # Check for duplex/simplex
            if 'duplex' in ia_str:
                metadata['fiber_duplex'] = 'Duplex'
            elif 'simplex' in ia_str:
                metadata['fiber_duplex'] = 'Simplex'

    # --- Storage enclosure-specific derived fields ---
    drive_size = row.get('DRIVESIZE')
    if pd.notna(drive_size):
        ds_str = str(drive_size).strip()
        if ds_str and ds_str != 'nan':
            # Normalize drive size format
            ds_str = ds_str.replace('&amp;', '&')
            # Clean up common formats
            if 'm.2' in ds_str.lower() or 'nvme' in ds_str.lower():
                if 'nvme' in ds_str.lower():
                    metadata['drive_size'] = 'M.2 NVMe'
                elif 'sata' in ds_str.lower():
                    metadata['drive_size'] = 'M.2 SATA'
                else:
                    metadata['drive_size'] = 'M.2'
            elif '2.5' in ds_str:
                if '3.5' in ds_str:
                    metadata['drive_size'] = '2.5"/3.5"'
                else:
                    metadata['drive_size'] = '2.5"'
            elif '3.5' in ds_str:
                metadata['drive_size'] = '3.5"'
            elif 'msata' in ds_str.lower():
                metadata['drive_size'] = 'mSATA'
            else:
                metadata['drive_size'] = ds_str

    num_drives = row.get('NUMHARDDRIVE')
    if pd.notna(num_drives):
        try:
            metadata['num_drives'] = int(float(num_drives))
        except (ValueError, TypeError):
            pass

    io_interface = row.get('IOINTERFACE')
    if pd.notna(io_interface):
        io_str = str(io_interface).strip()
        if io_str and io_str != 'nan':
            io_str = io_str.replace('&amp;', '&')
            # Simplify common interface names
            io_lower = io_str.lower()
            if 'thunderbolt 3' in io_lower or 'thunderbolt3' in io_lower:
                metadata['storage_interface'] = 'Thunderbolt 3'
            elif 'thunderbolt 4' in io_lower:
                metadata['storage_interface'] = 'Thunderbolt 4'
            elif 'usb 3.2 gen 2' in io_lower or '10gbps' in io_lower or '10 gbit' in io_lower:
                metadata['storage_interface'] = 'USB 3.2 Gen 2 (10Gbps)'
            elif 'usb 3.2 gen 1' in io_lower or 'usb 3.0' in io_lower or '5gbps' in io_lower:
                metadata['storage_interface'] = 'USB 3.0 (5Gbps)'
            elif 'usb 2.0' in io_lower:
                metadata['storage_interface'] = 'USB 2.0'
            elif 'esata' in io_lower:
                if 'usb' in io_lower:
                    metadata['storage_interface'] = 'USB 3.0 & eSATA'
                else:
                    metadata['storage_interface'] = 'eSATA'
            elif 'sata' in io_lower:
                metadata['storage_interface'] = 'SATA'
            else:
                metadata['storage_interface'] = io_str

    enclosure_type = row.get('ENCLOSURETYPE')
    if pd.notna(enclosure_type):
        et_str = str(enclosure_type).strip()
        if et_str and et_str != 'nan':
            metadata['enclosure_material'] = et_str

    # Check for tool-free design (often in product name or features)
    name = metadata.get('name', '')
    if 'tool-free' in name.lower() or 'toolless' in name.lower() or 'tool free' in name.lower():
        metadata['tool_free'] = True


# =============================================================================
# MAIN LOADER
# =============================================================================

def load_startech_products(excel_path: str, apply_filter: bool = False) -> List[Product]:
    """
    Load products from StarTech.com Excel file.

    Loads ALL columns from Excel, normalizes display-critical fields.

    Args:
        excel_path: Path to Main_Data_AI_Bot.xlsx
        apply_filter: If True, only load products marked "Keep"

    Returns:
        List of Product objects with complete metadata
    """
    print(f"Loading products from: {excel_path}")

    # Read Excel file
    df = pd.read_excel(excel_path)

    print(f"Total rows in Excel: {len(df)}")
    print(f"Total columns in Excel: {len(df.columns)}")

    # Optionally filter products
    if apply_filter and 'Keep & Filter' in df.columns:
        df = df[df['Keep & Filter'].notna() & df['Keep & Filter'].str.contains('Keep', na=False)]
        print(f"Products after filtering: {len(df)}")
    else:
        print(f"Loading all {len(df)} products (no filtering)")

    products = []
    skipped = 0
    errors = []

    for idx, row in df.iterrows():
        try:
            # Get Product Number (SKU)
            sku = row.get('Product Number')
            if pd.isna(sku):
                skipped += 1
                if len(errors) < 10:
                    errors.append(f"Row {idx}: No SKU")
                continue

            sku = str(sku).strip()

            # --- STEP 1: Load ALL columns into metadata ---
            metadata = {}
            for col in df.columns:
                val = row[col]
                if pd.notna(val):
                    # Convert numpy types to Python types
                    if hasattr(val, 'item'):
                        val = val.item()

                    # Apply column alias if defined
                    key = COLUMN_ALIASES.get(col, col)
                    metadata[key] = val

            # --- STEP 2: Compute derived/normalized fields ---
            compute_derived_fields(row, metadata)

            # --- STEP 3: Build product name ---
            sub_cat = row.get('Sub Category')
            if pd.notna(sub_cat):
                name = f"{sku} - {sub_cat}"
            else:
                name = sku
            metadata['name'] = name
            metadata['sku'] = sku

            # --- STEP 4: Build content string for search ---
            content_parts = [name]
            category = metadata.get('category')
            if category:
                content_parts.append(f"Category: {category}")
            length_display = metadata.get('length_display')
            if length_display:
                content_parts.append(f"Length: {length_display}")
            connectors = metadata.get('connectors')
            if connectors and len(connectors) >= 2:
                conn_str = f"{connectors[0]} to {connectors[1]}"
                content_parts.append(f"Connectors: {conn_str}")
            features = metadata.get('features', [])
            if features:
                content_parts.append(f"Features: {', '.join(features)}")

            content = " | ".join(content_parts)

            # --- STEP 5: Create Product object ---
            product = Product(
                product_number=sku,
                content=content,
                metadata=metadata,
                score=1.0
            )

            products.append(product)

        except Exception as e:
            skipped += 1
            if len(errors) < 10:
                errors.append(f"Row {idx}: {type(e).__name__}: {str(e)}")
            continue

    print(f"Successfully loaded {len(products)} products")
    if skipped > 0:
        print(f"Skipped {skipped} products due to errors")
        if errors:
            for err in errors[:5]:
                print(f"  - {err}")

    return products


def get_product_statistics(products: List[Product]) -> dict:
    """
    Get statistics about loaded products.

    Returns dict with:
    - total: Total product count
    - by_category: Count by category
    - with_length: Count with length data
    - with_connectors: Count with connector data
    - avg_metadata_fields: Average metadata fields per product
    """
    stats = {
        'total': len(products),
        'by_category': {},
        'with_length': 0,
        'with_connectors': 0,
        'with_features': 0,
        'avg_metadata_fields': 0,
    }

    total_fields = 0

    for product in products:
        # Category counts
        category = product.metadata.get('category', 'other')
        stats['by_category'][category] = stats['by_category'].get(category, 0) + 1

        # Length count
        if product.metadata.get('length_ft'):
            stats['with_length'] += 1

        # Connector count
        if product.metadata.get('connectors'):
            stats['with_connectors'] += 1

        # Features count
        if product.metadata.get('features'):
            stats['with_features'] += 1

        # Metadata field count
        total_fields += len(product.metadata)

    if products:
        stats['avg_metadata_fields'] = round(total_fields / len(products), 1)

    return stats
