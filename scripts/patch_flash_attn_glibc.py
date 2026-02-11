#!/usr/bin/env python3
"""Patch flash_attn .so to remove GLIBC_2.32 dependency on systems with GLIBC < 2.32.

Rewrites ELF .gnu.version_r and .gnu.version sections to replace the GLIBC_2.32
version requirement with GLIBC_2.14 (which only requires the __libc_single_threaded
symbol to be provided via LD_PRELOAD of glibc_compat.so).

Usage:
    python3 scripts/patch_flash_attn_glibc.py [path_to_flash_attn_so]

If no path is given, auto-detects the installed flash_attn .so location.
"""

import struct
import subprocess
import sys


def find_flash_so():
    """Auto-detect flash_attn .so path."""
    import os
    # Check common locations: site-packages root and flash_attn subdir
    try:
        import sysconfig
        site_pkg = sysconfig.get_path("purelib")
    except Exception:
        site_pkg = None

    search_dirs = []
    try:
        import flash_attn
        search_dirs.append(os.path.dirname(flash_attn.__file__))
    except ImportError:
        pass
    if site_pkg:
        search_dirs.append(site_pkg)

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.startswith("flash_attn_2_cuda") and f.endswith(".so"):
                return os.path.join(d, f)
    return None


def parse_elf_sections(data):
    """Parse ELF64 section headers to find .gnu.version and .gnu.version_r."""
    # ELF64 header
    e_shoff = struct.unpack_from("<Q", data, 40)[0]  # section header offset
    e_shentsize = struct.unpack_from("<H", data, 58)[0]  # section header entry size
    e_shnum = struct.unpack_from("<H", data, 60)[0]  # number of section headers
    e_shstrndx = struct.unpack_from("<H", data, 62)[0]  # section name string table index

    # Get section name string table
    shstrtab_off = struct.unpack_from("<Q", data, e_shoff + e_shstrndx * e_shentsize + 24)[0]

    sections = {}
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        sh_name_idx = struct.unpack_from("<I", data, off)[0]
        sh_type = struct.unpack_from("<I", data, off + 4)[0]
        sh_offset = struct.unpack_from("<Q", data, off + 24)[0]
        sh_size = struct.unpack_from("<Q", data, off + 32)[0]
        sh_entsize = struct.unpack_from("<Q", data, off + 56)[0]

        # Read section name
        name_end = data.index(b"\x00", shstrtab_off + sh_name_idx)
        name = data[shstrtab_off + sh_name_idx : name_end].decode("ascii", errors="replace")

        sections[name] = {
            "offset": sh_offset,
            "size": sh_size,
            "type": sh_type,
            "entsize": sh_entsize,
        }

    return sections


def patch_version_r(data, sections):
    """Patch .gnu.version_r to replace GLIBC_2.32 with GLIBC_2.14.
    Returns the version index that was associated with GLIBC_2.32, or None."""
    vr = sections.get(".gnu.version_r")
    if not vr:
        print("  WARNING: .gnu.version_r section not found")
        return None

    GLIBC_232_HASH = 0x069691B2
    GLIBC_214_HASH = 0x06969194

    # Find GLIBC_2.14 string in .dynstr for the name replacement
    glibc214_off = data.find(b"GLIBC_2.14\x00")

    old_version_index = None
    offset = vr["offset"]
    end = offset + vr["size"]

    # Walk Verneed entries
    while offset < end:
        vn_version = struct.unpack_from("<H", data, offset)[0]
        vn_cnt = struct.unpack_from("<H", data, offset + 2)[0]
        vn_aux = struct.unpack_from("<I", data, offset + 8)[0]
        vn_next = struct.unpack_from("<I", data, offset + 12)[0]

        # Walk Vernaux entries for this Verneed
        aux_offset = offset + vn_aux
        for _ in range(vn_cnt):
            vna_hash = struct.unpack_from("<I", data, aux_offset)[0]
            vna_flags = struct.unpack_from("<H", data, aux_offset + 4)[0]
            vna_other = struct.unpack_from("<H", data, aux_offset + 6)[0]
            vna_name = struct.unpack_from("<I", data, aux_offset + 8)[0]
            vna_next = struct.unpack_from("<I", data, aux_offset + 12)[0]

            if vna_hash == GLIBC_232_HASH:
                print(f"  Found GLIBC_2.32 vernaux at offset 0x{aux_offset:x}, version index {vna_other}")
                old_version_index = vna_other

                # Replace hash
                struct.pack_into("<I", data, aux_offset, GLIBC_214_HASH)

                # Replace name offset to point to GLIBC_2.14 string
                if glibc214_off >= 0:
                    struct.pack_into("<I", data, aux_offset + 8, glibc214_off)
                    print(f"  Patched vernaux: hash -> GLIBC_2.14, name -> offset 0x{glibc214_off:x}")
                else:
                    print("  WARNING: GLIBC_2.14 string not found in .dynstr, hash patched but name not updated")

            if vna_next == 0:
                break
            aux_offset += vna_next

        if vn_next == 0:
            break
        offset += vn_next

    return old_version_index


def patch_version(data, sections, old_index):
    """Patch .gnu.version to remap symbols that referenced old_index.
    Maps them to version index 5 (typically GLIBC_2.2.5 - a safe baseline)."""
    vs = sections.get(".gnu.version")
    if not vs:
        print("  WARNING: .gnu.version section not found")
        return 0

    # Find a safe target version index (GLIBC_2.2.5 is usually index 2-5)
    # We'll use the patched vernaux's own index since we already changed it to GLIBC_2.14
    # So we don't need to remap - the version index still points to the same vernaux
    # entry which now says GLIBC_2.14 instead of GLIBC_2.32.
    # This means no .gnu.version patching is needed!
    count = 0
    num_entries = vs["size"] // 2
    for i in range(num_entries):
        ver = struct.unpack_from("<H", data, vs["offset"] + i * 2)[0]
        if ver == old_index:
            count += 1

    print(f"  {count} symbol(s) reference version index {old_index} (now points to GLIBC_2.14)")
    return count


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = find_flash_so()
        if not path:
            print("ERROR: Could not find flash_attn .so. Pass path as argument.")
            sys.exit(1)

    print(f"  Patching: {path}")

    # Check if GLIBC_2.32 is referenced
    try:
        result = subprocess.run(["readelf", "-V", path], capture_output=True, text=True)
        if "GLIBC_2.32" not in result.stdout:
            print("  No GLIBC_2.32 dependency found, nothing to patch.")
            return
    except FileNotFoundError:
        pass  # readelf not available, proceed anyway

    with open(path, "rb") as f:
        data = bytearray(f.read())

    # Verify ELF magic
    if data[:4] != b"\x7fELF":
        print(f"ERROR: {path} is not an ELF file")
        sys.exit(1)

    sections = parse_elf_sections(data)
    old_index = patch_version_r(data, sections)

    if old_index is None:
        print("  WARNING: GLIBC_2.32 not found in .gnu.version_r")
        return

    patch_version(data, sections, old_index)

    with open(path, "wb") as f:
        f.write(data)

    print("  Patch complete!")

    # Verify
    try:
        result = subprocess.run(["readelf", "-V", path], capture_output=True, text=True)
        if "GLIBC_2.32" in result.stdout:
            print("  WARNING: GLIBC_2.32 still appears in readelf output")
        else:
            print("  Verified: GLIBC_2.32 no longer referenced")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
