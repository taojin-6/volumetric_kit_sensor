# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# Sanitizer flags, applied GLOBALLY (compile + link) when VS_SANITIZE is set.
#
# Unlike vs_warnings (per-target PRIVATE), sanitizers are applied globally on
# purpose: the flags must reach the link line, and ASan's ABI is contagious --
# mixing a sanitized library with an un-sanitized executable is broken.
# Including this before add_subdirectory(third_party) also instruments any
# vendored static deps, which otherwise raise alloc/dealloc-mismatch false
# positives when linked into a sanitized test binary. Off by default, so normal
# builds are unaffected. A project that adds this package sanitizes it with its
# own global flags.
#
# Example: cmake -S sensor -B build -DCMAKE_BUILD_TYPE=Debug
# -DVS_SANITIZE="address;undefined"
if(VS_SANITIZE)
  # address;undefined -> address,undefined (the -fsanitize= argument form).
  list(JOIN VS_SANITIZE "," _vs_sanitize_list)
  set(_vs_sanitize_flags -fsanitize=${_vs_sanitize_list}
                         -fno-omit-frame-pointer -fno-sanitize-recover=all)

  add_compile_options(${_vs_sanitize_flags})
  add_link_options(${_vs_sanitize_flags})

  message(STATUS "Sanitizers enabled (VS_SANITIZE): ${VS_SANITIZE}")
endif()
