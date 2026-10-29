# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# Shared compiler-warning flags, applied PRIVATE so they never leak to consumers
# (a consumer's build should not inherit our warning policy). Used as
# `vs_target_warnings(<target>)` on every first-party target.
function(vs_target_warnings target)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  if(VS_WARNINGS_AS_ERRORS)
    target_compile_options(${target} PRIVATE -Werror)
  endif()
endfunction()
