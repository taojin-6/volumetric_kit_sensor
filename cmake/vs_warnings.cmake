# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# Shared compiler-warning flags, applied PRIVATE so they never leak to consumers
# (a consumer's build should not inherit our warning policy). Used as
# `vs_target_warnings(<target>)` on every first-party target. CUDA sources
# (sensor_ffs's kernels) pass the host compiler's flags through nvcc, with
# nvcc's own warnings errors too.
function(vs_target_warnings target)
  target_compile_options(
    ${target}
    PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX,OBJC,OBJCXX>:-Wall;-Wextra;-Wpedantic>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>")
  if(VS_WARNINGS_AS_ERRORS)
    target_compile_options(
      ${target}
      PRIVATE
        "$<$<COMPILE_LANGUAGE:C,CXX,OBJC,OBJCXX>:-Werror>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror;--Werror=all-warnings>")
  endif()
endfunction()
