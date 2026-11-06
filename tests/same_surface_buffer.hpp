// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// A picture as it arrives from another process: a new pixel buffer over a
// surface looked up by Mach port, the surface's id the same and its
// IOSurfaceRef another. For the tests of what knows a surface by its id.

#include <CoreFoundation/CoreFoundation.h>
#include <CoreVideo/CoreVideo.h>
#include <IOSurface/IOSurfaceRef.h>
#include <mach/mach.h>

namespace vs_test {

// A buffer over the surface @p pixels is on, with its tags; null if none comes.
// The caller releases it.
inline CVPixelBufferRef same_surface_buffer(CVPixelBufferRef pixels) {
  const mach_port_t port =
      IOSurfaceCreateMachPort(CVPixelBufferGetIOSurface(pixels));
  IOSurfaceRef surface = IOSurfaceLookupFromMachPort(port);
  mach_port_deallocate(mach_task_self(), port);
  if (surface == nullptr) return nullptr;
  CVPixelBufferRef out = nullptr;
  const CVReturn made = CVPixelBufferCreateWithIOSurface(
      kCFAllocatorDefault, surface, nullptr, &out);
  CFRelease(surface);
  if (made != kCVReturnSuccess) return nullptr;
  CVBufferPropagateAttachments(pixels, out);
  return out;
}

}  // namespace vs_test
