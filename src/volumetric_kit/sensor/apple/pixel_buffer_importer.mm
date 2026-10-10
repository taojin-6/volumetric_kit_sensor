// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Before anything includes Vulkan: VK_EXT_metal_objects' structures.
#define VK_USE_PLATFORM_METAL_EXT

#include "volumetric_kit/recon/sensor/apple/pixel_buffer_importer.hpp"

#include <IOSurface/IOSurfaceRef.h>
#include <Metal/Metal.h>

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/external_memory.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"

namespace volumetric_kit::recon::sensor {
namespace {

// A surface no picture has arrived on in this many has left its producer's
// pool, as a decoder's after a reset, and is let go; frames still holding it
// keep it.
constexpr std::uint64_t kStale = 64;

// A surface's two planes as images, made once: a producer cycles its
// pictures through a few surfaces (VideoToolbox five for the lab's 4K clip).
struct Surface {
  explicit Surface(IOSurfaceRef s) : surface(s) { CFRetain(s); }
  ~Surface() { CFRelease(surface); }
  Surface(const Surface&) = delete;
  Surface& operator=(const Surface&) = delete;

  IOSurfaceRef surface;
  core::Image luma;
  core::Image chroma;
  std::uint64_t seen = 0;  // the last picture that arrived on it
};

// What a picture's images hold: their surface, and the pixel buffer, so its
// producer does not reuse the surface while anything reads it.
struct Held {
  Held(std::shared_ptr<const Surface> s, CVPixelBufferRef p)
      : surface(std::move(s)), pixels(p) {
    CVPixelBufferRetain(p);
  }
  ~Held() { CVPixelBufferRelease(pixels); }
  Held(const Held&) = delete;
  Held& operator=(const Held&) = delete;

  std::shared_ptr<const Surface> surface;
  CVPixelBufferRef pixels;
};

// @p image bound to memory of its own, or null. MoltenVK binds a texture's
// own storage, so the memory is a formality: device-local and not
// host-visible, which it would back with a buffer.
VkDeviceMemory bind_memory(const core::Device& device, VkImage image) {
  const VkDevice dev = device.handle();
  VkMemoryRequirements needs{};
  vkGetImageMemoryRequirements(dev, image, &needs);
  const std::optional<std::uint32_t> type = core::find_memory_type(
      device, needs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  if (!type) return VK_NULL_HANDLE;
  VkMemoryAllocateInfo alloc{};
  alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc.allocationSize = needs.size;
  alloc.memoryTypeIndex = *type;
  VkDeviceMemory out = VK_NULL_HANDLE;
  if (vkAllocateMemory(dev, &alloc, nullptr, &out) != VK_SUCCESS) {
    return VK_NULL_HANDLE;
  }
  if (vkBindImageMemory(dev, image, out, 0) != VK_SUCCESS) {
    vkFreeMemory(dev, out, nullptr);
    return VK_NULL_HANDLE;
  }
  return out;
}

}  // namespace

struct PixelBufferImporter::Impl {
  const core::Device* device = nullptr;
  std::string who;
  id<MTLDevice> metal = nil;  // the one MoltenVK runs the device on
  std::vector<std::shared_ptr<Surface>> surfaces;
  std::uint64_t pictures = 0;

  core::Status error(const std::string& what) const {
    return core::Status::out_of_memory(who + ": " + what);
  }

  // Plane @p plane of @p surface as an image of @p format, still UNDEFINED.
  core::Result<core::Image> plane_image(CVPixelBufferRef pixels,
                                        IOSurfaceRef surface, int plane,
                                        VkFormat format,
                                        MTLPixelFormat metal_format);
  // @p surface's two planes as images in GENERAL.
  core::Result<std::shared_ptr<Surface>> import_surface(CVPixelBufferRef pixels,
                                                        IOSurfaceRef surface);
};

core::Result<core::Image> PixelBufferImporter::Impl::plane_image(
    CVPixelBufferRef pixels, IOSurfaceRef surface, int plane, VkFormat format,
    MTLPixelFormat metal_format) {
  const auto width =
      static_cast<std::uint32_t>(CVPixelBufferGetWidthOfPlane(pixels, plane));
  const auto height =
      static_cast<std::uint32_t>(CVPixelBufferGetHeightOfPlane(pixels, plane));
  MTLTextureDescriptor* desc =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:metal_format
                                                         width:width
                                                        height:height
                                                     mipmapped:NO];
  desc.usage = MTLTextureUsageShaderRead;
  desc.storageMode = MTLStorageModeShared;  // Apple silicon: unified memory
  id<MTLTexture> texture =
      [metal newTextureWithDescriptor:desc
                            iosurface:surface
                                plane:static_cast<NSUInteger>(plane)];
  if (texture == nil) return error("wrapping a picture plane in a texture");

  VkImportMetalTextureInfoEXT import{};
  import.sType = VK_STRUCTURE_TYPE_IMPORT_METAL_TEXTURE_INFO_EXT;
  import.plane = VK_IMAGE_ASPECT_PLANE_0_BIT;
  import.mtlTexture = texture;
  VkImageCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  info.pNext = &import;
  info.imageType = VK_IMAGE_TYPE_2D;
  info.format = format;
  info.extent = {width, height, 1};
  info.mipLevels = 1;
  info.arrayLayers = 1;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  const VkDevice dev = device->handle();
  VkImage image = VK_NULL_HANDLE;
  if (vkCreateImage(dev, &info, nullptr, &image) != VK_SUCCESS) {
    return error("importing a picture plane");
  }
  const VkDeviceMemory memory_handle = bind_memory(*device, image);
  if (memory_handle == VK_NULL_HANDLE) {
    vkDestroyImage(dev, image, nullptr);
    return error("binding a picture plane");
  }
  core::ImageInfo adopted;
  adopted.image = image;
  adopted.format = format;
  adopted.extent = info.extent;
  adopted.usage = info.usage;
  adopted.layout = VK_IMAGE_LAYOUT_GENERAL;
  return core::Image(adopted, [dev, image, memory_handle, texture] {
    vkDestroyImage(dev, image, nullptr);
    vkFreeMemory(dev, memory_handle, nullptr);
    static_cast<void>(texture);  // released with the capture
  });
}

core::Result<std::shared_ptr<Surface>>
PixelBufferImporter::Impl::import_surface(CVPixelBufferRef pixels,
                                          IOSurfaceRef surface) {
  auto out = std::make_shared<Surface>(surface);
  VKC_ASSIGN(out->luma, plane_image(pixels, surface, 0, VK_FORMAT_R8_UNORM,
                                    MTLPixelFormatR8Unorm));
  VKC_ASSIGN(out->chroma, plane_image(pixels, surface, 1, VK_FORMAT_R8G8_UNORM,
                                      MTLPixelFormatRG8Unorm));
  // Into GENERAL, once. Vulkan may discard contents on a transition from
  // UNDEFINED, but Metal has no layouts, so MoltenVK keeps them.
  VkImageMemoryBarrier b[2]{};
  const VkImage images[2] = {out->luma.handle(), out->chroma.handle()};
  for (int i = 0; i < 2; ++i) {
    b[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b[i].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b[i].image = images[i];
    b[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  }
  // The surface rides along: should the submit fail with the barrier still on
  // the device, the device keeps the images it names until it has run.
  VKC_TRY(device->submit_single_time(
      [&](VkCommandBuffer cmd) {
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                             nullptr, 2, b);
      },
      out));
  return out;
}

core::Result<PixelBufferImporter> PixelBufferImporter::create(
    const core::Device& device, std::string who) {
  if (!device.imports_metal_textures()) {
    return core::Status::unsupported(who +
                                     ": the device imports no Metal textures "
                                     "(VK_EXT_metal_objects)");
  }
  // The MTLDevice, off the texture of a probe image: exporting the device
  // itself needs the instance to have asked for it when it was made, which
  // an embedder's may not have, and an image asks for its own.
  const auto export_objects = reinterpret_cast<PFN_vkExportMetalObjectsEXT>(
      vkGetDeviceProcAddr(device.handle(), "vkExportMetalObjectsEXT"));
  VkExportMetalObjectCreateInfoEXT exported{};
  exported.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECT_CREATE_INFO_EXT;
  exported.exportObjectType = VK_EXPORT_METAL_OBJECT_TYPE_METAL_TEXTURE_BIT_EXT;
  VkImageCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  info.pNext = &exported;
  info.imageType = VK_IMAGE_TYPE_2D;
  info.format = VK_FORMAT_R8_UNORM;
  info.extent = {1, 1, 1};
  info.mipLevels = 1;
  info.arrayLayers = 1;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  const VkDevice dev = device.handle();
  VkImage probe = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  id<MTLDevice> metal = nil;
  if (export_objects != nullptr &&
      vkCreateImage(dev, &info, nullptr, &probe) == VK_SUCCESS &&
      (memory = bind_memory(device, probe)) != VK_NULL_HANDLE) {
    VkExportMetalTextureInfoEXT texture{};
    texture.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_TEXTURE_INFO_EXT;
    texture.image = probe;
    texture.plane = VK_IMAGE_ASPECT_PLANE_0_BIT;
    VkExportMetalObjectsInfoEXT objects{};
    objects.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT;
    objects.pNext = &texture;
    export_objects(dev, &objects);
    metal = texture.mtlTexture.device;
  }
  vkDestroyImage(dev, probe, nullptr);
  vkFreeMemory(dev, memory, nullptr);
  if (metal == nil) {
    return core::Status::unsupported(who +
                                     ": the device names no Metal device");
  }
  auto impl = std::make_unique<Impl>();
  impl->device = &device;
  impl->who = std::move(who);
  impl->metal = metal;
  return PixelBufferImporter(std::move(impl));
}

PixelBufferImporter::PixelBufferImporter(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
PixelBufferImporter::PixelBufferImporter(PixelBufferImporter&& other) noexcept =
    default;
PixelBufferImporter& PixelBufferImporter::operator=(
    PixelBufferImporter&& other) noexcept = default;
PixelBufferImporter::~PixelBufferImporter() = default;

core::Status PixelBufferImporter::import(CVPixelBufferRef pixels,
                                         std::uint32_t width,
                                         std::uint32_t height, YuvImage& out) {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument("PixelBufferImporter: moved from");
  }
  if (pixels == nullptr) {
    return core::Status::invalid_argument(impl_->who + ": no pixel buffer");
  }
  const OSType format = CVPixelBufferGetPixelFormatType(pixels);
  IOSurfaceRef surface = CVPixelBufferGetIOSurface(pixels);
  if (format != kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange &&
      format != kCVPixelFormatType_420YpCbCr8BiPlanarFullRange) {
    return core::Status::unsupported(impl_->who +
                                     ": the pixel buffer is not 8-bit NV12");
  }
  if (surface == nullptr || CVPixelBufferGetPlaneCount(pixels) != 2 ||
      CVPixelBufferGetWidthOfPlane(pixels, 0) < width ||
      CVPixelBufferGetHeightOfPlane(pixels, 0) < height ||
      CVPixelBufferGetWidthOfPlane(pixels, 1) < (width + 1) / 2 ||
      CVPixelBufferGetHeightOfPlane(pixels, 1) < (height + 1) / 2) {
    return core::Status::unsupported(impl_->who +
                                     ": the pixel buffer is not on an "
                                     "IOSurface of the picture's size");
  }
  Impl& impl = *impl_;
  const std::uint64_t now = ++impl.pictures;
  auto& surfaces = impl.surfaces;
  surfaces.erase(std::remove_if(surfaces.begin(), surfaces.end(),
                                [now](const std::shared_ptr<Surface>& s) {
                                  return s->seen + kStale < now;
                                }),
                 surfaces.end());
  auto found = std::find_if(surfaces.begin(), surfaces.end(),
                            [surface](const std::shared_ptr<Surface>& s) {
                              return s->surface == surface;
                            });
  std::shared_ptr<Surface> entry;
  if (found != surfaces.end()) {
    entry = *found;
  } else {
    VKC_ASSIGN(entry, impl.import_surface(pixels, surface));
    surfaces.push_back(entry);
  }
  entry->seen = now;
  const auto held = std::make_shared<const Held>(entry, pixels);
  out.layout = YuvLayout::Nv12;
  out.device.reset();
  out.image[0] = std::shared_ptr<const core::Image>(held, &entry->luma);
  out.image[1] = std::shared_ptr<const core::Image>(held, &entry->chroma);
  out.width = width;
  out.height = height;
  return {};
}

}  // namespace volumetric_kit::recon::sensor
