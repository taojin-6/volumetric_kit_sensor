# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# Pinned dependencies (FetchContent). The package takes Vulkan and glm from the
# system (find_package in the tiers). Besides the family's volumetric_kit_core,
# what is fetched here is header-only with no system package to rely on across
# the platforms: nlohmann/json and stb. The fetches run on the first configure
# (a network is required then; afterwards the FetchContent cache serves it).
# VS_CAMERA_ONLY stops after the core and nlohmann/json.

include(FetchContent)

# --- volumetric_kit_core: the family's shared foundation ---------------------
# Status / Result / VKC_CHECK and the log sink from its base tier, and its
# Vulkan foundation from its vulkan tier, so they are the same types recon,
# calib and gfx use, and one VkDevice can serve all of them. Pinned by commit,
# never main. FetchContent keeps the first declaration of a name: a project that
# adds this package (recon, or an application) declares the core first, so this
# pin then yields to that one, and vkc_require_core refuses a core older than
# VS_CORE_MIN_VERSION, or one without the vulkan tier outside VS_CAMERA_ONLY.
# Work on both checkouts at once with
# -DFETCHCONTENT_SOURCE_DIR_VOLUMETRIC_KIT_CORE=<core checkout>. VS_CAMERA_ONLY
# needs the base tier alone; it leaves VKC_WITH_VULKAN unset rather than OFF, so
# an application that turns the tier on keeps it.
#
# The pin is core 0.29.0 (a4a6fff, core #65's merge). The tests are main()s of
# their own on the core's test support, whose Vulkan-free part a camera-only
# build's tests use too.
FetchContent_Declare(
  volumetric_kit_core
  GIT_REPOSITORY https://github.com/taojin-6/volumetric_kit_core.git
  GIT_TAG a4a6fff4af65b26b25df27c8ed8c9eecbfbfcd87)
if(NOT VS_CAMERA_ONLY)
  set(VKC_WITH_VULKAN ON)
endif()
if(VS_BUILD_TESTS)
  set(VKC_BUILD_TEST_SUPPORT ON)
endif()
FetchContent_MakeAvailable(volumetric_kit_core)

# The set()s above are lost when the project that added this package made the
# core available first, without its vulkan tier or its test support.
if(VS_CAMERA_ONLY)
  vkc_require_core(${VS_CORE_MIN_VERSION})
else()
  vkc_require_core(${VS_CORE_MIN_VERSION} VULKAN)
endif()
if(VS_BUILD_TESTS
   AND NOT (TARGET volumetric_kit::core_test_base
            AND (VS_CAMERA_ONLY OR TARGET volumetric_kit::core_test_policy)))
  message(
    FATAL_ERROR
      "the sensor package's tests need volumetric_kit_core's test support: set "
      "VKC_BUILD_TEST_SUPPORT ON before the first FetchContent_MakeAvailable "
      "of the core")
endif()

# --- nlohmann/json (header-only; private to the targets that parse JSON) -----
# sensor_camera parses the sensor array's calibration file with it, so a
# camera-only build fetches it too, and sensor_orbbec its sync file. The
# release's source archive, hash-pinned; only its include/ tree is used, so its
# CMake is not run (SOURCE_SUBDIR names a directory the archive does not have).
# It is a PRIVATE SYSTEM include of each target that parses, never part of its
# interface. volumetric_kit_gfx carries the same library through tinygltf.
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW) # extracted files get the extraction time
endif()
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  URL_HASH
    SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
  SOURCE_SUBDIR do-not-add-json-cmake)
FetchContent_MakeAvailable(nlohmann_json)

set(VS_JSON_INCLUDE_DIR
    "${nlohmann_json_SOURCE_DIR}/include"
    PARENT_SCOPE)

# The camera tier needs nothing below.
if(VS_CAMERA_ONLY)
  return()
endif()

# --- stb (image decoding and PNG writing; private to sensor_image_io) --------
# Its implementation TU lives in sensor_image_io; no helper target leaks through
# its link interface.
FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG 2c980bb59875b0d32144a71867fbdebb2f77cd20
  GIT_SHALLOW FALSE)
FetchContent_MakeAvailable(stb)
set(VS_STB_INCLUDE_DIR
    "${stb_SOURCE_DIR}"
    PARENT_SCOPE)
