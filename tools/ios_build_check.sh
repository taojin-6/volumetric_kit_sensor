#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin
#
# The iOS build check the macOS CI leg runs (.github/workflows/_build.yml):
# sensor_apple, sensor_video and the H.265 syntax it reads with
# (sensor_video_syntax), which every Apple build has, the frames' wire form
# (sensor_wire), which an iOS writer links, and the encoder rules an app's
# device test runs (sensor_video_testing),
# compiled for the iOS floor with the include paths and definitions of a macOS
# build's compile commands, warnings as errors, the importer with ARC, and
# its header and the other Apple headers as plain C++ too; then linked into
# an iOS library against the SDK's frameworks, and with an app's calls to
# code a camera's pictures (tools/ios_link_check.cpp) into an iOS
# executable, so a symbol iOS does not have, an FFmpeg or CUDA one among
# them, fails the link. Only their dependencies' symbols are left
# unresolved: the core's, the package's other tiers' and Vulkan's.
#
#   tools/ios_build_check.sh <build>/compile_commands.json <out dir>
#
# Needs an iOS SDK, which only full Xcode has. Runs on macOS's bash 3.2.
set -euo pipefail

db="$1"
out="$2"
main="$(cd "$(dirname "$0")" && pwd)/ios_link_check.cpp"
mkdir -p "$out"
ios=(xcrun --sdk iphoneos clang++ -target arm64-apple-ios26.0)

# Each source of the five targets, with its own flags, listed as compiled;
# then the Apple headers alone as plain C++, and the app's calls, with the
# flags of the target that sees them all.
python3 -I - "$db" "$out" "$main" "${ios[@]}" >"$out/objects.txt" <<'EOF'
import json, os, shlex, subprocess, sys

db, out, main, ios = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
ios += ["-std=c++17", "-Wall", "-Wextra", "-Wpedantic", "-Werror"]
importer = b'#include "volumetric_kit/sensor/apple/pixel_buffer_importer.hpp"\n'
headers = ["apple/core_video_color.hpp", "video/hevc_encoder_apple.hpp",
           "video/backend.hpp", "testing/hevc_conformance.hpp",
           "testing/hevc_slices.hpp", "testing/nv12_pixel_buffer.hpp"]
targets = ("sensor_apple", "sensor_video", "sensor_video_syntax",
           "sensor_video_testing", "sensor_wire")
objects = []
# sensor_video_testing's flags and sensor_apple's, which together see every
# header, as an app linking both does.
widest = []
seen = set()
for entry in json.load(open(db)):
    output = entry.get("output", "")
    if not any("/%s.dir/" % target in output for target in targets):
        continue
    args = iter(entry.get("arguments") or shlex.split(entry["command"]))
    flags = []
    for arg in args:
        if arg in ("-I", "-isystem", "-D"):
            flags += [arg, next(args)]
        elif arg.startswith(("-I", "-isystem", "-D")):
            flags.append(arg)
    if any("/%s.dir/" % t in output
           for t in ("sensor_video_testing", "sensor_apple")):
        for i, flag in enumerate(flags):
            if flag in ("-I", "-isystem", "-D") or flag in seen:
                continue
            seen.add(flag)
            prev = flags[i - 1] if i > 0 else ""
            widest += [prev, flag] if prev in ("-I", "-isystem", "-D") \
                else [flag]
    source = os.path.join(entry["directory"], entry["file"])
    objc = source.endswith(".mm")
    language = ["-x", "objective-c++", "-fobjc-arc"] if objc else ["-x", "c++"]
    obj = os.path.join(out, "%02d.o" % len(objects))
    subprocess.run(ios + flags + language + ["-c", source, "-o", obj],
                   check=True, stdout=sys.stderr)
    if objc:
        subprocess.run(ios + flags + ["-x", "c++", "-fsyntax-only", "-"],
                       input=importer, check=True, stdout=sys.stderr)
    objects.append(obj)
if not objects or not widest:
    sys.exit("no compile command of " + ", ".join(targets) + " in " + db)
for header in headers:
    line = '#include "volumetric_kit/sensor/%s"\n' % header
    subprocess.run(ios + widest + ["-x", "c++", "-fsyntax-only", "-"],
                   input=line.encode(), check=True, stdout=sys.stderr)
subprocess.run(ios + widest + ["-x", "c++", "-c", main, "-o",
                               os.path.join(out, "main.o")],
               check=True, stdout=sys.stderr)
print("\n".join(objects))
EOF
objects=()
while IFS= read -r obj; do
  objects+=("$obj")
done <"$out/objects.txt"
frameworks=(-framework VideoToolbox -framework CoreMedia -framework CoreVideo
  -framework CoreFoundation -framework IOSurface -framework Metal
  -framework Foundation)

# The symbols @p object leaves undefined that are its dependencies' (the
# core's, the package's other tiers', Vulkan's), as linker flags that let
# them stay so; any other undefined symbol fails the link.
dependency_flags() {
  nm -u -j "$1" | sort -u >"$out/undefined.txt"
  xcrun c++filt <"$out/undefined.txt" >"$out/demangled.txt"
  paste "$out/undefined.txt" "$out/demangled.txt" |
    awk -F'\t' '$2 ~ /volumetric_kit::(core|sensor)::/ || $1 ~ /^_vk[A-Z]/ {
      print "-Wl,-U," $1 }'
}

# One object, so only the symbols the targets do not define stay undefined.
"${ios[@]}" -r -nostdlib -o "$out/targets.o" "${objects[@]}"
dependency_flags "$out/targets.o" >"$out/dependency_symbols.txt"
link=("${ios[@]}" -dynamiclib -o "$out/libios_build_check.dylib"
  "$out/targets.o" "${frameworks[@]}")
while IFS= read -r flag; do
  link+=("$flag")
done <"$out/dependency_symbols.txt"
"${link[@]}"

# The app's calls with the targets, as an executable.
"${ios[@]}" -r -nostdlib -o "$out/app.o" "$out/main.o" "$out/targets.o"
dependency_flags "$out/app.o" >"$out/app_dependency_symbols.txt"
link=("${ios[@]}" -o "$out/ios_link_check" "$out/app.o" "${frameworks[@]}")
while IFS= read -r flag; do
  link+=("$flag")
done <"$out/app_dependency_symbols.txt"
"${link[@]}"
echo "iOS build check: ${#objects[@]} sources linked, and an app's calls" \
  "into an executable; $(wc -l <"$out/app_dependency_symbols.txt" |
    tr -d ' ') dependency symbols left unresolved"
