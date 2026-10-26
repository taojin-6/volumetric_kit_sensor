#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin
#
# The iOS build check the macOS CI leg runs (.github/workflows/_build.yml):
# recon_sensor_apple, recon_sensor_video and the H.265 syntax it reads with
# (recon_sensor_video_syntax), which every Apple build has, compiled for the
# iOS floor with the include paths and definitions of a macOS
# build's compile commands, warnings as errors, the importer with ARC and its
# header as plain C++ too; then linked into an iOS library against the SDK's
# frameworks, so a symbol iOS does not have fails the link. Only their
# dependencies' symbols are left unresolved: the core's, recon's other tiers'
# and Vulkan's.
#
#   tools/ios_build_check.sh <build>/compile_commands.json <out dir>
#
# Needs an iOS SDK, which only full Xcode has. Runs on macOS's bash 3.2.
set -euo pipefail

db="$1"
out="$2"
mkdir -p "$out"
ios=(xcrun --sdk iphoneos clang++ -target arm64-apple-ios26.0)

# Each source of the three targets, with its own flags, listed as compiled.
python3 -I - "$db" "$out" "${ios[@]}" >"$out/objects.txt" <<'EOF'
import json, os, shlex, subprocess, sys

db, out, ios = sys.argv[1], sys.argv[2], sys.argv[3:]
ios += ["-std=c++17", "-Wall", "-Wextra", "-Wpedantic", "-Werror"]
header = b'#include "volumetric_kit/recon/sensor/apple/pixel_buffer_importer.hpp"\n'
targets = ("recon_sensor_apple", "recon_sensor_video",
           "recon_sensor_video_syntax")
objects = []
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
    source = os.path.join(entry["directory"], entry["file"])
    objc = source.endswith(".mm")
    language = ["-x", "objective-c++", "-fobjc-arc"] if objc else ["-x", "c++"]
    obj = os.path.join(out, "%02d.o" % len(objects))
    subprocess.run(ios + flags + language + ["-c", source, "-o", obj],
                   check=True, stdout=sys.stderr)
    if objc:
        subprocess.run(ios + flags + ["-x", "c++", "-fsyntax-only", "-"],
                       input=header, check=True, stdout=sys.stderr)
    objects.append(obj)
if not objects:
    sys.exit("no compile command of " + ", ".join(targets) + " in " + db)
print("\n".join(objects))
EOF
objects=()
while IFS= read -r obj; do
  objects+=("$obj")
done <"$out/objects.txt"

# One object, so only the symbols the targets do not define stay undefined;
# their dependencies' symbols may stay so (-U), and any other fails.
"${ios[@]}" -r -nostdlib -o "$out/targets.o" "${objects[@]}"
nm -u -j "$out/targets.o" | sort -u >"$out/undefined.txt"
xcrun c++filt <"$out/undefined.txt" >"$out/demangled.txt"
paste "$out/undefined.txt" "$out/demangled.txt" |
  awk -F'\t' '$2 ~ /volumetric_kit::(core|recon)::/ || $1 ~ /^_vk[A-Z]/ {
    print "-Wl,-U," $1 }' >"$out/dependency_symbols.txt"
link=("${ios[@]}" -dynamiclib -o "$out/libios_build_check.dylib"
  "$out/targets.o" -framework VideoToolbox -framework CoreMedia
  -framework CoreVideo -framework CoreFoundation -framework IOSurface
  -framework Metal -framework Foundation)
while IFS= read -r flag; do
  link+=("$flag")
done <"$out/dependency_symbols.txt"
"${link[@]}"
echo "iOS build check: ${#objects[@]} sources linked," \
  "$(wc -l <"$out/dependency_symbols.txt" | tr -d ' ') dependency symbols" \
  "left unresolved"
