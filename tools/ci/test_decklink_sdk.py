# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin
"""The DeckLink SDK's files in decklink-sdk-16.0/ stay as Blackmagic ships
them (decklink-sdk-16.0/README.md), where the action that exports them looks;
lint runs these tests."""

import hashlib
import os
import re
import unittest

SDK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decklink-sdk-16.0")


def listed():
    """{path: SHA256} from SHA256SUMS."""
    with open(os.path.join(SDK, "SHA256SUMS"), encoding="utf-8") as file:
        rows = [line.split("  ", 1) for line in file.read().splitlines() if line]
    return {path: digest for digest, path in rows}


class DecklinkSdkTest(unittest.TestCase):
    def test_every_file_is_listed_and_unchanged(self):
        sums = listed()
        present = sorted("include/" + name for name in os.listdir(os.path.join(SDK, "include")))
        self.assertEqual(sorted(sums), present)
        for path, digest in sums.items():
            with open(os.path.join(SDK, path), "rb") as file:
                self.assertEqual(hashlib.sha256(file.read()).hexdigest(), digest, path)

    def test_every_file_keeps_its_licence_statement(self):
        for path in listed():
            with open(os.path.join(SDK, path), encoding="utf-8") as file:
                head = file.read(4096)
            self.assertIn("-LICENSE-START-", head, path)
            self.assertIn("Blackmagic Design", head, path)

    def test_what_sensor_builds_against_is_there(self):
        # cmake/vs_decklink.cmake: DeckLinkSDK_ROOT/include holds
        # DeckLinkAPI.h, DeckLinkAPIDispatch.cpp and DeckLinkAPIVersion.h.
        for name in ("DeckLinkAPI.h", "DeckLinkAPIDispatch.cpp", "DeckLinkAPIVersion.h"):
            self.assertTrue(os.path.isfile(os.path.join(SDK, "include", name)), name)
        with open(os.path.join(SDK, "include", "DeckLinkAPIVersion.h"), encoding="utf-8") as file:
            self.assertIn('BLACKMAGIC_DECKLINK_API_VERSION_STRING\t\t\t"16.0"', file.read())

    def test_the_action_exports_this_directory(self):
        # .github/actions/decklink-sdk finds the headers from its own
        # directory, which a consumer's job has in place of a checkout.
        action = os.path.join(os.path.dirname(SDK), "..", "..", ".github", "actions", "decklink-sdk")
        with open(os.path.join(action, "action.yml"), encoding="utf-8") as file:
            found = re.findall(r'sdk="\$\{GITHUB_ACTION_PATH\}/([^"]+)"', file.read())
        self.assertEqual(len(found), 1, found)
        self.assertTrue(os.path.samefile(os.path.join(action, found[0]), SDK), found[0])


if __name__ == "__main__":
    unittest.main()
