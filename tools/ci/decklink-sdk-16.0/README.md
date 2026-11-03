# DeckLink SDK 16.0, Linux headers

`include/` is the `Linux/include` directory of Blackmagic Design's Desktop
Video SDK 16.0, unmodified: the headers and `DeckLinkAPIDispatch.cpp` that
the DeckLink driver builds against (`VS_WITH_DECKLINK`). The action
`.github/actions/decklink-sdk` holds them to `SHA256SUMS` and points
`DeckLinkSDK_ROOT` at this directory, for the 24.04 leg here and for a
consumer's DeckLink leg at its sensor pin. Why they are kept here:
DECISIONS.md, the 2026-11-01 DeckLink entry.

Each file carries Blackmagic's copyright notice and licence statement, which
must stay with every copy; they are redistributed under that statement and
the SDK's EULA (https://www.blackmagicdesign.com/EULA/DeckLinkSDK). Nothing
here is the package's, so no formatting hook rewrites them
(`.pre-commit-config.yaml`).

`SHA256SUMS` lists each file's SHA256 as the SDK ships it, and
`../test_decklink_sdk.py`, which lint runs, holds every file to it and to
its licence statement. To move to a newer SDK, replace `include/` with its
`Linux/include`, regenerate the list from this directory with
`(cd include && LC_ALL=C ls | while read f; do shasum -a 256 "$f" | sed "s|  |  include/|"; done) > SHA256SUMS`,
and rename the directory for its version, with the action's path to it.
