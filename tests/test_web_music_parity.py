"""Keep the browser's checked-in song data aligned with the Python source."""

import json
from pathlib import Path
import unittest

from music import INSTRUMENTS, MELODY_NOTES, MELODY_STEP_SECONDS, MELODY_TITLE


class WebMusicParityTests(unittest.TestCase):
    def test_browser_music_matches_desktop(self):
        path = Path(__file__).resolve().parents[1] / "web" / "src" / "music-data.json"
        with path.open(encoding="utf-8") as source:
            browser = json.load(source)

        self.assertEqual(browser["title"], MELODY_TITLE)
        self.assertEqual(
            browser["melody"],
            [[pitch, duration] for pitch, duration in zip(MELODY_NOTES, MELODY_STEP_SECONDS)],
        )
        self.assertEqual(
            [(item["key"], item["label"], item["freestyleNote"]) for item in browser["instruments"]],
            [(item.key, item.label, item.freestyle_note) for item in INSTRUMENTS],
        )
        self.assertEqual(
            [item["color"] for item in browser["instruments"]],
            ["#{:02x}{:02x}{:02x}".format(*reversed(item.color)) for item in INSTRUMENTS],
        )


if __name__ == "__main__":
    unittest.main()
