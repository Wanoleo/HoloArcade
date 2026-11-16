"""Non-blocking Windows speech narrator for HoloEscape."""

from __future__ import annotations

import subprocess
import sys


class Narrator:
    """Speaks escape-room narration aloud using Windows SAPI without blocking the game loop."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and sys.platform.startswith("win")
        self.current_line: str = ""
        self.history: list[str] = []
        self._proc: subprocess.Popen | None = None

    def speak(self, text: str) -> None:
        """Speak a narration line asynchronously, interrupting any previous speech."""
        clean = text.strip()
        if not clean:
            return

        self.current_line = clean
        if not self.history or self.history[-1] != clean:
            self.history.append(clean)

        if not self.enabled:
            return

        self.stop()
        # Escape single quotes for PowerShell single-quoted string literal.
        ps_text = clean.replace("'", "''")
        ps_cmd = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$s.Rate = 0; "
            f"$s.Speak('{ps_text}')"
        )
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creation_flags,
            )
        except OSError:
            self._proc = None

    def replay(self) -> None:
        """Replay the most recent narration line."""
        if self.current_line:
            self.speak(self.current_line)

    def stop(self) -> None:
        """Stop any currently playing speech process."""
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except OSError:
                pass
        self._proc = None
