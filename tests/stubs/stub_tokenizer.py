from __future__ import annotations

from collections.abc import Sequence


class PieceTableTextCodecStub:
    def __init__(self) -> None:
        self.id_to_piece = {
            1: "(SSNHL)",
            2: ".",
            3: " ",
            4: " suggest",
            5: " shows",
            6: "Old sentence.",
            7: " noisy preface",
            8: " SSNHL",
            9: " is urgent.",
            10: " forbidden",
            11: " allowed",
            12: " this suggests",
            13: " hearing loss",
            14: " maybe",
            15: " end.",
            16: " fallback",
            17: " next.",
        }
        self.piece_to_id = {v: k for k, v in self.id_to_piece.items()}
        self._next_id = 1000

    def decode(
        self, ids: Sequence[int], *, _skip_special_tokens: bool = False
    ) -> str:
        return "".join(self.id_to_piece.get(i, "") for i in ids)

    def encode(
        self, text: str, *, _add_special_tokens: bool = False
    ) -> list[int]:
        if text == "":
            return list()
        out: list[int] = []
        i = 0
        pieces = sorted(self.piece_to_id.keys(), key=len, reverse=True)
        while i < len(text):
            matched = None
            for p in pieces:
                if text.startswith(p, i):
                    matched = p
                    break
            if matched is not None:
                out.append(self.piece_to_id[matched])
                i += len(matched)
                continue
            ch = text[i]
            if ch not in self.piece_to_id:
                self.piece_to_id[ch] = self._next_id
                self.id_to_piece[self._next_id] = ch
                self._next_id += 1
            out.append(self.piece_to_id[ch])
            i += 1
        return out
