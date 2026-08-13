"""Pixel-size (mpp) resolution with a non-silent verification guardrail.

The store's ``mpp`` (microns per pixel of the root image) is load-bearing:
it is the only pixel <-> micron scale, and every physical measurement and
cross-sample comparison rides on it. So it is resolved the way CORAL's
proteomics ingest resolves markers — **loudly**. The value is read from the
data when the platform writes it, cross-checked against a second source, and
when the result is not confident ingest REFUSES to proceed until the user
confirms it, exactly as an unresolved marker blocks a proteomics ingest.
Nothing about mpp is decided silently.

Confidence tiers (highest to lowest):

- ``instrument`` — read directly from a field the instrument wrote
  (Xenium ``experiment.xenium:pixel_size``, CosMx ``ExptConfig.txt:ImPixel_nm``,
  G4X OME-XML ``PhysicalSizeX``, Visium HD ``scalefactors:microns_per_pixel``).
- ``derived`` — computed from geometry in the data (spot diameter/spacing,
  FOV pixel/mm columns). Correct only up to the nominal geometry it assumes.
- ``estimated`` — a third-party pipeline's estimate (HEST).
- ``default`` — the platform's constant used as a fallback because the data
  carried no value. **Always needs confirmation.**
- ``unknown`` — no value and no default. **Always needs confirmation.**

Only ``instrument``/``derived``/``estimated`` values that also pass every
cross-check proceed without confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Instrument-fixed pixel sizes (µm/px). ``None`` means the value is specific
#: to the individual scan (Visium/HEST H&E resolution) and MUST be read from
#: the data — there is deliberately no fake default for those.
PLATFORM_PIXEL_SIZE_UM: dict[str, dict] = {
    "Xenium": {
        "default": 0.2125,
        "note": "experiment.xenium:pixel_size (morphology frame)",
    },
    "CosMx": {
        "default": 0.120281,
        "note": "RunSummary/*_ExptConfig.txt:ImPixel_nm (120.280945 nm)",
    },
    "G4X": {
        "default": 0.3125,
        "note": "g4x_viewer/*.ome.tiff OME-XML PhysicalSizeX",
    },
    "VisiumHD": {"default": None, "note": "scalefactors_json:microns_per_pixel"},
    "Visium": {"default": None, "note": "spot geometry (diameter vs spacing)"},
    "Spatial Transcriptomics": {"default": None, "note": "spot geometry"},
    "HEST": {"default": None, "note": "HEST metadata estimate"},
}

#: Fractional agreement a cross-check must be within to pass (2%).
DEFAULT_TOLERANCE = 0.02


@dataclass(frozen=True)
class CrossCheck:
    """One independent estimate the read value is compared against."""

    name: str
    value: float
    agree: bool


@dataclass(frozen=True)
class PixelSize:
    """A resolved pixel size plus everything needed to audit or confirm it."""

    technology: str
    value: float | None
    tier: str
    source: str
    default: float | None = None
    crosschecks: list[CrossCheck] = field(default_factory=list)
    needs_confirm: bool = False
    reason: str = ""

    def as_config(self) -> dict:
        """A JSON-serializable provenance block for ``config.json``."""
        return {
            "value": self.value,
            "tier": self.tier,
            "source": self.source,
            "default": self.default,
            "crosschecks": [
                {"name": c.name, "value": c.value, "agree": c.agree}
                for c in self.crosschecks
            ],
            "needs_confirm": self.needs_confirm,
            "reason": self.reason,
        }


def resolve_pixel_size(
    technology: str,
    *,
    read_value: float | None = None,
    read_source: str | None = None,
    read_tier: str | None = None,
    crosses: list[tuple[str, float | None]] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> PixelSize:
    """Resolve the pixel size for one sample and decide if it needs confirming.

    Args:
        technology: Canonical technology name.
        read_value: The pixel size read from the data (``None`` if the data
            carried none).
        read_source: Where ``read_value`` came from (file:field), for the audit
            trail.
        read_tier: ``instrument`` / ``derived`` / ``estimated``.
        crosses: Independent ``(name, value)`` estimates to compare against;
            a ``None`` value is skipped. The platform default is added
            automatically as one more cross-check.
        tolerance: Fractional agreement a cross-check must be within.

    Returns:
        A :class:`PixelSize`. ``needs_confirm`` is ``True`` when the value fell
        back to the platform default, is unknown, or any cross-check disagreed.
    """
    reg = PLATFORM_PIXEL_SIZE_UM.get(technology, {"default": None, "note": ""})
    default = reg["default"]

    if read_value:
        checks: list[CrossCheck] = []
        for name, cv in (crosses or []):
            if cv:
                checks.append(
                    CrossCheck(name, float(cv), abs(read_value / cv - 1) <= tolerance)
                )
        if default:
            checks.append(
                CrossCheck(
                    "platform_default",
                    float(default),
                    abs(read_value / default - 1) <= tolerance,
                )
            )
        bad = [c for c in checks if not c.agree]
        reason = (
            ""
            if not bad
            else "cross-check disagreement — "
            + "; ".join(
                f"{c.name}={c.value:.5f} vs read={read_value:.5f}" for c in bad
            )
        )
        return PixelSize(
            technology=technology,
            value=float(read_value),
            tier=read_tier or "read",
            source=read_source or "reader",
            default=default,
            crosschecks=checks,
            needs_confirm=bool(bad),
            reason=reason,
        )

    if default is not None:
        return PixelSize(
            technology=technology,
            value=float(default),
            tier="default",
            source=f"platform default — {reg['note']}",
            default=default,
            crosschecks=[],
            needs_confirm=True,
            reason=(
                f"the data carried no pixel size, so the {technology} platform "
                f"default {default} µm/px was used — confirm it applies to this "
                f"sample"
            ),
        )

    return PixelSize(
        technology=technology,
        value=None,
        tier="unknown",
        source="none",
        default=None,
        crosschecks=[],
        needs_confirm=True,
        reason=(
            f"no pixel size in the data and no platform default for "
            f"{technology}; pass --mpp"
        ),
    )


class MppNeedsConfirmation(RuntimeError):
    """Raised when a sample's mpp is not confident and was not confirmed.

    Mirrors the proteomics marker guardrail: ingest stops and names exactly
    what is uncertain and how to confirm it, rather than writing a store around
    an unverified scale.
    """

    def __init__(self, sample: str, ps: PixelSize) -> None:
        self.sample = sample
        self.pixel_size = ps
        val = "unknown" if ps.value is None else f"{ps.value:.5f} µm/px"
        super().__init__(
            f"{sample}: mpp not confirmed ({ps.tier}, {val}). {ps.reason}. "
            f"Confirm by re-running with --mpp <µm/px> to set it explicitly, or "
            f"--confirm-mpp to accept {val}."
        )
