"""tau-vision: image observation returned by tools in vision modality."""

from typing import Optional

from pydantic import BaseModel, Field


class ImageObservation(BaseModel):
    """A tool result that is an image.

    ``alt_text`` is a deliberately non-leaking placeholder (e.g. "Status bar
    screenshot attached") — it must never restate the facts depicted, or the
    vision arm degenerates into the text arm.
    ``kappa`` carries the symbolic record the image was rendered from; it is
    consumed by the evaluator and oracle-caption substitution ONLY and must
    never be shown to the agent.
    """

    image_b64: str = Field(description="Base64-encoded PNG.")
    alt_text: str = Field(description="Non-leaking placeholder text.")
    kappa: Optional[dict] = Field(
        default=None,
        description="Symbolic source record (evaluator/oracle only; never for the agent).",
    )
