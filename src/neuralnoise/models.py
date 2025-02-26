from pathlib import Path
from textwrap import dedent
from typing import Literal

from pydantic import BaseModel, Field


class VoiceSettings(BaseModel):
    stability: float = Field(..., ge=0.0, le=1.0)
    similarity_boost: float = Field(..., ge=0.0, le=1.0)
    style: float = Field(default=0.0, ge=0.0, le=1.0)
    speaker_boost: bool = Field(default=False)


class SpeakerSettings(BaseModel):
    voice_id: str

    provider: Literal["elevenlabs", "openai"] = "elevenlabs"
    voice_model: Literal["eleven_multilingual_v2", "tts-1", "tts-1-hd"] = (
        "eleven_multilingual_v2"
    )
    voice_settings: VoiceSettings | None = None


def _display_field(field: str):
    return " ".join([f.capitalize() for f in field.split("_")])


class BaseModelDisplay(BaseModel):
    def render(self, title: str, fields: list[str] | None = None):
        if fields is None:
            fields = list(self.__dict__.keys())

        content = "\n".join(
            [f"\t{_display_field(f)}: {getattr(self, f)}" for f in fields]
        )

        return dedent(f"""
            {title}:
            {content}
        """)


class Speaker(BaseModelDisplay):
    name: str
    about: str

    settings: SpeakerSettings


class Show(BaseModelDisplay):
    name: str
    about: str
    language: str

    min_segments: int = 4
    max_segments: int = 10


class StudioConfig(BaseModelDisplay):
    show: Show
    speakers: dict[str, Speaker]
    prompts_dir: Path | None = None

    def render_show_details(self) -> str:
        return self.show.render("Show")

    def render_speakers_details(self) -> str:
        return "\n\n".join(
            speaker.render(speaker_id, ["name", "about"])
            for speaker_id, speaker in self.speakers.items()
        )


class ContentSegment(BaseModel):
    topic: str
    duration: float  # in minutes
    discussion_points: list[str]


class ContentAnalysis(BaseModelDisplay):
    title: str
    summary: str
    key_points: list[str]
    tone: str
    target_audience: str
    potential_segments: list[ContentSegment]
    controversial_topics: list[str]


class ScriptSegment(BaseModel):
    id: int
    speaker: Literal["speaker1", "speaker2"]
    content: str
    type: Literal["narrative", "reaction", "question"]
    blank_duration: float | None = Field(
        None, description="Time in seconds for silence after speaking"
    )


class PodcastScript(BaseModel):
    section_id: int
    section_title: str
    segments: list[ScriptSegment]
