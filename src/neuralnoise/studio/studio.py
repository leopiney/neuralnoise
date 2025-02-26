import hashlib
import os
import json
from pathlib import Path
from string import Template
from typing import Any

from pydub import AudioSegment
from pydub.effects import normalize
from tqdm.auto import tqdm

import yaml
from autogen import ChatResult
from neuralnoise.models import StudioConfig
from neuralnoise.studio.agents.agents_manager import AgentsManager
from neuralnoise.studio.agents.context_manager import SharedContext
from neuralnoise.tts import generate_audio_segment
from neuralnoise.utils import package_root


# Custom JSON encoder to handle ChatResult serialization
class ChatResultEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ChatResult):
            # Convert ChatResult to dictionary with relevant fields
            return {
                "chat_history": obj.chat_history,
                "summary": obj.summary,
                "cost": obj.cost,
            }
        return super().default(obj)


class PodcastStudio:
    """Manages the end-to-end process of podcast generation using agents and TTS."""

    def __init__(
        self, work_dir: str | Path, config: StudioConfig, max_round: int = 50
    ) -> None:
        """
        Initialize the podcast studio with configuration.

        Args:
            work_dir: Working directory for outputs and intermediate files
            config: Studio configuration with show details, speakers etc.
            max_round: Maximum conversation rounds for agent-based generation
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.language = config.show.language
        self.max_round = max_round

        # Load system prompts
        prompts_dir = (
            config.prompts_dir if config.prompts_dir else package_root / "prompts"
        )
        self.system_msgs = self._load_system_messages(prompts_dir)

        # Create agents manager
        self.agents_manager = AgentsManager(
            system_msgs=self.system_msgs,
            llm_config=self._load_llm_config(),
            work_dir=self.work_dir,
            language=self.language,
        )

    def _load_system_messages(self, prompts_dir: Path) -> dict[str, str]:
        """Load all system messages from prompts directory."""
        system_msgs = {}

        # Load built-in system messages
        system_msgs["ContentAnalyzerAgent"] = self.load_prompt_file(
            prompts_dir / "content_analyzer.system.xml", language=self.language
        )
        system_msgs["PlannerAgent"] = self.load_prompt_file(
            prompts_dir / "planner.system.xml", language=self.language
        )
        system_msgs["ScriptGeneratorAgent"] = self.load_prompt_file(
            prompts_dir / "script_generation.system.xml",
            language=self.language,
            min_segments=str(self.config.show.min_segments),
            max_segments=str(self.config.show.max_segments),
        )
        system_msgs["EditorAgent"] = self.load_prompt_file(
            prompts_dir / "editor.system.xml", language=self.language
        )
        system_msgs["UserMessage"] = self.load_prompt_file(
            prompts_dir / "user_proxy.message.xml",
            content="$content",
            show=self.config.show.model_dump_json(),
            speakers=str(self.config.speakers),
        )

        return system_msgs

    def load_prompt_file(self, path: Path, **kwargs: str) -> str:
        """Load a prompt from a file and substitute variables."""
        if not path.exists():
            return ""

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        if kwargs:
            template = Template(content)
            content = template.safe_substitute(kwargs)

        return content

    def _load_llm_config(self) -> dict:
        """Load LLM configuration."""
        return {
            "config_list": [
                {
                    "model": "gpt-4o",
                    "api_key": os.environ.get("OPENAI_API_KEY", ""),
                }
            ]
        }

    def load_prompt(self, prompt_name: str, **kwargs: str) -> str:
        """Load and substitute a prompt from the prompts directory."""
        content = self.system_msgs.get(prompt_name, "")
        if content and kwargs:
            template = Template(content)
            content = template.safe_substitute(kwargs)
        return content

    def generate_script(self, content: str) -> dict[str, Any]:
        """Generate a podcast script using AgentsManager and return the final script and chat log."""
        # For debugging
        print(f"DEBUG - Content length: {len(content) if content else 0}")
        print(f"DEBUG - Content preview: {content[:100] if content else 'None'}")

        # Prepare initial message using the user message template with content properly embedded
        if "$content" in self.system_msgs["UserMessage"]:
            initial_message = self.system_msgs["UserMessage"].replace(
                "$content", content
            )
        else:
            # If no $content placeholder, just pass the content directly
            initial_message = content

        # For debugging
        print(f"DEBUG - Initial message length: {len(initial_message)}")
        print(f"DEBUG - Initial message preview: {initial_message[:100]}")

        # Explicitly create a content-loaded shared state
        shared_state = SharedContext()
        shared_state.update_content(content)

        chat_result, shared_state, last_agent = self.agents_manager.run_swarm_chat(
            initial_message
        )

        script_data = {
            "sections": shared_state.section_scripts,
            "messages": chat_result,
        }

        # Use custom encoder for JSON serialization
        return json.loads(json.dumps(script_data, cls=ChatResultEncoder))

    def generate_podcast_from_script(self, script: dict[str, Any]) -> AudioSegment:
        """Generate the podcast audio from the script using TTS."""
        temp_dir = self.work_dir / "segments"
        temp_dir.mkdir(exist_ok=True)

        # Gather all segments from each section
        script_segments: list[tuple[str, dict]] = []
        for section_id in sorted(script["sections"].keys()):
            section = script["sections"][section_id]
            for segment in section.get("segments", []):
                script_segments.append((section_id, segment))

        audio_segments: list[AudioSegment] = []
        for section_id, segment in tqdm(
            script_segments, desc="Generating audio segments"
        ):
            speaker = self.config.speakers[segment["speaker"]]
            text = segment["content"].replace("¡", "").replace("¿", "")
            content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
            segment_path = temp_dir / f"{section_id}_{segment['id']}_{content_hash}.mp3"
            audio_segment = generate_audio_segment(
                text, speaker, output_path=segment_path
            )
            audio_segments.append(audio_segment)
            if blank_duration := segment.get("blank_duration"):
                silence = AudioSegment.silent(duration=blank_duration * 1000)
                audio_segments.append(silence)

        podcast = AudioSegment.empty()
        for seg in audio_segments:
            podcast += seg
        podcast = normalize(podcast)
        return podcast
