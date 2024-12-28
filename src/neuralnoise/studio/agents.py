import hashlib
import json
import os
from pathlib import Path
from string import Template
from typing import Any, Callable

from autogen import (  # type: ignore
    Agent,
    AssistantAgent,
    GroupChat,
    GroupChatManager,
    UserProxyAgent,
)
from pydub import AudioSegment
from pydub.effects import normalize
from tqdm.auto import tqdm

from neuralnoise.models import ContentAnalysis, StudioConfig, PodcastScript
from neuralnoise.studio.hooks import (
    optimize_chat_history_hook,
    save_last_json_message_hook,
    save_last_message_hook,
)
from neuralnoise.tts import generate_audio_segment
from neuralnoise.utils import package_root


def agent(func: Callable) -> Callable:
    func.is_agent = True  # type: ignore

    return func


class PodcastStudio:
    def __init__(self, work_dir: str | Path, config: StudioConfig, max_round: int = 50):
        self.work_dir = Path(work_dir)
        self.config = config
        self.language = config.show.language
        self.max_round = max_round

        self.llm_default_config = {
            "model": "gpt-4o",
            "api_key": os.environ["OPENAI_API_KEY"],
        }

        self.agents: list[Agent] = []
        for attr in dir(self):
            if hasattr(getattr(self, attr), "is_agent"):
                self.agents.append(getattr(self, attr)())

    def load_prompt(self, prompt_name: str, **kwargs: str) -> str:
        root_folder = self.config.prompts_dir or package_root / "prompts"
        prompt_path = root_folder / f"{prompt_name}.xml"

        with open(prompt_path, "r") as f:
            content = f.read()

        if kwargs:
            template = Template(content)
            content = template.safe_substitute(kwargs)

        return content

    @agent
    def content_analyzer_agent(self) -> AssistantAgent:
        agent = AssistantAgent(
            name="ContentAnalyzerAgent",
            system_message=self.load_prompt(
                "content_analyzer.system", language=self.language
            ),
            llm_config={
                "config_list": [
                    {
                        **self.llm_default_config,
                        "response_format": ContentAnalysis,
                    }
                ]
            },
        )
        agent.register_hook(
            hookable_method="process_message_before_send",
            hook=save_last_json_message_hook(
                "content_analyzer", self.work_dir / "analyzer"
            ),
        )

        return agent

    @agent
    def planner_agent(self) -> AssistantAgent:
        return AssistantAgent(
            name="PlannerAgent",
            system_message=self.load_prompt("planner.system", language=self.language),
            llm_config={"config_list": [self.llm_default_config]},
        )

    @agent
    def script_generator_agent(self) -> AssistantAgent:
        agent = AssistantAgent(
            name="ScriptGeneratorAgent",
            system_message=self.load_prompt(
                "script_generation.system",
                language=self.language,
                min_segments=str(self.config.show.min_segments),
                max_segments=str(self.config.show.max_segments),
            ),
            llm_config={
                "config_list": [
                    {
                        **self.llm_default_config,
                        "response_format": PodcastScript,
                    }
                ]
            },
        )
        agent.register_hook(
            hookable_method="process_message_before_send",
            hook=save_last_json_message_hook(
                "script_generator", self.work_dir / "scripts"
            ),
        )
        agent.register_hook(
            hookable_method="process_all_messages_before_reply",
            hook=optimize_chat_history_hook(
                agents=["ScriptGeneratorAgent", "EditorAgent", "PlannerAgent"]
            ),
        )

        return agent

    @agent
    def editor_agent(self) -> AssistantAgent:
        agent = AssistantAgent(
            name="EditorAgent",
            system_message=self.load_prompt("editor.system", language=self.language),
            llm_config={"config_list": [self.llm_default_config]},
        )
        agent.register_hook(
            hookable_method="process_all_messages_before_reply",
            hook=optimize_chat_history_hook(
                agents=["ScriptGeneratorAgent", "EditorAgent", "PlannerAgent"]
            ),
        )

        return agent

    def generate_script(self, content: str) -> dict[str, Any]:
        def is_termination_msg(message):
            return isinstance(message, dict) and (
                message.get("content", "") == ""
                or message.get("content", "").rstrip().endswith("TERMINATE")
            )

        user_proxy = UserProxyAgent(
            name="UserProxy",
            system_message=self.load_prompt("user_proxy.system"),
            human_input_mode="TERMINATE",
            code_execution_config=False,
        )

        groupchat = GroupChat(
            agents=self.agents,
            messages=[],
            max_round=self.max_round,
        )

        manager = GroupChatManager(
            groupchat=groupchat,
            llm_config={"config_list": [self.llm_default_config]},
            system_message=self.load_prompt("manager.system"),
            is_termination_msg=is_termination_msg,
        )

        # Initiate the chat with a clear task description
        user_proxy.initiate_chat(
            manager,
            message=self.load_prompt(
                "user_proxy.message",
                content=content,
                show=self.config.render_show_details(),
                speakers=self.config.render_speakers_details(),
            ),
        )

        # Extract the final script from the chat history
        # TODO: improve this logic
        script_sections: dict[str, Any] = {}
        for script_filepath in sorted(self.work_dir.glob("scripts/*.json")):
            with open(script_filepath) as f:
                script = json.load(f)
                script_sections[script["section_id"]] = script

        # Combine all approved script sections
        final_script = {
            "sections": script_sections,
            "messages": groupchat.messages,
        }

        return final_script

    def generate_podcast_from_script(self, script: dict[str, Any]) -> AudioSegment:
        script_segments = []

        temp_dir = self.work_dir / "segments"
        temp_dir.mkdir(exist_ok=True)

        sections_ids = list(sorted(script["sections"].keys()))
        script_segments = [
            (section_id, segment)
            for section_id in sections_ids
            for segment in script["sections"][section_id]["segments"]
        ]

        audio_segments = []

        for section_id, segment in tqdm(
            script_segments,
            desc="Generating audio segments",
        ):
            speaker = self.config.speakers[segment["speaker"]]
            content = segment["content"]

            content = content.replace("¡", "").replace("¿", "")

            content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
            segment_path = temp_dir / f"{section_id}_{segment['id']}_{content_hash}.mp3"

            audio_segment = generate_audio_segment(
                content, speaker, output_path=segment_path
            )

            audio_segments.append(audio_segment)

            if blank_duration := segment.get("blank_duration"):
                silence = AudioSegment.silent(duration=blank_duration * 1000)
                audio_segments.append(silence)

        podcast = AudioSegment.empty()

        for chunk in audio_segments:
            podcast += chunk

        podcast = normalize(podcast)

        return podcast
