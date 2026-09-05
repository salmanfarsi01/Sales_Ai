"""Pre-Call Folder & Prompt Assembly Module.

Manages the immutable Pre-Call Folder, the stacked text prompt builder, and
the deterministic "Respond Now or Wait" suppression gate:

0. Core Intelligence Engine (Permanent Foundation — frozen text, always included on every turn)
1. Lead Type (Paragraph about this specific prospect lead type)
2. Playbook Lens (Methodology style, objection response rules, boundaries)
3. Calibration (Salesperson natural cadence, pacing, WPM, rhythm)
4. Training Snippet (High-confidence RAG / company knowledge match)
5. Conversation So Far (Running transcript history + latest prospect utterance)
6. Respond Now or Wait Gate (Deterministic code-level timing & cooldown gate)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any, Dict, List, Optional

from .playbook import (
    Playbook,
    PlaybookStore,
    STANDARDIZED_LEAD_TYPES,
    serialize_playbook_prompt_section,
)

LOGGER = logging.getLogger("copilot.call_folder")
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = WORKSPACE_DIR / "reports"


# =========================================================================
# Piece 1: Core Intelligence — Frozen Text (Written once, always included)
# =========================================================================
CORE_INSTRUCTIONS = (
    "You are the reasoning engine for a live sales call. On every turn:\n"
    "1. Identify literally what the prospect just said.\n"
    "2. Determine the underlying concern or intent behind it — not just the surface words.\n"
    "3. Decide the single best strategic move available right now (e.g., reframe,\n"
    "   validate, clarify, de-risk, quantify, challenge, redirect) based on the\n"
    "   full context provided below.\n"
    "4. Produce ONE exact sentence the rep should say — not advice, not options,\n"
    "   not a coaching note. Just the words to speak (<25 words).\n"
    "Never become defensive. Never simply list facts in response to a challenge.\n"
    "Always ground the move in what would actually move this specific conversation forward."
)

CORE_INTELLIGENCE_INSTRUCTION = f"=== [CORE INTELLIGENCE — ALWAYS ACTIVE] ===\n{CORE_INSTRUCTIONS}"


# =========================================================================
# Deterministic "Respond Now or Wait" Gate (Code-Level Suppression)
# =========================================================================
TIMING_SENSITIVITY_THRESHOLDS: Dict[str, float] = {
    "Aggressive": 0.25,  # Jump in quickly on short gap
    "Normal": 0.45,      # Balanced enterprise standard
    "Relaxed": 0.85,     # Patient, waits for prospect to settle
}

DANGLING_CONNECTORS = {
    "and", "but", "or", "because", "so", "if", "that", "like",
    "when", "with", "to", "for", "as", "um", "uh", "i mean", "you know"
}

PASSIVE_ACKNOWLEDGMENT_WORDS = {
    "um", "uh", "yeah", "yes", "ok", "okay", "sure", "right",
    "mhm", "mm-hmm", "ah", "yep", "nope", "gotcha", "cool", "alright"
}

CRITICAL_SHORT_OBJECTIONS = {
    "why", "how", "who", "when", "what", "where", "no", "never", "cost",
    "price", "stop", "pass", "more"
}


def should_generate_now(
    utterance: str,
    is_final: bool,
    speech_final: bool,
    last_generation_time: float,
    current_time: float,
    cooldown_seconds: float = 4.0,
    timing_sensitivity: str = "Normal",
    min_characters: int = 6,
) -> tuple[bool, str]:
    """Deterministic, code-level suppression gate between STT transcription and LLM generation.

    Evaluates before RAG retrieval or prompt assembly:
    1. Minimum substance / short objection whitelist check
    2. Passive filler acknowledgment filter
    3. Mid-thought dangling connector / ellipsis pause check
    4. Speech endpoint / finality check (speech_final vs is_final)
    5. Playbook prompt cooldown threshold check

    Returns (should_generate: bool, reason: str).
    """
    clean = utterance.strip()
    if not clean or len(clean) < 2:
        return False, "insufficient_length"

    lowered = clean.lower()
    cleaned_tokens = [w.strip(".,!?\"'") for w in lowered.split() if w]
    first_clean_token = cleaned_tokens[0] if cleaned_tokens else ""

    # 1. Critical short objection check (e.g. "Why?", "No.", "How?", "Cost?", "Pass.")
    is_critical_short = (
        lowered.rstrip(".,!?") in CRITICAL_SHORT_OBJECTIONS
        or first_clean_token in CRITICAL_SHORT_OBJECTIONS
        or (clean.endswith("?") and len(first_clean_token) >= 2)
    )

    if len(clean) < min_characters and not is_critical_short:
        return False, "insufficient_length"

    # 2. Passive filler acknowledgment filter (e.g. "yeah, okay", "uh-huh", "mhm")
    if not is_critical_short:
        if lowered.rstrip(".,!?") in PASSIVE_ACKNOWLEDGMENT_WORDS or (
            cleaned_tokens and all(token in PASSIVE_ACKNOWLEDGMENT_WORDS for token in cleaned_tokens)
        ):
            return False, "passive_acknowledgment_filler"

    # 3. Mid-thought dangling connector check (e.g. "I guess my concern is so..." or trailing ellipsis)
    words = lowered.split()
    if words and words[-1] in DANGLING_CONNECTORS and not speech_final:
        return False, f"mid_thought_dangling_connector_'{words[-1]}'"

    if clean.endswith("...") and not speech_final:
        return False, "mid_thought_trailing_ellipsis"

    # 4. Finality & Speech Endpoint check (speech_final vs is_final)
    # Deepgram emits multiple is_final=True interim chunks while speech_final=False.
    # Suppress interim chunks until speech_final=True (actual speaker pause), unless terminated with a clear question.
    if not speech_final:
        if not is_final:
            return False, "interim_speech_in_progress"
        if not (clean.endswith("?") or clean.endswith("!")):
            return False, "awaiting_speech_final_pause"

    # 5. Playbook Prompt Cooldown check
    elapsed_since_last = current_time - last_generation_time
    if last_generation_time > 0 and elapsed_since_last < cooldown_seconds:
        return False, f"cooldown_active_{elapsed_since_last:.2f}s_of_{cooldown_seconds:.1f}s"

    return True, "ready_to_respond"


# =========================================================================
# Modular Section Formatters (Plain Text Stacking)
# =========================================================================
def lead_type_section(lead_type: str, lead_type_desc: str = "") -> str:
    """Format the lead type context paragraph."""
    desc_str = f"\n{lead_type_desc}" if lead_type_desc else ""
    return f"\n\n[LEAD TYPE: {lead_type}]{desc_str}"


def playbook_section(playbook_prompt_lens: str) -> str:
    """Format the active playbook methodology lens."""
    return f"\n\n{playbook_prompt_lens.strip()}"


def calibration_section(calibration_summary: str) -> str:
    """Format the sales rep calibration profile."""
    return f"\n\n[CALIBRATION PROFILE]\n{calibration_summary.strip()}"


def training_section(training_snippet: str) -> str:
    """Format matching RAG knowledge/company training evidence."""
    return f"\n\n[AI TRAINING / COMPANY KNOWLEDGE]:\n{training_snippet.strip()}"


def conversation_section(conversation_turns: Optional[List[Dict[str, str]]] = None, current_utterance: str = "") -> str:
    """Format conversation transcript so far and the latest utterance."""
    lines = ["\n\n[CONVERSATION SO FAR]"]
    if conversation_turns:
        for turn in conversation_turns:
            role = "Rep" if turn.get("role") in ("assistant", "salesperson") else "Prospect"
            content = turn.get("content", "").strip()
            if content:
                lines.append(f'{role}: "{content}"')

    if current_utterance:
        lines.append(f'Prospect just said: "{current_utterance}"')

    lines.append("\nCore Reasoning & Response: Produce the exact single spoken sentence for the rep (<25 words):")
    return "\n".join(lines)


def build_prompt(
    lead_type: Optional[str] = None,
    lead_type_desc: str = "",
    playbook_prompt_lens: Optional[str] = None,
    calibration_summary: Optional[str] = None,
    training_snippet: Optional[str] = None,
    conversation_turns: Optional[List[Dict[str, str]]] = None,
    current_utterance: str = "",
) -> str:
    """Build the complete, stacked text prompt starting with permanent Core."""
    prompt = CORE_INTELLIGENCE_INSTRUCTION  # Always the same, always included
    if lead_type:
        prompt += lead_type_section(lead_type, lead_type_desc)
    if playbook_prompt_lens:
        prompt += playbook_section(playbook_prompt_lens)
    if calibration_summary:
        prompt += calibration_section(calibration_summary)
    if training_snippet:
        prompt += training_section(training_snippet)
    prompt += conversation_section(conversation_turns, current_utterance)
    return prompt


@dataclass
class PreCallFolder:
    """Immutable pre-call context assembled once before call begins."""

    call_sid: str
    lead_type: str
    lead_type_desc: str
    playbook_id: str
    playbook_title: str
    playbook_prompt_lens: str
    calibration_profile: Dict[str, Any]
    calibration_summary_text: str
    prompt_cooldown_seconds: float = 4.0
    prompt_timing_sensitivity: str = "Normal"
    core_engine_name: str = "PitchProX Core Intelligence (Permanent)"
    salesman_id: Optional[str] = None
    created_at_monotonic: float = field(default_factory=monotonic)

    def should_respond_now(
        self,
        utterance: str,
        is_final: bool,
        speech_final: bool,
        last_generation_time: float,
        current_time: float,
    ) -> tuple[bool, str]:
        """Evaluate the deterministic 'Respond Now or Wait' code gate."""
        return should_generate_now(
            utterance=utterance,
            is_final=is_final,
            speech_final=speech_final,
            last_generation_time=last_generation_time,
            current_time=current_time,
            cooldown_seconds=self.prompt_cooldown_seconds,
            timing_sensitivity=self.prompt_timing_sensitivity,
        )

    def assemble_prompt(
        self,
        conversation_context: List[Dict[str, str]],
        current_utterance: str,
        rag_evidence: str = "",
    ) -> List[Dict[str, str]]:
        """Assembles the chat message list for LLM using the stacked prompt builder."""
        # 1. Base System Instruction: Core Brain FIRST & PERMANENT, then contextual modifiers
        system_instruction = (
            f"{CORE_INTELLIGENCE_INSTRUCTION}\n\n"
            f"[LEAD TYPE: {self.lead_type}]\n"
            f"{self.lead_type_desc}\n\n"
            f"{self.playbook_prompt_lens}\n\n"
            f"[CALIBRATION PROFILE]\n"
            f"{self.calibration_summary_text}"
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_instruction}
        ]

        # 2. Conversation Transcript History so far
        if conversation_context:
            for turn in conversation_context:
                role = "assistant" if turn.get("role") in ("assistant", "salesperson") else "user"
                content = turn.get("content", "").strip()
                if content:
                    messages.append({"role": role, "content": content})

        # 3. Latest Client Utterance + Optional RAG Knowledge + Core Execution Trigger
        user_message_parts = [
            f"[CONVERSATION STATE]",
            f"Prospect just said: \"{current_utterance}\"",
        ]
        if rag_evidence:
            user_message_parts.append(f"\n[AI TRAINING / COMPANY KNOWLEDGE]:\n{rag_evidence}")

        user_message_parts.append(
            f"\nCore Reasoning & Response: Produce the exact single spoken sentence applying the {self.playbook_title} lens for {self.lead_type}:"
        )
        user_message_content = "\n".join(user_message_parts)

        messages.append({"role": "user", "content": user_message_content})
        return messages

    def assemble_raw_text_prompt(
        self,
        conversation_context: List[Dict[str, str]],
        current_utterance: str,
        rag_evidence: str = "",
    ) -> str:
        """Assembles the entire prompt as one single stacked text block."""
        return build_prompt(
            lead_type=self.lead_type,
            lead_type_desc=self.lead_type_desc,
            playbook_prompt_lens=self.playbook_prompt_lens,
            calibration_summary=self.calibration_summary_text,
            training_snippet=rag_evidence,
            conversation_turns=conversation_context,
            current_utterance=current_utterance,
        )


# Global In-Memory Cache for Active Calls
CALL_FOLDERS: Dict[str, PreCallFolder] = {}


def load_latest_calibration_profile() -> tuple[Dict[str, Any], str]:
    """Load latest calibration report from disk or supply clean fallback."""
    latest_file = REPORTS_DIR / "latest_calibration_report.json"
    if latest_file.exists():
        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            wpm = data.get("wpm") or 145
            pacing_score = data.get("score_breakdown", {}).get("pacing", 80)
            summary = (
                f"Pacing: {wpm} WPM (Pacing Score: {pacing_score}/100) | "
                f"Tone: Natural & Confident | Delivery: Keep responses concise and avoid filler words."
            )
            return data, summary
        except Exception as e:
            LOGGER.warning("Could not read latest calibration report: %s", e)

    fallback = {
        "user_id": "sales_rep",
        "wpm": 140,
        "score_breakdown": {"pacing": 85, "word_choice": 90, "sentiment": 85},
    }
    return fallback, "Pacing: 140 WPM (Optimal Enterprise Cadence) | Delivery: Crisp, confident, zero fillers."


def build_pre_call_folder(
    call_sid: str,
    lead_type: Optional[str] = None,
    playbook_id: Optional[str] = None,
    salesman_id: Optional[str] = None,
    store: Optional[PlaybookStore] = None,
) -> PreCallFolder:
    """Build and cache the PreCallFolder for a specific call session."""
    if store is None:
        store = PlaybookStore()

    # 1. Resolve Lead Type
    resolved_lead_type = lead_type or "Expired Listings"
    desc = "Listings that recently expired without selling."
    for lt in STANDARDIZED_LEAD_TYPES:
        if lt["name"].lower() == resolved_lead_type.lower():
            resolved_lead_type = lt["name"]
            desc = lt["desc"]
            break

    # 2. Resolve Playbook
    playbook = None
    if playbook_id:
        playbook = store.get(playbook_id)

    if not playbook:
        all_pbs = store.list_all()
        if all_pbs:
            playbook = store.get(all_pbs[0]["id"])

    if not playbook:
        # Fallback to default
        from .playbook import PlaybookBasicInfo, PlaybookStyle
        playbook = Playbook(
            title="PitchProX Standard",
            basic_info=PlaybookBasicInfo(
                name="PitchProX Standard",
                industry="General Sales",
                lead_types=[resolved_lead_type],
                philosophy="Focus on value, diagnostic discovery, and clear commitment next steps.",
            ),
            style=PlaybookStyle(),
        )

    # 3. Serialize Playbook Lens & extract runtime settings
    playbook_lens = serialize_playbook_prompt_section(
        playbook,
        current_lead_type=resolved_lead_type,
    )

    runtime_cfg = getattr(playbook.style, "runtime_settings", None)
    cooldown = float(getattr(runtime_cfg, "prompt_cooldown_seconds", 4)) if runtime_cfg else 4.0
    timing_sensitivity = str(getattr(runtime_cfg, "prompt_timing_sensitivity", "Normal")) if runtime_cfg else "Normal"

    # 4. Calibration Profile
    calib_data, calib_summary = load_latest_calibration_profile()

    folder = PreCallFolder(
        call_sid=call_sid,
        lead_type=resolved_lead_type,
        lead_type_desc=desc,
        playbook_id=playbook.id,
        playbook_title=playbook.title,
        playbook_prompt_lens=playbook_lens,
        calibration_profile=calib_data,
        calibration_summary_text=calib_summary,
        prompt_cooldown_seconds=cooldown,
        prompt_timing_sensitivity=timing_sensitivity,
        core_engine_name="PitchProX Core Intelligence (Permanent)",
        salesman_id=salesman_id,
    )

    CALL_FOLDERS[call_sid] = folder
    LOGGER.info("Built and cached PreCallFolder for call %s (Core: Permanent, Cooldown: %ss, Sensitivity: %s, LeadType: %s, Playbook: %s)", call_sid, cooldown, timing_sensitivity, resolved_lead_type, playbook.title)
    return folder


def get_or_create_pre_call_folder(
    call_sid: str,
    lead_type: Optional[str] = None,
    playbook_id: Optional[str] = None,
    salesman_id: Optional[str] = None,
) -> PreCallFolder:
    """Retrieve existing cached folder or create on-demand."""
    if call_sid in CALL_FOLDERS:
        folder = CALL_FOLDERS[call_sid]
        # If user explicitly provided new lead_type or playbook_id, update folder
        if (lead_type and lead_type != folder.lead_type) or (playbook_id and playbook_id != folder.playbook_id):
            return build_pre_call_folder(call_sid, lead_type, playbook_id, salesman_id)
        return folder

    return build_pre_call_folder(call_sid, lead_type, playbook_id, salesman_id)
