"""Pre-Call Folder & Prompt Assembly Module.

Manages the immutable Pre-Call Folder created once before the call starts:
1. Lead Type (one of 16 standardized prospect types)
2. Active Playbook (Methodology lens, objection matrix, tone, rules)
3. Calibration Profile (Salesperson natural cadence, pacing, WPM, rhythm)

Provides rapid zero-allocation prompt assembly during live calls.
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
    salesman_id: Optional[str] = None
    created_at_monotonic: float = field(default_factory=monotonic)

    def assemble_prompt(
        self,
        conversation_context: List[Dict[str, str]],
        current_utterance: str,
        rag_evidence: str = "",
    ) -> List[Dict[str, str]]:
        """Assembles the single unified prompt message for the Groq LLM."""
        # 1. Base System Instruction with Pre-Call Folder
        system_instruction = (
            "You are a real-time live sales copilot teleprompter assisting the salesperson during a live customer call.\n"
            "Your objective is to output ONLY the exact, punchy first-person sentence (<25 words) the salesperson should say right now.\n"
            "Do NOT include meta-advice, filler, or quotes.\n\n"
            "=== 1. [PRE-CALL FOLDER: IMMUTABLE BASELINE] ===\n"
            f"- Prospect Lead Type: {self.lead_type} ({self.lead_type_desc})\n"
            f"- Salesperson Calibration Profile: {self.calibration_summary_text}\n\n"
            f"{self.playbook_prompt_lens}\n"
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

        # 3. Latest Client Utterance + Relevant RAG Knowledge
        knowledge_block = f"\n\n[VERIFIED COMPANY KNOWLEDGE]:\n{rag_evidence}" if rag_evidence else ""
        user_message_content = (
            f"Client just said: \"{current_utterance}\"{knowledge_block}\n\n"
            f"Respond directly as the salesperson applying the {self.playbook_title} lens for {self.lead_type}:"
        )

        messages.append({"role": "user", "content": user_message_content})
        return messages


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

    # 3. Serialize Playbook Lens
    playbook_lens = serialize_playbook_prompt_section(
        playbook,
        current_lead_type=resolved_lead_type,
    )

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
        salesman_id=salesman_id,
    )

    CALL_FOLDERS[call_sid] = folder
    LOGGER.info("Built and cached PreCallFolder for call %s (LeadType: %s, Playbook: %s)", call_sid, resolved_lead_type, playbook.title)
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
