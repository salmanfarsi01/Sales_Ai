"""Pre-Call Folder & Prompt Assembly Module.

Manages the immutable Pre-Call Folder created once before the call starts:
0. Core Intelligence Engine (Permanent Foundation — always active on every turn)
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


# =========================================================================
# 0. Core Intelligence Engine — Permanent Thinking Pattern (Fixed Text)
# =========================================================================
CORE_INTELLIGENCE_INSTRUCTION = (
    "=== [CORE INTELLIGENCE — ALWAYS ACTIVE] ===\n"
    "You are the permanent reasoning engine for a live sales call. On every turn:\n"
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
    core_engine_name: str = "PitchProX Core Intelligence (Permanent)"
    salesman_id: Optional[str] = None
    created_at_monotonic: float = field(default_factory=monotonic)

    def assemble_prompt(
        self,
        conversation_context: List[Dict[str, str]],
        current_utterance: str,
        rag_evidence: str = "",
    ) -> List[Dict[str, str]]:
        """Assembles the single unified prompt message for the Groq LLM."""
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
        core_engine_name="PitchProX Core Intelligence (Permanent)",
        salesman_id=salesman_id,
    )

    CALL_FOLDERS[call_sid] = folder
    LOGGER.info("Built and cached PreCallFolder for call %s (Core: Permanent, LeadType: %s, Playbook: %s)", call_sid, resolved_lead_type, playbook.title)
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
