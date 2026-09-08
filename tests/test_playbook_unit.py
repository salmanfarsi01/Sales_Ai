"""Unit tests for Custom Playbook Creator module (5-Step Wizard).

Verifies:
1. Pydantic models for Stage, BasicInfo, Style, Objections, ResponseSequence, and Playbook.
2. Playbook quality score calculation.
3. PlaybookStore persistence layer (save, get, list_all, delete).
4. FastAPI router endpoints (/templates/default, /list, /save, /{id}, /preview-prompt, delete).
5. Error cases and validation.
"""

import pytest
from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient

from copilot.playbook import (
    Playbook,
    PlaybookBasicInfo,
    PlaybookStyle,
    PlaybookStage,
    PlaybookObjection,
    ResponseSequenceStep,
    PlaybookStore,
    get_playbook_router,
    calculate_playbook_quality_score,
    DEFAULT_STAGES,
    DEFAULT_OBJECTIONS,
)


@pytest.fixture
def temp_store(tmp_path):
    """Fixture providing an isolated PlaybookStore using a temporary directory."""
    return PlaybookStore(storage_dir=tmp_path / "playbooks_test")


@pytest.fixture
def client(temp_store):
    """Fixture providing FastAPI test client connected to the isolated store."""
    app = FastAPI()
    router = get_playbook_router(temp_store)
    app.include_router(router)
    return TestClient(app)


def test_playbook_models_default_initialization():
    """Verify that default models initialize with valid default fields across 5 steps."""
    info = PlaybookBasicInfo(
        name="Daniel G. Method",
        short_description="My approach to building trust and closing more high-value deals.",
        industry="Real Estate",
        lead_types=["Expired, FSBO, Pre-Expired"],
        experience_level="Intermediate to Advanced",
        voice_phrases="Here's the thing\nBottom line",
        philosophy="1. Establish trust through diagnostic questioning. 2. Frame price in context of total value and net ROI.",
        goals=["Build stronger trust", "Increase appointments", "Close more deals"],
    )
    assert info.name == "Daniel G. Method"
    assert len(info.goals) == 3

    style = PlaybookStyle()
    assert style.overall_tone == "Confident"
    assert style.energy_level == "Medium"
    assert style.humor_level == "Light"

    pb = Playbook(basic_info=info, style=style)
    assert pb.id.startswith("pb_")
    assert pb.title == "Untitled Playbook"
    assert len(pb.stages) == 6
    assert len(pb.objections) == 8
    assert len(pb.response_sequence) == 4
    assert pb.stages[0].name == "Opening"
    assert pb.stages[0].icon == "hand"
    assert pb.objections[0].objection == "I don't want to pay commission."


def test_quality_score_calculation():
    """Verify the quality score calculation algorithm."""
    info = PlaybookBasicInfo(
        name="Daniel G. Method",
        short_description="My approach to building trust and closing more high-value deals.",
        industry="Real Estate",
        lead_types=["Expired, FSBO, Pre-Expired"],
        experience_level="Intermediate to Advanced",
        voice_phrases="Here's the thing\nBottom line",
        philosophy="1. Establish trust through diagnostic questioning. 2. Frame price in context of total value and net ROI.",
        goals=["Build stronger trust", "Increase appointments", "Close more deals"],
    )
    pb = Playbook(basic_info=info)
    score = calculate_playbook_quality_score(pb)
    assert score >= 90
    assert score <= 100


def test_playbook_store_save_get_and_list(temp_store):
    """Test saving a playbook, retrieving it, and listing all playbooks."""
    info = PlaybookBasicInfo(
        name="Enterprise Closer",
        industry="B2B Enterprise Services",
    )
    pb = Playbook(id="pb_test_123", basic_info=info)
    
    saved = temp_store.save(pb)
    assert saved.id == "pb_test_123"
    assert saved.title == "Enterprise Closer"
    assert saved.quality_score >= 20

    # Retrieve by ID
    retrieved = temp_store.get("pb_test_123")
    assert retrieved is not None
    assert retrieved.id == "pb_test_123"
    assert retrieved.basic_info.name == "Enterprise Closer"
    assert len(retrieved.stages) == 6
    assert len(retrieved.objections) == 8

    # List all
    all_pbs = temp_store.list_all()
    assert len(all_pbs) == 1
    assert all_pbs[0]["id"] == "pb_test_123"
    assert all_pbs[0]["title"] == "Enterprise Closer"
    assert all_pbs[0]["objections_count"] == 8


def test_playbook_store_delete(temp_store):
    """Test deleting a playbook."""
    info = PlaybookBasicInfo(name="Temporary Playbook")
    pb = Playbook(id="pb_delete_me", basic_info=info)
    temp_store.save(pb)

    assert temp_store.get("pb_delete_me") is not None
    deleted = temp_store.delete("pb_delete_me")
    assert deleted is True
    assert temp_store.get("pb_delete_me") is None
    assert temp_store.delete("pb_non_existent") is False


def test_api_templates_default(client):
    """Test the /api/playbook/templates/default endpoint."""
    res = client.get("/api/playbook/templates/default")
    assert res.status_code == 200
    data = res.json()
    assert "stages" in data
    assert len(data["stages"]) == 6
    assert "objections" in data
    assert len(data["objections"]) == 8
    assert "response_sequence" in data
    assert len(data["response_sequence"]) == 4
    assert "sample_voices" in data
    assert "industries" in data
    assert "default_tones" in data


def test_api_preview_prompt(client):
    """Test dynamic live coaching prompt preview generation."""
    payload = {
        "name": "Daniel",
        "overall_tone": "Confident",
        "communication_style": "Consultative",
        "sentence_style": "Short & Conversational",
        "formality": "Professional",
    }
    res = client.post("/api/playbook/preview-prompt", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "preview_text" in data
    assert "Hey Daniel" in data["preview_text"]
    assert "confident" in data["preview_text"]
    assert "consultative" in data["preview_text"]


def test_api_save_and_retrieve_playbook(client):
    """Test saving a new playbook via API with objections and getting it."""
    payload = {
        "basic_info": {
            "name": "Daniel G. Method",
            "short_description": "Convert inbound enterprise demos and expireds.",
            "industry": "Real Estate",
            "lead_types": ["Expired", "FSBO"],
            "experience_level": "Intermediate to Advanced",
            "voice_phrases": "Here's the thing",
            "philosophy": "Deliver value upfront and net more equity.",
            "goals": ["Build stronger trust", "Increase appointments", "Close more deals"],
        },
        "style": {
            "overall_tone": "Confident",
            "communication_style": "Consultative",
            "energy_level": "High",
            "sentence_style": "Short & Conversational",
            "formality": "Professional",
            "humor_level": "Light",
        },
        "stages": [
            {
                "id": "stg_1",
                "order": 1,
                "name": "Opening",
                "goal": "Grab attention in 15 seconds.",
                "description": "Establish immediate credibility.",
                "icon": "hand",
            }
        ],
        "objections": [
            {
                "id": "obj_custom_1",
                "objection": "I don't want to pay commission.",
                "category": "Pricing Objection",
                "lead_types": ["FSBO"],
                "response_style": "Value & ROI Focus",
                "ai_suggestion": "Highlight net proceeds.",
                "last_updated": "May 2, 2024",
            }
        ],
    }

    # Save
    res = client.post("/api/playbook/save", json=payload)
    assert res.status_code == 200
    save_data = res.json()
    assert save_data["status"] == "success"
    pb_id = save_data["playbook"]["id"]
    assert pb_id.startswith("pb_")
    assert save_data["playbook"]["title"] == "Daniel G. Method"

    # Get
    res_get = client.get(f"/api/playbook/{pb_id}")
    assert res_get.status_code == 200
    got_pb = res_get.json()["playbook"]
    assert got_pb["id"] == pb_id
    assert len(got_pb["objections"]) == 1
    assert got_pb["objections"][0]["objection"] == "I don't want to pay commission."

    # List
    res_list = client.get("/api/playbook/list")
    assert res_list.status_code == 200
    assert len(res_list.json()["playbooks"]) == 1
    assert res_list.json()["playbooks"][0]["objections_count"] == 1


def test_serialize_playbook_prompt_section():
    """Verify that structured playbook data gets serialized into the labeled prompt block."""
    from copilot.playbook import serialize_playbook_prompt_section
    info = PlaybookBasicInfo(
        name="Daniel G. Method",
        industry="Real Estate",
        lead_types=["Expired Listings", "FSBO (For Sale By Owner)"],
        philosophy="1. Establish trust. 2. Frame price as net ROI.",
        voice_phrases="Here's the thing\nBottom line",
    )
    pb = Playbook(title="Daniel G. Method", basic_info=info)

    prompt_section = serialize_playbook_prompt_section(
        pb,
        current_lead_type="Expired Listings",
        detected_objection="I don't want to pay commission."
    )

    assert "### ACTIVE PLAYBOOK METHODOLOGY LENS: DANIEL G. METHOD ###" in prompt_section
    assert "Coaching Style: Tone=Confident" in prompt_section
    assert "Objection Response Sequence: Acknowledge -> Validate -> Reframe -> Guide Forward" in prompt_section
    assert "Candidate Objection Match" in prompt_section
    assert "I don't want to pay commission." in prompt_section
    assert "Quantify" in prompt_section
    assert "Do / Don't Boundaries" in prompt_section


def test_semantic_objection_matching_paraphrased():
    """Verify semantic matching correctly maps real paraphrased spoken objections."""
    from copilot.playbook import match_objection_semantically, PlaybookObjection

    objections = [
        PlaybookObjection(
            id="obj_1",
            objection="I don't want to pay commission.",
            category=["Financial"],
            response_style="Quantify",
        ),
        PlaybookObjection(
            id="obj_2",
            objection="Not a good time to sell.",
            category=["Logistics"],
            response_style="Future Pace",
        ),
    ]

    # Spoken paraphrase that lacks substring containment
    spoken = "Why would I pay you guys a commission when I can sell it myself"
    match = match_objection_semantically(spoken, objections, threshold=0.75)
    assert match is not None
    matched_obj, confidence = match
    assert matched_obj.id == "obj_1"
    assert matched_obj.objection == "I don't want to pay commission."
    assert confidence >= 0.75


def test_objection_confidence_threshold_gating():
    """Verify that objections with similarity below the threshold are rejected."""
    from copilot.playbook import match_objection_semantically, PlaybookObjection

    objections = [
        PlaybookObjection(
            id="obj_1",
            objection="I don't want to pay commission.",
            category=["Financial"],
            response_style="Quantify",
        ),
    ]

    # Irrelevant or low-confidence query
    spoken = "The weather in Texas is nice this week"
    match = match_objection_semantically(spoken, objections, threshold=0.75)
    assert match is None


def test_trust_sensitivity_threshold_enforcement():
    """Verify trust_sensitivity_threshold injects recovery guidance when trust is below sensitivity point."""
    from copilot.playbook import serialize_playbook_prompt_section

    info = PlaybookBasicInfo(name="Trust Test Playbook")
    pb = Playbook(basic_info=info)
    pb.style.runtime_settings.trust_sensitivity_threshold = 0.70
    pb.style.trust_building_style = "Validation-First"

    # 1. Low trust score triggers note
    low_trust_prompt = serialize_playbook_prompt_section(pb, current_trust_score=0.45)
    assert "Trust Sensitivity Note" in low_trust_prompt
    assert "Validation-First" in low_trust_prompt
    assert "Increase the weight given to trust-recovery strategies" in low_trust_prompt
    assert "CRITICAL TRUST ALERT TRIGGERED" not in low_trust_prompt
    assert "MANDATORY" not in low_trust_prompt

    # 2. Healthy trust score does not trigger note
    healthy_trust_prompt = serialize_playbook_prompt_section(pb, current_trust_score=0.85)
    assert "Trust Sensitivity Note" not in healthy_trust_prompt


def test_multi_category_objection_support():
    """Verify category allows multiple values as a list and coerces legacy string."""
    from copilot.playbook import PlaybookObjection

    # 1. List input
    obj1 = PlaybookObjection(objection="Too much risk", category=["Risk", "Financial"])
    assert obj1.category == ["Risk", "Financial"]

    # 2. Comma-separated string coerced to list
    obj2 = PlaybookObjection(objection="Need to think", category="Logistics, Risk")
    assert obj2.category == ["Logistics", "Risk"]

    # 3. Single string coerced to list
    obj3 = PlaybookObjection(objection="Not interested", category="Relationship")
    assert obj3.category == ["Relationship"]


def test_spec_section_3_distinct_style_components():
    """Verify all 5 Spec Section 3 named style components are present and serialized."""
    from copilot.playbook import serialize_playbook_prompt_section

    info = PlaybookBasicInfo(name="Elite Methodology")
    style = PlaybookStyle(
        trust_building_style="Credibility-First",
        objection_tone="Assertive & Reframing",
        discovery_style="Problem-Agitation",
        closing_style="Direct Close",
        follow_up_style="Multi-Touch Education",
    )
    pb = Playbook(basic_info=info, style=style)
    prompt = serialize_playbook_prompt_section(pb)

    assert "Methodology Style Choices:" in prompt
    assert "Trust=Credibility-First" in prompt
    assert "Objection Tone=Assertive & Reframing" in prompt
    assert "Discovery=Problem-Agitation" in prompt
    assert "Close=Direct Close" in prompt
    assert "Follow-Up=Multi-Touch Education" in prompt


def test_readonly_premium_playbook_protection(temp_store, client):
    """Verify premium and read-only playbooks cannot be modified or deleted."""
    from fastapi import HTTPException

    info = PlaybookBasicInfo(name="PitchProX Elite Purchased Playbook")
    pb = Playbook(id="pb_premium_pack", basic_info=info, is_readonly=True, source="premium")
    temp_store.save(pb)

    # 1. Attempting to overwrite via store.save raises 403
    pb_mod = Playbook(id="pb_premium_pack", basic_info=info, title="Hacked Title")
    with pytest.raises(HTTPException) as exc_info:
        temp_store.save(pb_mod)
    assert exc_info.value.status_code == 403

    # 2. Attempting to overwrite via API raises 403
    api_res = client.post("/api/playbook/save", json={"id": "pb_premium_pack", "basic_info": {"name": "Hacked Title"}})
    assert api_res.status_code == 403
    assert "read-only methodology package" in api_res.json()["detail"]

    # 3. Attempting to delete via store.delete raises 403
    with pytest.raises(HTTPException) as exc_del:
        temp_store.delete("pb_premium_pack")
    assert exc_del.value.status_code == 403

    # 4. Attempting to delete via API raises 403
    api_del_res = client.delete("/api/playbook/pb_premium_pack")
    assert api_del_res.status_code == 403


def test_ai_stage_tag_generator(client):
    """Verify AI stage tag generator produces concise tags from goal text and reports engine metadata."""
    from copilot.playbook import classify_stage_tag_heuristic

    cases = [
        ("Set a positive tone and gain engagement.", "Opening", "Goal: Engage"),
        ("Understand their motivation and timeline.", "Discovery", "Goal: Understand"),
        ("Build credibility and differentiate yourself.", "Value Prop", "Goal: Differentiate"),
        ("Overcome doubt and strengthen confidence.", "Handle Objections", "Goal: Resolve"),
        ("Get a commitment to move forward.", "Close", "Goal: Commit"),
        ("Stay top of mind and convert later.", "Follow Up", "Goal: Nurture"),
    ]
    for goal, stage_name, expected_tag in cases:
        res = client.post("/api/playbook/generate-stage-tag", json={"goal": goal, "stage_name": stage_name})
        assert res.status_code == 200
        data = res.json()
        assert data["tag"] == expected_tag
        assert data["engine"] in ("ai_llm", "semantic_heuristic")

        # Directly verify deterministic heuristic classifier
        h_tag = classify_stage_tag_heuristic(goal=goal, stage_name=stage_name)
        assert h_tag == expected_tag


def test_ai_quality_evaluation_benchmark(client):
    """Verify quality evaluation endpoint returns genuine structural completeness and coverage metrics."""
    payload = {
        "basic_info": {
            "name": "Daniel G. Method",
            "short_description": "Complete top-tier sales playbook.",
            "industry": "Real Estate",
            "lead_types": ["Expired Listings", "FSBO"],
            "experience_level": "Intermediate to Advanced",
            "voice_phrases": "Here's the thing\nBottom line",
            "philosophy": "Deliver value upfront and guide the prospect to clear next steps.",
            "goals": ["Build stronger trust", "Increase appointments", "Close more deals"],
        }
    }
    res = client.post("/api/playbook/evaluate-quality", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "quality_score" in data
    assert "completeness_score" in data
    assert "rating" in data
    assert "methodology_assessment" in data
    assert "coverage_metrics" in data

    # No fabricated cohort comparisons
    assert "benchmark_comparison" not in data
    assert "cohort_comparison_metrics" not in data

    assert data["quality_score"] >= 90
    assert data["rating"] == "Excellent"
    assert "Structural quality assessment" in data["methodology_assessment"]

    metrics = data["coverage_metrics"]
    assert metrics["stages_count"] == 6
    assert metrics["objections_count"] == 8
    assert "category_diversity" in metrics
    assert "4 categories covered" in metrics["category_diversity"]


def test_evaluate_playbook_structural_quality_calculation():
    """Verify evaluate_playbook_structural_quality calculates scores from completeness, coverage, and balance."""
    from copilot.playbook import Playbook, PlaybookBasicInfo, evaluate_playbook_structural_quality

    info = PlaybookBasicInfo(name="Structural Evaluation Playbook")
    pb = Playbook(basic_info=info)

    res = evaluate_playbook_structural_quality(pb)
    assert 50 <= res["quality_score"] <= 98
    assert res["quality_score"] == res["structural_score"]
    assert res["rating"] in ("Excellent", "Strong", "Developing")
    assert "coverage_metrics" in res
    assert "benchmark_comparison" not in res
    assert "cohort_comparison_metrics" not in res


def test_simulate_runtime_prompt_with_draft(client):
    """Verify POST /api/playbook/simulate-runtime-prompt with an in-memory draft payload."""
    payload = {
        "playbook": {
            "title": "Live In-Memory Closer",
            "basic_info": {
                "name": "Live In-Memory Closer",
                "industry": "Real Estate",
                "lead_types": ["FSBO"],
            },
            "style": {
                "overall_tone": "Direct",
                "trust_building_style": "Validation-First",
                "objection_tone": "Assertive & Reframing",
                "runtime_settings": {
                    "objection_confidence_threshold": 0.70,
                    "trust_alert_threshold": 0.65,
                    "do_dont_boundaries": "DO NOT argue on pricing directly.",
                    "language_rules": "Always frame fees as investment.",
                },
            },
            "objections": [
                {
                    "id": "obj_test_1",
                    "objection": "I don't want to pay commission.",
                    "category": ["Financial", "Risk"],
                    "response_style": "Quantify",
                    "ai_suggestion": "Highlight net proceeds comparison.",
                }
            ],
        },
        "lead_type": "FSBO",
        "detected_objection": "Why should I pay a real estate commission?",
        "trust_score": 0.50,
    }

    res = client.post("/api/playbook/simulate-runtime-prompt", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert "prompt_section" in data
    prompt = data["prompt_section"]
    assert "### ACTIVE PLAYBOOK METHODOLOGY LENS: LIVE IN-MEMORY CLOSER ###" in prompt
    assert "Trust Sensitivity Note" in prompt
    assert "Candidate Objection Match" in prompt
    assert "DO NOT argue on pricing directly." in prompt
    assert "Always frame fees as investment." in prompt
    assert "I don't want to pay commission." in prompt

    # Introspection flags
    assert data["trust_sensitivity_triggered"] is True
    assert data["trust_alert_triggered"] is True
    assert data["threshold_met"] is True
    assert data["confidence"] >= 0.70
    assert data["match_info"] is not None
    assert data["match_info"]["matched_objection"] == "I don't want to pay commission."
    assert data["match_info"]["note"] == "lexical similarity only, not final relevance"
    assert "Financial" in data["match_info"]["category"]
    assert "Risk" in data["match_info"]["category"]

    # Display summary
    assert "display_summary" in data
    assert "summary_source" in data
    assert "Hey Live," in data["display_summary"] or "Hey there," in data["display_summary"]
    assert data["summary_source"] in ("ai_llm", "template_fallback")


def test_simulate_runtime_prompt_with_saved_id(client, temp_store):
    """Verify POST /api/playbook/simulate-runtime-prompt using saved playbook ID with healthy trust."""
    info = PlaybookBasicInfo(name="Stored Architecture Playbook")
    pb = Playbook(id="pb_saved_sim", title="Stored Architecture Playbook", basic_info=info)
    pb.style.runtime_settings.trust_sensitivity_threshold = 0.60
    temp_store.save(pb)

    sim_payload = {
        "playbook_id": "pb_saved_sim",
        "trust_score": 0.88,
        "detected_objection": "Not interested right now.",
    }
    res = client.post("/api/playbook/simulate-runtime-prompt", json=sim_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["playbook_id"] == "pb_saved_sim"
    assert data["trust_sensitivity_triggered"] is False
    assert data["trust_alert_triggered"] is False
    assert "Trust Sensitivity Note" not in data["prompt_section"]
    assert "CRITICAL TRUST ALERT TRIGGERED" not in data["prompt_section"]
    assert "display_summary" in data
    assert "Hey Stored," in data["display_summary"] or "Hey there," in data["display_summary"]


def test_get_runtime_prompt_introspection_endpoint(client, temp_store):
    """Verify GET /api/playbook/{id}/runtime-prompt returns full introspection metadata."""
    info = PlaybookBasicInfo(name="Introspection Engine Playbook")
    pb = Playbook(id="pb_intro_test", title="Introspection Engine Playbook", basic_info=info)
    temp_store.save(pb)

    res = client.get(
        "/api/playbook/pb_intro_test/runtime-prompt"
        "?lead_type=Expired&detected_objection=I%20don't%20want%20to%20pay%20commission&trust_score=0.4"
    )
    assert res.status_code == 200
    data = res.json()
    assert data["playbook_id"] == "pb_intro_test"
    assert data["trust_score"] == 0.4
    assert data["trust_sensitivity_triggered"] is True
    assert data["trust_alert_triggered"] is True
    assert data["match_info"] is not None
    assert data["match_info"]["note"] == "lexical similarity only, not final relevance"
    assert data["threshold_met"] is True
    assert "ACTIVE PLAYBOOK METHODOLOGY LENS" in data["prompt_section"]


def test_runtime_settings_and_section3_persistence_in_api(client):
    """Verify that runtime_settings and Section 3 style dimensions survive API save and get cycles."""
    payload = {
        "basic_info": {
            "name": "Full Spec Playbook",
            "industry": "Consulting",
        },
        "style": {
            "overall_tone": "Direct",
            "trust_building_style": "Diagnostic-Authority",
            "objection_tone": "Empathetic & Curious",
            "discovery_style": "Solution-Led",
            "closing_style": "Assumptive Close",
            "follow_up_style": "Rapid Action",
            "runtime_settings": {
                "prompt_cooldown_seconds": 7.0,
                "objection_confidence_threshold": 0.82,
                "trust_alert_threshold": 0.68,
                "prompt_timing": "Proactive",
                "emotional_sensitivity": "High",
                "do_dont_boundaries": "DO NOT quote rates without diagnosing need.",
                "language_rules": "Always speak in terms of net client ROI.",
            },
        },
    }

    save_res = client.post("/api/playbook/save", json=payload)
    assert save_res.status_code == 200
    pb_id = save_res.json()["playbook"]["id"]

    get_res = client.get(f"/api/playbook/{pb_id}")
    assert get_res.status_code == 200
    pb_data = get_res.json()["playbook"]
    style = pb_data["style"]

    assert style["trust_building_style"] == "Diagnostic-Authority"
    assert style["objection_tone"] == "Empathetic & Curious"
    assert style["discovery_style"] == "Solution-Led"
    assert style["closing_style"] == "Assumptive Close"
    assert style["follow_up_style"] == "Rapid Action"

    rs = style["runtime_settings"]
    assert rs["prompt_cooldown_seconds"] == 7
    assert rs["objection_confidence_threshold"] == 0.82
    assert rs["trust_sensitivity_threshold"] == 0.68
    assert rs["prompt_timing_sensitivity"] == "Proactive"
    assert rs["emotional_sensitivity"] == "High"
    assert any("DO NOT quote rates" in b for b in rs["do_dont_boundaries"])
    assert any("Always speak in terms of net client ROI." in r for r in rs["language_rules"])


@pytest.mark.asyncio
async def test_ai_quality_evaluation_with_mocked_llm(monkeypatch):
    """Verify that evaluate_playbook_quality_benchmark blends 40% structural and 60% LLM score."""
    from copilot.playbook import Playbook, PlaybookBasicInfo, evaluate_playbook_quality_benchmark
    import copilot.playbook as pb_module

    info = PlaybookBasicInfo(
        name="AI Quality Test Playbook",
        industry="Real Estate",
        philosophy="Diagnostic questioning first.",
    )
    pb = Playbook(basic_info=info)

    mock_llm_response = {
        "llm_quality_score": 90,
        "strengths": ["Clear stage progression", "Structured philosophy"],
        "gaps": ["Consider adding logistics objections"],
        "coherence_feedback": "Highly coherent strategy progressing naturally to close.",
    }

    async def mock_ai_eval(playbook):
        return mock_llm_response

    monkeypatch.setattr(pb_module, "ai_evaluate_playbook_quality", mock_ai_eval)

    eval_res = await evaluate_playbook_quality_benchmark(pb)

    assert eval_res["quality_source"] == "ai_llm"
    assert eval_res["llm_evaluation"] == mock_llm_response
    assert "strengths" in eval_res
    assert eval_res["summary"] == "Compared against internal baseline heuristics and other playbooks created in this workspace"
    assert eval_res["methodology_assessment"] == "Highly coherent strategy progressing naturally to close."

    # Check the 40% structural / 60% LLM formula
    structural_score = eval_res["structural_score"]
    expected_final = int(round((structural_score * 0.4) + (90 * 0.6)))
    assert eval_res["quality_score"] == expected_final


@pytest.mark.asyncio
async def test_ai_quality_evaluation_fallback_on_llm_failure(monkeypatch):
    """Verify graceful fallback to structural estimate when LLM is unavailable or fails."""
    from copilot.playbook import Playbook, PlaybookBasicInfo, evaluate_playbook_quality_benchmark
    import copilot.playbook as pb_module

    info = PlaybookBasicInfo(
        name="Fallback Quality Test Playbook",
        industry="Real Estate",
    )
    pb = Playbook(basic_info=info)

    async def mock_ai_eval_fail(playbook):
        return None

    monkeypatch.setattr(pb_module, "ai_evaluate_playbook_quality", mock_ai_eval_fail)

    eval_res = await evaluate_playbook_quality_benchmark(pb)

    assert eval_res["quality_source"] == "structural_fallback"
    assert eval_res["llm_evaluation"] is None
    assert eval_res["quality_score"] == eval_res["structural_score"]
    assert eval_res["quality_score"] >= 50


def test_save_endpoint_persists_llm_quality_score_end_to_end(client, temp_store, monkeypatch):
    """Verify that calling /api/playbook/save triggers the LLM quality evaluation end-to-end,
    correctly blends the score via AwaitablePlaybook.__await__, returns the AI score in the API response,
    and updates the persisted JSON file on disk.
    """
    import copilot.playbook as pb_module

    mock_llm_result = {
        "llm_quality_score": 96,
        "strengths": ["Outstanding stage alignment", "Crisp objection reframes"],
        "gaps": ["None identified"],
        "coherence_feedback": "Masterclass methodology that hangs together into an actionable sales system.",
    }

    async def mock_ai_eval(pb):
        return mock_llm_result

    monkeypatch.setattr(pb_module, "ai_evaluate_playbook_quality", mock_ai_eval)

    payload = {
        "basic_info": {
            "name": "End-to-End LLM Persisted Playbook",
            "industry": "Real Estate",
            "philosophy": "Diagnostic inquiry with rapid value positioning.",
        },
        "style": {
            "overall_tone": "Confident",
            "communication_style": "Consultative",
        }
    }

    # 1. Post to /api/playbook/save
    res = client.post("/api/playbook/save", json=payload)
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["status"] == "success"

    saved_pb = data["playbook"]
    saved_id = saved_pb["id"]
    feedback = saved_pb["quality_feedback"]

    # 2. Verify API response contains the AI-evaluated score and feedback
    assert feedback["quality_source"] == "ai_llm"
    assert feedback["methodology_assessment"] == mock_llm_result["coherence_feedback"]
    assert feedback["strengths"] == mock_llm_result["strengths"]
    
    structural_score = feedback["structural_score"]
    expected_blended = int(round((structural_score * 0.4) + (96 * 0.6)))
    assert saved_pb["quality_score"] == expected_blended
    assert feedback["quality_score"] == expected_blended

    # 3. Verify disk persistence: the file on disk MUST have the updated LLM evaluation
    persisted = temp_store.get(saved_id)
    assert persisted is not None
    assert persisted.quality_score == expected_blended
    assert persisted.quality_feedback.get("quality_source") == "ai_llm"
    assert persisted.quality_feedback.get("methodology_assessment") == mock_llm_result["coherence_feedback"]


@pytest.mark.asyncio
async def test_awaitable_playbook_dual_mode_and_private_attr(temp_store, monkeypatch):
    """Verify that AwaitablePlaybook:
    1. Functions synchronously without awaiting (immediate disk persistence with structural score).
    2. Correctly preserves _store as a Pydantic PrivateAttr across model initialization.
    3. Runs __await__ properly when awaited, enriching both the returned object and disk file with LLM feedback.
    """
    from copilot.playbook import Playbook, PlaybookBasicInfo, AwaitablePlaybook
    import copilot.playbook as pb_module

    mock_llm_result = {
        "llm_quality_score": 92,
        "strengths": ["Strong narrative flow"],
        "gaps": [],
        "coherence_feedback": "Seamless customer progression.",
    }

    async def mock_ai_eval(pb):
        return mock_llm_result

    monkeypatch.setattr(pb_module, "ai_evaluate_playbook_quality", mock_ai_eval)

    info = PlaybookBasicInfo(name="Dual Mode Test Playbook", industry="SaaS / Software")
    pb = Playbook(basic_info=info)

    # 1. Synchronous save (without await)
    sync_saved = temp_store.save(pb)
    assert isinstance(sync_saved, Playbook)
    assert isinstance(sync_saved, AwaitablePlaybook)
    # Verify _store PrivateAttr is preserved and points to temp_store
    assert sync_saved._store is temp_store
    # Verify synchronous score is structural
    assert sync_saved.quality_feedback.get("quality_source") == "structural_fallback"
    initial_score = sync_saved.quality_score

    # 2. Awaiting the already-created awaitable
    async_enriched = await sync_saved
    assert async_enriched is sync_saved
    assert async_enriched.quality_feedback.get("quality_source") == "ai_llm"
    expected_blended = int(round((initial_score * 0.4) + (92 * 0.6)))
    assert async_enriched.quality_score == expected_blended

    # 3. Direct async save: await temp_store.save(...)
    pb2 = Playbook(basic_info=PlaybookBasicInfo(name="Direct Async Save", industry="Real Estate"))
    awaited_saved = await temp_store.save(pb2)
    assert awaited_saved._store is temp_store
    assert awaited_saved.quality_feedback.get("quality_source") == "ai_llm"
    assert awaited_saved.quality_feedback.get("methodology_assessment") == "Seamless customer progression."

    # Check disk persistence for pb2
    from_disk = temp_store.get(pb2.id)
    assert from_disk.quality_score == awaited_saved.quality_score
    assert from_disk.quality_feedback.get("quality_source") == "ai_llm"


@pytest.mark.asyncio
async def test_ai_summarize_playbook_voice_mocked_and_fallback(monkeypatch):
    """Verify ai_summarize_playbook_voice with mocked LLM and fallback path."""
    import sys
    from copilot.playbook import Playbook, PlaybookBasicInfo, ai_summarize_playbook_voice

    pb = Playbook(basic_info=PlaybookBasicInfo(name="Salman Farsi"))
    full_prompt = "### ACTIVE PLAYBOOK METHODOLOGY LENS: SALMAN FARSI ###\n- Objection Tone: Empathetic & Resilient"

    # 1. Fallback when no API keys
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    res_none = await ai_summarize_playbook_voice(pb, full_prompt)
    assert res_none is None

    # 2. Mocked LLM response
    mock_summary = (
        "Hey Salman, I've got you.\n"
        "I'll keep things confident, consultative, and short — leading with empathy when objections come up.\n"
        "Let's build conversations people actually want to keep having."
    )
    class DummyChoice:
        message = type("Msg", (), {"content": mock_summary})()
    class DummyResp:
        choices = [DummyChoice()]
    class DummyCompletions:
        def create(self, **kwargs):
            return DummyResp()
    class DummyChat:
        completions = DummyCompletions()
    class DummyGroq:
        def __init__(self, **kwargs):
            self.chat = DummyChat()

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_key")
    dummy_module = type("dummy_groq", (), {"Groq": DummyGroq})
    monkeypatch.setitem(sys.modules, "groq", dummy_module)

    res_mock = await ai_summarize_playbook_voice(pb, full_prompt)
    assert res_mock == mock_summary
    assert "Hey Salman" in res_mock


def test_client_feedback_methodology_lens_points_1_to_5():
    """Verify all 5 client feedback items:
    1. Trust style removes 'MANDATORY' and 'CRITICAL TRUST ALERT TRIGGERED'.
    2. trust_alert_threshold renamed to trust_sensitivity_threshold with backwards-compat alias.
    3. Objection match labeled as 'Candidate Objection Match — Lexical Similarity' and match_info note.
    4. Operational constraint explicitly preserves Core override authority.
    5. Adaptive emotional sensitivity clarifies call-only scope and injects scoped note.
    """
    from copilot.playbook import (
        Playbook,
        PlaybookBasicInfo,
        PlaybookObjection,
        PlaybookRuntimeSettings,
        serialize_playbook_prompt_section,
    )

    # Point 2: Backwards compatibility of alias in PlaybookRuntimeSettings
    legacy_settings = PlaybookRuntimeSettings.model_validate({
        "trust_alert_threshold": 0.62,
        "prompt_timing": "Aggressive",
    })
    assert legacy_settings.trust_sensitivity_threshold == 0.62
    assert legacy_settings.trust_alert_threshold == 0.62
    assert legacy_settings.prompt_timing_sensitivity == "Aggressive"

    # Property setter works
    legacy_settings.trust_alert_threshold = 0.55
    assert legacy_settings.trust_sensitivity_threshold == 0.55

    # Point 5: Description check
    field_desc = PlaybookRuntimeSettings.model_fields["emotional_sensitivity"].description
    assert "Adaptive adjusts emphasis to live in-call signals only" in field_desc

    # Setup playbook for prompt serialization
    pb = Playbook(
        title="Lens Alignment Playbook",
        basic_info=PlaybookBasicInfo(name="Lens Alignment Playbook"),
        objections=[
            PlaybookObjection(
                id="obj_cand_1",
                objection="Why would I pay you guys a commission when I can sell it myself",
                category=["Financial"],
                response_style="Quantify",
                ai_suggestion="Focus on net walkaway dollar amounts.",
            )
        ],
    )
    pb.style.trust_building_style = "Validation-First"
    pb.style.runtime_settings.trust_sensitivity_threshold = 0.70
    pb.style.runtime_settings.emotional_sensitivity = "Adaptive"

    # Compile prompt with low trust and objection detected
    prompt = serialize_playbook_prompt_section(
        pb,
        detected_objection="Why would I pay you a commission when I can sell myself?",
        current_trust_score=0.45,
    )

    # Point 1 & 2: Trust Sensitivity Note, no "MANDATORY", no "CRITICAL TRUST ALERT"
    assert "Trust Sensitivity Note (Trust: 45% below sensitivity point 70%)" in prompt
    assert "CRITICAL TRUST ALERT TRIGGERED" not in prompt
    assert "MANDATORY" not in prompt
    assert "Increase the weight given to trust-recovery strategies" in prompt
    assert "preferred methodology: 'Validation-First'" in prompt
    assert "Trust Sensitivity=70%" in prompt

    # Point 3: Candidate Objection Match with Lexical Similarity note
    assert "[Candidate Objection Match — Lexical Similarity:" in prompt
    assert "(reference signal only; final relevance should be determined using full conversational context, objection recurrence, and lead type)]" in prompt
    assert "Suggested Response Style (if Core confirms relevance): Quantify" in prompt
    assert "Reference Strategy Tip: Focus on net walkaway dollar amounts." in prompt
    assert "[Matched Objection Rule" not in prompt

    # Point 4: Explicit override constraint
    expected_constraint = (
        "- Runtime Operational Constraint: This methodology lens guides HOW a response is formulated and offers "
        "suggested response styles and strategic preferences. The Core retains full authority to select a "
        "different response style (Clarify, Validate, De-Risk, Reframe, etc.) if current conversational evidence "
        "indicates a better strategic fit. Never invent unverified facts; always output exactly ONE concise "
        "prompt (<25 words) for the salesperson."
    )
    assert expected_constraint in prompt

    # Point 5: Adaptive scoped note
    expected_adaptive_note = (
        "- Note: Adaptive emotional sensitivity applies to live signals within THIS call only. "
        "It does not create any permanent change to future calls, other users, or stored methodology."
    )
    assert expected_adaptive_note in prompt

    # Verify that non-Adaptive does not include the scoped note
    pb.style.runtime_settings.emotional_sensitivity = "Normal"
    normal_prompt = serialize_playbook_prompt_section(pb, current_trust_score=0.45)
    assert expected_adaptive_note not in normal_prompt


def test_point_6_no_fabricated_benchmarks():
    """Verify Point 6: complete removal of fabricated benchmark data structure and cohort functions."""
    import copilot.playbook as pb_mod

    # Step 1 & 2: Fabricated structures and helper functions must not exist
    assert not hasattr(pb_mod, "INDUSTRY_METHODOLOGY_BENCHMARKS")
    assert not hasattr(pb_mod, "_norm_cdf")
    assert not hasattr(pb_mod, "get_cohort_benchmark_stats")

    # Step 3: evaluate_playbook_structural_quality returns structural completeness with no claimed cohort
    info = pb_mod.PlaybookBasicInfo(name="Unbenchmarked Playbook", industry="Real Estate")
    pb = pb_mod.Playbook(basic_info=info)

    eval_res = pb_mod.evaluate_playbook_structural_quality(pb)
    assert "benchmark_comparison" not in eval_res
    assert "cohort_comparison_metrics" not in eval_res
    assert eval_res["quality_source"] == "structural_fallback"
    assert "quality_score" in eval_res
    assert "completeness_score" in eval_res
    assert "coverage_metrics" in eval_res
    assert eval_res["coverage_metrics"]["category_diversity"] == "4 of 4 categories covered"

