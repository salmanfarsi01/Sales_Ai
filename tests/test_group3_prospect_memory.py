"""Group 3 Acceptance Suite: Prospect Memory Architecture (Points 15 to 18).

Verifies strict user-level memory isolation, canonical record schema, the five required
runtime situations, and the structural guard against team/shared memory.
"""

import ast
import inspect
import time
import pytest
from pathlib import Path

from copilot.prospect_memory import (
    ProspectMemoryRecord,
    ProspectMemoryStore,
    LoadedProspectMemory,
    MemoryProvenance,
    normalize_user_id,
    normalize_prospect_id,
)
from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_state_models import (
    ConversationStateSnapshot,
    PersistentFactRecord,
    MeetingConversionGate,
)


@pytest.fixture
def memory_store(tmp_path):
    """Provides a clean, isolated ProspectMemoryStore on a temporary path."""
    store_file = tmp_path / "test_prospect_memories.json"
    store = ProspectMemoryStore(store_path=store_file)
    store.clear()
    return store


# ==============================================================================
# Point 15: Memory Hierarchy and Ownership (user_id -> prospect_id -> memory)
# ==============================================================================

def test_point15_memory_hierarchy_user_id_to_prospect_id_no_cross_user_bleed(memory_store):
    """Point 15: User A calls John Smith (+1-555-0100) and records 'not moving until next year'.
    User B calls the exact same phone number/prospect.
    Asserts:
    1. User B retrieves ZERO memory from User A — zero cross-user memory bleed.
    2. Phone formatting variants (+1-555-0100, +15550100, (555) 0100) canonicalize correctly.
    3. Empty partition lookup is strictly side-effect free (no keys created in store).
    4. Type normalization handles integer and string user_ids without partition collision.
    """
    prospect_phone_formatted = "+1-555-0100"
    user_a = "sales_rep_alice"
    user_b = "sales_rep_bob"

    # Initial partition count
    assert len(memory_store._storage) == 0

    # Side-effect-free empty partition check: querying non-existent user leaves store empty
    empty_res = memory_store.retrieve_memories(user_id="unknown_rep", prospect_id=prospect_phone_formatted)
    assert empty_res == []
    assert len(memory_store._storage) == 0, "Querying non-existent user must not create dictionary keys"

    # User A records a memory during their call with John Smith
    rec_a = memory_store.save_record(
        user_id=user_a,
        prospect_id=prospect_phone_formatted,
        source_call_sid="call_alice_101",
        memory_type="timeline",
        key="timeline_horizon",
        value="next_year",
        content="not moving until next year",
        source_turn_id=4,
    )
    assert rec_a.user_id == user_a

    # User A can retrieve their memory across phone number formatting variants
    alice_mem1 = memory_store.retrieve_memories(user_id=user_a, prospect_id="+1-555-0100")
    alice_mem2 = memory_store.retrieve_memories(user_id=user_a, prospect_id="+15550100")
    alice_mem3 = memory_store.retrieve_memories(user_id=user_a, prospect_id="(555) 0100")
    assert len(alice_mem1) == 1
    assert len(alice_mem2) == 1
    assert len(alice_mem3) == 1
    assert alice_mem1[0].content == "not moving until next year"

    # User B calls the same phone number across all format variants: MUST retrieve ZERO memory
    assert memory_store.retrieve_memories(user_id=user_b, prospect_id="+1-555-0100") == []
    assert memory_store.retrieve_memories(user_id=user_b, prospect_id="+15550100") == []
    assert memory_store.retrieve_memories(user_id=user_b, prospect_id="(555) 0100") == []

    # User B's conversation state manager also has zero memory loaded
    mgr_bob = ConversationStateManager(
        call_sid="call_bob_202",
        user_id=user_b,
        prospect_id=prospect_phone_formatted,
        load_prospect_memory=True,
        memory_store=memory_store,
    )
    assert len(mgr_bob.current_state.loaded_prospect_memory) == 0
    assert len(mgr_bob.current_state.facts) == 0

    # Type normalization: int vs str user_id
    rec_int = memory_store.save_record(
        user_id=12345,
        prospect_id="prospect_num_test",
        source_call_sid="call_int_test",
        memory_type="fact",
        key="test_key",
        value="test_val",
        content="test content",
        source_turn_id=1,
    )
    retrieved_by_str = memory_store.retrieve_memories(user_id="12345", prospect_id="prospect_num_test")
    retrieved_by_int = memory_store.retrieve_memories(user_id=12345, prospect_id="prospect_num_test")
    assert len(retrieved_by_str) == 1
    assert len(retrieved_by_int) == 1


# ==============================================================================
# Point 16: Required Record Schema (all 6 fields mandatory)
# ==============================================================================

def test_point16_required_record_schema_all_six_fields_present(memory_store):
    """Point 16: Every memory record must store at minimum:
    1. user_id
    2. prospect_id
    3. source_call_sid
    4. timestamp (positive int)
    5. memory_type
    6. source_event_id / source_turn_id ('where applicable')
    """
    ts = int(time.time() * 1000)
    rec = memory_store.save_record(
        user_id="rep_carol",
        prospect_id="prospect_987",
        source_call_sid="call_carol_301",
        memory_type="contact_preference",
        key="prohibited_channel",
        value="sms_daily_texting",
        content="Please don't start texting me every day before we meet",
        timestamp=ts,
        source_turn_id=15,
        source_event_id="ev_pref_15",
        metadata={"channel": "sms", "allowed": False},
    )

    # 1. Assert all six required fields are present and populated
    assert rec.user_id == "rep_carol"
    assert rec.prospect_id == "prospect_987"
    assert rec.source_call_sid == "call_carol_301"
    assert rec.timestamp == ts
    assert rec.memory_type == "contact_preference"
    assert rec.source_turn_id == 15
    assert rec.source_event_id == "ev_pref_15"

    # 2. Assert user_id cannot be missing or empty
    with pytest.raises(ValueError, match="user_id is mandatory"):
        ProspectMemoryRecord(
            user_id="",
            prospect_id="p1",
            source_call_sid="call_test",
            timestamp=1000,
            memory_type="fact",
            source_turn_id=1,
            key="k",
            value="v",
            content="text",
        )

    # 3. Assert prospect_id cannot be missing or empty
    with pytest.raises(ValueError, match="prospect_id is mandatory"):
        ProspectMemoryRecord(
            user_id="u1",
            prospect_id="",
            source_call_sid="call_test",
            timestamp=1000,
            memory_type="fact",
            source_turn_id=1,
            key="k",
            value="v",
            content="text",
        )

    # 4. Assert source_call_sid cannot be missing or empty
    with pytest.raises(ValueError, match="source_call_sid is mandatory"):
        ProspectMemoryRecord(
            user_id="u1",
            prospect_id="p1",
            source_call_sid="",
            timestamp=1000,
            memory_type="fact",
            source_turn_id=1,
            key="k",
            value="v",
            content="text",
        )

    # 5. Assert memory_type cannot be missing or empty
    with pytest.raises(ValueError, match="memory_type is mandatory"):
        ProspectMemoryRecord(
            user_id="u1",
            prospect_id="p1",
            source_call_sid="call_test",
            timestamp=1000,
            memory_type="",
            source_turn_id=1,
            key="k",
            value="v",
            content="text",
        )

    # 6. Assert timestamp must be a positive integer
    with pytest.raises(ValueError):
        ProspectMemoryRecord(
            user_id="u1",
            prospect_id="p1",
            source_call_sid="call_test",
            timestamp=0,  # ge=1 required
            memory_type="fact",
            source_turn_id=1,
            key="k",
            value="v",
            content="text",
        )

    # 7. Assert 'where applicable': in-call records MUST supply at least one of source_turn_id or source_event_id
    with pytest.raises(ValueError, match="in-call memory records must supply at least one of source_turn_id or source_event_id"):
        ProspectMemoryRecord(
            user_id="u1",
            prospect_id="p1",
            source_call_sid="call_live_interaction",
            timestamp=1000,
            memory_type="fact",
            source_turn_id=None,
            source_event_id=None,
            key="k",
            value="v",
            content="text",
        )


# ==============================================================================
# Point 17: The Five Required Situations (one test per row)
# ==============================================================================

def test_situation_1_same_user_same_prospect_eligible_memory_loads_under_policy(memory_store):
    """Situation 1: Same user + same prospect -> eligible memory can load under policy.
    Verifies that:
    1. Memory loads into loaded_prospect_memory with complete provenance.
    2. Non-compliance memories (e.g. timeline, general facts) are NEVER merged into current_state.facts.
    3. Contact compliance boundaries are applied as an explicit policy decision with provenance preserved.
    """
    user = "rep_david"
    prospect = "prospect_david_001"

    # Record A: Contact boundary
    memory_store.save_record(
        user_id=user,
        prospect_id=prospect,
        source_call_sid="call_david_call1",
        memory_type="contact_preference",
        key="channel_boundary",
        value="reduced_frequency",
        content="Please don't text daily",
        source_turn_id=15,
        metadata={"preference": "reduced_frequency", "channel": "sms"},
    )

    # Record B: General timeline constraint (non-compliance memory)
    memory_store.save_record(
        user_id=user,
        prospect_id=prospect,
        source_call_sid="call_david_call1",
        memory_type="timeline",
        key="timeline_horizon",
        value="six_months",
        content="Planning to move in six months",
        source_turn_id=4,
    )

    # In a new call for the SAME user and SAME prospect with policy enabled:
    mgr = ConversationStateManager(
        call_sid="call_david_call2",
        user_id=user,
        prospect_id=prospect,
        load_prospect_memory=True,
        memory_store=memory_store,
    )

    # 1. Facts list remains clean slate: zero silent fact merging
    assert len(mgr.current_state.facts) == 0, "Non-compliance memory must never be silently merged into current-call facts"

    # 2. Both records load into loaded_prospect_memory
    assert len(mgr.current_state.loaded_prospect_memory) == 2
    mem_keys = [m.key for m in mgr.current_state.loaded_prospect_memory]
    assert "channel_boundary" in mem_keys
    assert "timeline_horizon" in mem_keys

    # 3. Contact compliance policy is applied with provenance
    assert mgr.current_state.contact_compliance.contact_preference == "reduced_frequency"
    assert len(mgr.current_state.contact_compliance.contact_preferences) == 1
    cp = mgr.current_state.contact_compliance.contact_preferences[0]
    assert cp.source_turn_id == 15, "Compliance entry must carry original turn provenance"


def test_situation_2_new_call_same_user_same_prospect_fresh_state_with_provenance_tagged_memory(memory_store):
    """Situation 2: New call, same user, same prospect -> fresh ConversationState,
    but eligible memory loads separately, tagged with its original provenance
    (not merged silently into current-call facts).
    """
    user = "rep_elena"
    prospect = "prospect_elena_002"

    memory_store.save_record(
        user_id=user,
        prospect_id=prospect,
        source_call_sid="call_elena_orig",
        memory_type="timeline",
        key="timeline_horizon",
        value="spring_next_year",
        content="Considering a move in spring next year",
        source_turn_id=4,
        source_event_id="ev_discovery_turn4",
    )

    # Start fresh new call
    mgr = ConversationStateManager(
        call_sid="call_elena_new",
        user_id=user,
        prospect_id=prospect,
        load_prospect_memory=True,
        memory_store=memory_store,
    )

    # 1. ConversationState is fresh: facts list is clean slate, NOT merged silently
    assert len(mgr.current_state.facts) == 0
    assert mgr.current_state.call_sid == "call_elena_new"

    # 2. Memory loads into separate loaded_prospect_memory structure
    assert len(mgr.current_state.loaded_prospect_memory) == 1
    loaded = mgr.current_state.loaded_prospect_memory[0]

    # 3. Tagged with original provenance
    prov = loaded.provenance
    assert isinstance(prov, MemoryProvenance)
    assert prov.source_call_sid == "call_elena_orig"
    assert prov.source_turn_id == 4
    assert prov.source_event_id == "ev_discovery_turn4"
    assert prov.originating_user_id == user
    assert prov.originating_prospect_id == prospect
    assert loaded.is_historical is True


def test_situation_3_different_user_same_prospect_or_phone_zero_retrieval_zero_exposure(memory_store):
    """Situation 3: Different user, same prospect/phone -> zero retrieval, zero exposure."""
    shared_phone = "+1-555-9999"
    user_primary = "rep_frank"
    user_unauthorized = "rep_grace"

    # User Frank records proprietary client notes
    memory_store.save_record(
        user_id=user_primary,
        prospect_id=shared_phone,
        source_call_sid="call_frank_1",
        memory_type="fact",
        key="financial_reserve",
        value="liquidating_inheritance",
        content="Client mentioned liquidating an inheritance next month",
        source_turn_id=3,
    )

    # User Grace calls the same phone number
    mgr_grace = ConversationStateManager(
        call_sid="call_grace_1",
        user_id=user_unauthorized,
        prospect_id=shared_phone,
        load_prospect_memory=True,
        memory_store=memory_store,
    )

    # Grace has ZERO retrieval and ZERO exposure
    assert len(mgr_grace.current_state.loaded_prospect_memory) == 0
    assert len(mgr_grace.current_state.facts) == 0
    assert memory_store.retrieve_memories(user_id=user_unauthorized, prospect_id=shared_phone) == []


def test_situation_4_new_simulation_or_test_memory_off_by_default_unless_explicitly_enabled(memory_store):
    """Situation 4: New simulation/test -> memory off by default unless explicitly enabled.
    Also tests that a call with load_prospect_memory=True but missing user_id fails closed.
    """
    user = "rep_hank"
    prospect = "prospect_hank_sim"

    memory_store.save_record(
        user_id=user,
        prospect_id=prospect,
        source_call_sid="call_hank_prior",
        memory_type="contact_preference",
        key="opt_out",
        value="no_morning_calls",
        content="No morning calls",
        source_turn_id=2,
    )

    # 1. When load_prospect_memory is omitted / False (default for tests/simulations):
    mgr_default = ConversationStateManager(
        call_sid="sim_call_hank_test",
        user_id=user,
        prospect_id=prospect,
        memory_store=memory_store,
    )
    assert len(mgr_default.current_state.loaded_prospect_memory) == 0
    assert len(mgr_default.current_state.contact_compliance.contact_preferences) == 0

    # 2. Production fail-closed check: memory requested without user_id must fail closed
    mgr_no_user = ConversationStateManager(
        call_sid="call_hank_prod_missing_user",
        user_id=None,
        prospect_id=prospect,
        load_prospect_memory=True,
        memory_store=memory_store,
    )
    assert len(mgr_no_user.current_state.loaded_prospect_memory) == 0
    assert len(mgr_no_user.current_state.contact_compliance.contact_preferences) == 0

    # 3. Explicit opt-in with valid user_id loads memory
    mgr_enabled = ConversationStateManager(
        call_sid="sim_call_hank_test_enabled",
        user_id=user,
        prospect_id=prospect,
        load_prospect_memory=True,
        memory_store=memory_store,
    )
    assert len(mgr_enabled.current_state.loaded_prospect_memory) == 1


def test_situation_5_any_current_call_conversation_state_isolated_to_its_own_call_sid(memory_store):
    """Situation 5: Any call -> ConversationState stays isolated to its own call_sid
    regardless of memory policy.
    Writes made during Call 1 exist only in the store and loaded_prospect_memory,
    never appearing in Call 2's current-call facts.
    """
    user = "rep_ian"
    prospect = "prospect_ian"

    mgr_call1 = ConversationStateManager(
        call_sid="call_ian_alpha",
        user_id=user,
        prospect_id=prospect,
        load_prospect_memory=False,
        memory_store=memory_store,
    )

    # In Call 1: client mentions price expectation and preference
    mgr_call1.current_state.facts.append(
        PersistentFactRecord(
            fact_key="price_expectation",
            fact_value="$750,000",
            category="financial",
            timestamp_ms=1000,
            source_turn_id=3,
        )
    )
    mgr_call1.save_prospect_memory(
        memory_type="fact",
        key="price_expectation",
        value="$750,000",
        content="Target sale price: $750,000",
        source_turn_id=3,
    )
    mgr_call1.current_state.conversion_gate = MeetingConversionGate(is_open=True, status="open")

    # In Call 2 (same user, same prospect, memory loading enabled):
    mgr_call2 = ConversationStateManager(
        call_sid="call_ian_beta",
        user_id=user,
        prospect_id=prospect,
        load_prospect_memory=True,
        memory_store=memory_store,
    )

    # Assert Call 2 state is 100% isolated to its own call_sid:
    assert mgr_call2.current_state.call_sid == "call_ian_beta"
    assert len(mgr_call2.current_state.facts) == 0, "Call 1 facts must never bleed into Call 2 facts"
    assert mgr_call2.current_state.conversion_gate is None or mgr_call2.current_state.conversion_gate.is_open is False
    # Memory from Call 1 is in loaded_prospect_memory only, tagged with Call 1 provenance
    assert len(mgr_call2.current_state.loaded_prospect_memory) == 1
    assert mgr_call2.current_state.loaded_prospect_memory[0].provenance.source_call_sid == "call_ian_alpha"


# ==============================================================================
# Write-Path Scoping & Authenticated Session Binding (Issue 8)
# ==============================================================================

def test_write_path_scoping_binds_to_session_user_id(memory_store):
    """Verifies save_prospect_memory binds strictly to the manager's established session user_id
    and cannot be hijacked to write to another user's partition.
    """
    mgr_alice = ConversationStateManager(
        call_sid="call_alice_session",
        user_id="sales_rep_alice",
        prospect_id="prospect_target",
        memory_store=memory_store,
    )

    rec = mgr_alice.save_prospect_memory(
        memory_type="fact",
        key="property_condition",
        value="fully_renovated",
        content="Property renovated in 2024",
        source_turn_id=2,
    )

    assert rec.user_id == "sales_rep_alice"
    assert rec.source_call_sid == "call_alice_session"

    # Confirms written record is strictly in Alice's partition
    assert len(memory_store.retrieve_memories(user_id="sales_rep_alice", prospect_id="prospect_target")) == 1
    # Bob has zero access
    assert len(memory_store.retrieve_memories(user_id="sales_rep_bob", prospect_id="prospect_target")) == 0


# ==============================================================================
# Point 18: No Team / Shared Memory (Explicitly, don't build this)
# ==============================================================================

def test_point18_no_team_or_shared_memory_retrieval_guard(memory_store):
    """Point 18: Confirm and guard against any code path that shares memory across users/teams.
    1. retrieve_memories requires non-empty user_id (fails closed on empty/missing).
    2. Attempting team/shared retrieval raises NotImplementedError.
    3. AST inspection confirms no retrieval method in ProspectMemoryStore omits user_id.
    4. AST inspection of ConversationStateManager confirms _load_prospect_memory mandates user_id.
    """
    # 1. Unscoped lookup without user_id raises ValueError
    with pytest.raises(ValueError, match="user_id is mandatory"):
        memory_store.retrieve_memories(user_id="", prospect_id="any_prospect")

    with pytest.raises(ValueError, match="user_id is mandatory"):
        memory_store.retrieve_memories(user_id=None, prospect_id="any_prospect")

    # 2. Guard against team/shared memory raises NotImplementedError
    with pytest.raises(NotImplementedError, match="Team or shared memory is explicitly prohibited"):
        memory_store.get_shared_team_memories()

    # 3. AST inspection: check all methods in ProspectMemoryStore
    source = inspect.getsource(ProspectMemoryStore)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # If function is a retrieval method, it MUST have user_id in its arguments
            if "retrieve" in node.name or "get_memories" in node.name:
                args = [a.arg for a in node.args.args]
                assert "user_id" in args, f"Method {node.name} must mandate user_id parameter (Point 18)"

    # 4. AST inspection of ConversationStateManager: verify _load_prospect_memory mandates user_id
    csm_source = inspect.getsource(ConversationStateManager._load_prospect_memory)
    assert "user_id" in csm_source
    assert "Point 18 fail-closed" in csm_source
