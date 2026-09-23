import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
from copilot.conversation_replay import ConversationReplayEngine

old_report = json.load(open('reports/synthetic/conversation_state_sim_mucj0p5s.json', encoding='utf-8'))
raw_turns = [{'turn_id': t['turn_id'], 'speaker_id': t['speaker_id'], 'text': t['text']} for t in old_report['timeline']]

engine = ConversationReplayEngine()
new_report = engine.replay_dialogue_turns(
    call_sid='sim_double_check',
    raw_turns=raw_turns,
    save_report=False
)

print('=== 1. CHECKING DORMANCY TRANSITION ACROSS TURNS 14-18 ===')
for step in new_report.timeline:
    if step.turn_id >= 14:
        old_step = next(s for s in old_report['timeline'] if s['turn_id'] == step.turn_id)
        print(f'\n--- Turn {step.turn_id} ({step.speaker_id}): "{step.text[:50]}..." ---')
        print(f'Old Materiality Targets: {sorted(list(old_step["materiality"]["affected_targets"]))}')
        print(f'New Materiality Targets: {sorted(list(step.materiality.affected_targets))}')
        
        # Check objections
        for o in step.state_after.objections:
            print(f'New Obj: id={o.objection_id}, cat={o.canonical_category}, state={o.lifecycle_state}, last_updated_turn={o.last_updated_turn_id}')
        
        # Check state changes related to objections
        obj_changes = [c for c in step.state_changes if 'objection' in c.field_path.lower()]
        old_obj_changes = [c for c in old_step['state_changes'] if 'objection' in c.get('field_path', '').lower()]
        print(f'New Obj State Changes: {[c.reason for c in obj_changes]}')
        print(f'Old Obj State Changes: {[c.get("reason") for c in old_obj_changes]}')

print('\n=== 2. CHECKING MOMENTUM & OBJECTION_MOVEMENT ACROSS TURNS 14-18 ===')
for step in new_report.timeline:
    if step.turn_id >= 14:
        old_step = next(s for s in old_report['timeline'] if s['turn_id'] == step.turn_id)
        old_mom = old_step.get('state_after', {}).get('momentum', {})
        new_mom = step.state_after.momentum
        
        old_obj_mov = old_mom.get('family_scores', {}).get('objection_movement')
        new_obj_mov = new_mom.family_scores.get('objection_movement') if new_mom else None
        
        old_score = old_mom.get('momentum_score')
        new_score = new_mom.momentum_score if new_mom else None
        
        print(f'Turn {step.turn_id}:')
        print(f'  objection_movement: old={old_obj_mov} -> new={new_obj_mov}')
        print(f'  momentum_score:     old={old_score} -> new={new_score}')
        if old_mom.get('family_scores') and new_mom:
            all_keys = sorted(list(new_mom.family_scores.keys()))
            for k in all_keys:
                ov = old_mom['family_scores'].get(k)
                nv = new_mom.family_scores.get(k)
                if ov != nv:
                    print(f'    diff in {k}: old={ov} -> new={nv}')
