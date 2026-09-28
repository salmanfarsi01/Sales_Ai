import sys
sys.path.insert(0, ".")
from copilot.conversation_state_manager import ConversationStateManager
from tests.test_conversation_state_corroborating_dormancy import _create_bundle
mgr = ConversationStateManager("CA_silence_test")
b1 = _create_bundle(1, "client", "Your commission fee of 6 percent is way too high.")
mgr.process_turn_bundle(b1)
b2 = _create_bundle(2, "salesperson", "What is your timeline for the move?")
mgr.process_turn_bundle(b2)
b3 = _create_bundle(3, "client", "We might look at moving next year.", agreement=0.0, future_lang=0.0)
mgr.process_turn_bundle(b3)
b4 = _create_bundle(4, "salesperson", "Understood.", agreement=0.0, future_lang=0.0)
s4 = mgr.process_turn_bundle(b4)
obj = s4.objections[0]
print("obj lifecycle:", obj.lifecycle_state)
print("dormancy evidence:", obj.dormancy_evidence)
