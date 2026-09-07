"""Static Content Bank for AI Voice Calibration.

Provides verbatim multi-track conversational arcs authored for PitchProX Voice Calibration:
- 'seller': Expired listing homeowner negotiation arc (Frustrated Homeowner)
- 'internal_rep': High-producing agent onboarding arc (Skeptical Top Producer)

Each track contains 7 fixed rounds. In each round:
- seller_line: Verbatim prospect speech synthesized to audio.
- response_variants: 4 pre-authored response variants (A, B, C, D) spanning
  increasing linguistic/emotional delivery complexity. The engine selects exactly
  one variant at random per round and presents it as the single teleprompt line.
- final_statement: Closing prospect statement synthesized as an audio payoff upon
  Round 7 completion, immediately terminating the session without further recording.
"""

from typing import Any, Dict

CALIBRATION_TRACKS: Dict[str, Dict[str, Any]] = {
    "seller": {
        "track_id": "seller",
        "track_name": "Expired Listing Seller",
        "persona_name": "Frustrated Homeowner",
        "persona_title": "Expired Listing Homeowner",
        "voice_id": "VR6AewLTigWG4xSOukaG",  # Arnold / mature male
        "voice_name": "Arnold",
        "rounds": [
            {
                "round_number": 1,
                "title": "The Interruption Barrier",
                "seller_line": "Look, I’m going to stop you right there. You’re probably the fifteenth agent who’s called me since this morning, and honestly, I’m already over it.",
                "response_variants": {
                    "A": "I don’t blame you. Fifteen calls before you’ve probably finished your morning would wear anybody out. Give me twenty seconds, and if this sounds like every other call, you can stop me.",
                    "B": "I can hear that, and honestly, the last thing I want to do is become call number sixteen saying exactly what the other fifteen agents already said. So rather than pitch you, can I ask you one quick question about what actually happened?",
                    "C": "That frustration makes complete sense. When a listing comes off the market, the sudden flood of nearly identical calls can make every agent sound interchangeable. I’m not asking you to trust that I’m different yet—just give me a moment to understand why the property didn’t sell.",
                    "D": "Fair enough—and if I were hearing the same recycled promises from agent after agent, I’d probably be skeptical too. But before you write off this conversation with all the others, let me ask you something: looking back at the last few months, where do you feel the strategy actually stopped working?",
                },
            },
            {
                "round_number": 2,
                "title": "Loyalty to the Previous Agent",
                "seller_line": "Honestly, I don’t think my agent was the problem. We’ve known her for years, she worked really hard, and I trust her. The market just wasn’t on our side.",
                "response_variants": {
                    "A": "That may be completely true. And if you trust her, I’m not going to sit here and tell you she did a bad job. I’m more interested in whether the strategy gave the home its best chance to sell.",
                    "B": "I respect that, and loyalty matters. The difficult part is that a good relationship and a good strategy aren’t always the same thing. If you separated the person from the process for a minute, what do you think actually held the sale back?",
                    "C": "Absolutely—and I wouldn’t confuse an unsuccessful outcome with a lack of effort. Someone can work incredibly hard and still have a strategy that isn’t producing the necessary response from the market. Looking back, was there anything you expected to happen that simply never did?",
                    "D": "I actually respect you for saying that, because it’s easy to blame someone when the result is disappointing. But you can appreciate how hard she worked and still ask a different question: if the home didn’t sell, what part of the strategy deserved to be reconsidered before you ever put it back on the market?",
                },
            },
            {
                "round_number": 3,
                "title": "We Already Tried Everything",
                "seller_line": "I mean, we did everything we were supposed to do. We had professional photos, open houses, dropped the price twice… and still nothing. At some point, what else are you supposed to do?",
                "response_variants": {
                    "A": "That’s the frustrating part—you did make changes, and still didn’t get the result. But doing more isn’t always the answer. Sometimes it’s figuring out which part of the strategy wasn’t creating a response.",
                    "B": "I can understand why you’d feel like you exhausted your options. But photos, open houses, and price reductions are actions; the bigger question is whether those actions were solving the reason buyers weren’t moving forward.",
                    "C": "And that’s where I’d be careful about assuming the market simply rejected the house. You had activity, adjustments, and exposure—but if none of it changed buyer behavior, I’d want to understand whether the strategy was addressing the actual resistance or just reacting to the lack of a sale.",
                    "D": "That’s exactly why I wouldn’t tell you to just relist it and try harder. You already tried harder. The more useful question is why each adjustment failed to change the outcome. Because if we can isolate where buyers were disengaging—whether it was positioning, perception, price, or something else entirely—then we’re no longer guessing at what needs to change.",
                },
            },
            {
                "round_number": 4,
                "title": "Maybe We’ll Just Wait",
                "seller_line": "To be honest, after everything we went through, I’m not even sure we want to put it back on the market right now. Maybe we just wait six months and see what happens.",
                "response_variants": {
                    "A": "I can understand wanting a break. Before you decide on six months, though, I’d just want to know what you’re hoping will be different by then.",
                    "B": "That’s completely reasonable after what you’ve been through. I’d only be careful about letting a frustrating experience make the timing decision for you. If waiting makes sense strategically, great—but what would need to change in six months for you to feel differently?",
                    "C": "I certainly wouldn’t suggest relisting simply for the sake of being back on the market. But postponing the decision and improving the probability of a different outcome aren’t necessarily the same thing. I’d want to understand what specifically you believe six months would solve.",
                    "D": "And you may ultimately decide that waiting is exactly the right move. I just wouldn’t want you making a six-month decision because the last six months were disappointing. Those are two very different things. So before I ever suggested putting the home back on the market, I’d want to understand: are you waiting for the market to change—or are you waiting because you don’t want to go through that experience again?",
                },
            },
            {
                "round_number": 5,
                "title": "If We Sell, I’d Probably Use Her Again",
                "seller_line": "And honestly, if we do decide to put it back on the market, I’d probably just use our old agent again. She worked really hard for us, and I’d feel pretty bad giving the listing to somebody else.",
                "response_variants": {
                    "A": "I respect that. Loyalty says a lot about you. I’d just separate being loyal to her from being loyal to the outcome you and your family actually need.",
                    "B": "I actually think that says something good about you—you don’t want to disregard someone who worked hard for you. But choosing a different strategy doesn’t have to mean you’re disrespecting the person. It may simply mean the next attempt needs something different.",
                    "C": "That sense of loyalty is understandable, especially when there’s a personal relationship involved. The distinction I’d make is between appreciating someone’s effort and feeling obligated to repeat a strategy that didn’t produce the outcome you needed. Those two decisions don’t necessarily have to be tied together.",
                    "D": "And I wouldn’t try to talk you out of being loyal—that’s obviously important to you. I’d only ask you to consider where that loyalty should end and your responsibility to yourself begins. Because if you genuinely believe she gives you the best chance of getting the home sold, that’s one decision. If you’re choosing her primarily because you’d feel guilty choosing somebody else… that’s a very different decision.",
                },
            },
            {
                "round_number": 6,
                "title": "I’m Not Going to Give It Away",
                "seller_line": "The other thing is, I’m not giving this house away. We already dropped the price twice, and every agent I talk to keeps telling me it needs to come down more. I know what my house is worth.",
                "response_variants": {
                    "A": "And I wouldn’t ask you to give it away. Price matters, but it’s not the only thing that determines how buyers respond. I’d first want to understand why they weren’t seeing the value you see.",
                    "B": "I understand, and another price reduction without understanding the problem would concern me too. The question isn’t simply, ‘How low do we go?’ It’s, ‘What kept buyers from justifying the value?’ Those are two very different conversations.",
                    "C": "I’d be reluctant to recommend another reduction without first understanding whether price was actually the resistance. There’s an important distinction between a property being overpriced and a property whose value wasn’t effectively communicated, positioned, or supported in the buyer’s mind.",
                    "D": "And this is where I think sellers sometimes get put in an unfair position: the property doesn’t sell, so the automatic answer becomes, ‘Reduce the price.’ Maybe price was the issue—but maybe it wasn’t. Before asking you to concede another dollar, I’d want to know whether buyers rejected the value itself, or whether the strategy failed to make that value compelling enough for them to act.",
                },
            },
            {
                "round_number": 7,
                "title": "Why Should I Believe You’ll Be Any Different?",
                "seller_line": "Okay, but every agent I’ve talked to says they have a different strategy and they can get it sold. That’s pretty much what my last agent said too. So why should I believe you’d be any different?",
                "response_variants": {
                    "A": "You shouldn’t—not just because I tell you I’m different. I’d rather show you what I see, explain what I’d change, and let you decide whether the difference is meaningful.",
                    "B": "That’s fair. Another agent promising you a better result probably doesn’t mean much right now. So I wouldn’t ask you to believe me—I’d ask for twenty minutes at the property to show you what I believe happened and what I would do differently.",
                    "C": "Skepticism is probably appropriate after what you’ve experienced. The distinction shouldn’t come from another promise; it should come from whether I can identify what prevented the home from selling, substantiate that with evidence, and present a strategy that materially addresses those problems.",
                    "D": "I actually don’t think you should believe I’m different yet. You’ve heard the promises, you’ve made the adjustments, and you’ve already given one professional the opportunity to get this done. So rather than asking you for another leap of faith, give me twenty minutes at the property. Let me show you where I believe the previous strategy lost leverage, what I would change, and why—and then you can decide whether another six months with me would genuinely look any different from the six months you just experienced.",
                },
            },
        ],
        "final_statement": "You know what, that actually makes sense. I’ll give you that—you definitely know your stuff, and this has been a very different conversation from most of the calls I’ve gotten. I’m not promising we’re ready to relist, but I’d be open to hearing what you see. Come by, take a look at the house, and show me what you’d do differently.",
    },
    "internal_rep": {
        "track_id": "internal_rep",
        "track_name": "Top Producer Onboarding",
        "persona_name": "Skeptical Top Producer",
        "persona_title": "Top-Producing Real Estate Agent",
        "voice_id": "TxGEqnHWrfWFTfGW9XjX",  # Josh / authoritative decision maker
        "voice_name": "Josh",
        "rounds": [
            {
                "round_number": 1,
                "title": "I Don't Really Need This",
                "seller_line": "I get what you're saying, but I'll be honest with you—I’m already one of the top agents in my market. I’ve been doing this a long time, my business is doing extremely well, and I really don’t need something teaching me how to talk to clients.",
                "response_variants": {
                    "A": "And you probably don’t need anyone teaching you how to sell. That’s actually not what caught my attention. I’m more interested in what happens when someone at your level has that same intelligence available on every single call.",
                    "B": "I wouldn’t expect a top producer to need basic sales coaching. You’ve already proven you know how to have the conversation. The interesting question is whether technology can give an already successful agent an advantage they simply couldn’t have while processing everything themselves.",
                    "C": "Your production actually changes the conversation for me. At your level, I’m less interested in correcting sales ability than augmenting it—giving you real-time intelligence around objections, trust, timing, and conversational movement while you stay completely focused on the person in front of you.",
                    "D": "And frankly, if you needed software to teach you how to sell, you probably wouldn’t be a top producer. That’s not the premise. The premise is that even an exceptional salesperson is still listening, thinking, remembering, reading the client, managing the objection, and deciding what to say next—all in real time. PitchProX gives that salesperson another layer of intelligence without taking the conversation away from them.",
                },
            },
            {
                "round_number": 2,
                "title": "Most AI Sales Tools Are Gimmicks",
                "seller_line": "Yeah, but I hear that from every tech company. Everything is AI now, everything is supposed to make me better, and half of it ends up being another piece of software nobody actually uses. What makes this any different?",
                "response_variants": {
                    "A": "That’s fair. The difference is PitchProX isn’t something you have to remember to use after the conversation. It’s working with you during the call, helping you navigate what’s actually happening in real time.",
                    "B": "I’d be skeptical too, especially with how loosely the word AI gets thrown around. PitchProX is different because it isn’t giving you generic advice after the fact. While you’re talking, it’s interpreting the conversation and giving you the exact language to respond in that moment.",
                    "C": "And I think that skepticism is justified. Most technology asks you to interrupt your existing process in order to get value from it. PitchProX is designed around the opposite idea: it analyzes conversational signals, objections, trust, timing, and progression while the conversation is actually unfolding, then translates that intelligence into what you can say next.",
                    "D": "I think the mistake would be asking you to care that it uses AI. You shouldn’t. What matters is whether it can do something useful while you’re actually selling. If a client pushes back, hesitates, changes tone, repeats an objection, or starts pulling away, PitchProX is processing those signals in real time, deciding what the conversation needs next, and putting the exact response in front of you before the moment is gone.",
                },
            },
            {
                "round_number": 3,
                "title": "I Already Have a Team for That",
                "seller_line": "But I already have a pretty sophisticated operation. I have a team, a CRM, follow-up systems, coaching—we track our numbers. I’m not really seeing where another platform fits into what we’re already doing.",
                "response_variants": {
                    "A": "And I wouldn’t replace any of that. Those systems support everything around the conversation. PitchProX is focused on the part only you can handle—the conversation while it’s actually happening.",
                    "B": "It sounds like you’ve built the infrastructure really well. But your CRM can tell you who to call, and your team can make sure the follow-up happens. Neither one can help you decide what to say when a client gives you an objection you weren’t expecting.",
                    "C": "That infrastructure is probably part of why you produce at the level you do. PitchProX isn’t designed to duplicate it; it addresses a different gap. Your systems can organize the opportunity before and after the call, while PitchProX provides real-time conversation intelligence during the moment when the opportunity can actually change.",
                    "D": "And that’s actually where I’d expect someone at your level to challenge this. You’ve already built systems for lead generation, follow-up, accountability, marketing, and transaction management. But when you’re thirty seconds into a difficult conversation and the client suddenly hesitates, pushes back, or says something you weren’t expecting—your entire operation still comes down to you, in that moment, deciding what to say next. That’s the gap PitchProX is built around.",
                },
            },
            {
                "round_number": 4,
                "title": "I Don’t Want to Sound Like I’m Reading a Script",
                "seller_line": "Here’s my concern. I’ve spent years developing my own style, and people respond to me because I’m natural. The last thing I want is to be staring at some AI script and suddenly sound like I don’t know how to have a conversation.",
                "response_variants": {
                    "A": "And you shouldn’t give that up. PitchProX gives you the words, but you’re still the one delivering them. Your voice, your judgment, and your relationship with the client stay yours.",
                    "B": "That would defeat the purpose. The goal isn’t to turn a great salesperson into someone reading a script. PitchProX gives you the next response in real time, while Calibration adapts how those prompts are presented to the way you naturally communicate.",
                    "C": "I’d actually argue that your individual communication style becomes more important, not less. PitchProX determines the strategic response the conversation calls for, while Calibration shapes how that response is presented around your natural pace, vocabulary, cadence, and delivery.",
                    "D": "And I think that concern matters more for someone at your level, because your communication style is already part of why people trust you. PitchProX isn’t trying to replace that. It’s listening to what’s happening, identifying the objection, reading the conversational signals, deciding what the moment calls for—and then giving you the exact language to work with. You remain the salesperson. The intelligence is there to support the moment, not take it over.",
                },
            },
            {
                "round_number": 5,
                "title": "I Don’t Have Time to Learn Another Platform",
                "seller_line": "It sounds interesting, but honestly, I barely have time for the systems I already use. I’m on calls, in appointments, dealing with clients all day. I’m not looking for another platform I have to spend weeks learning.",
                "response_variants": {
                    "A": "And I wouldn’t expect you to spend weeks learning it. The whole point is for PitchProX to become useful while you’re doing what you already do—having conversations.",
                    "B": "That makes sense, especially when your calendar is already full. But this isn’t supposed to become another job for you. You get it calibrated, set up the intelligence you want behind you, and then use it when you make the calls you’re already making.",
                    "C": "I’d have the same concern if adopting it required you to restructure your existing workflow. The objective is the opposite: configure it around how you already sell, calibrate it to how you naturally communicate, and make the intelligence available inside the conversations you’re already having.",
                    "D": "And for somebody producing at your level, I’d argue the adoption burden has to be incredibly low or the technology isn’t worth having. You shouldn’t have to change the way you run your business just to accommodate another piece of software. The setup is there to teach PitchProX about you—not to make you reorganize yourself around PitchProX. Once that foundation is there, the intelligence shows up where you actually need it: inside the conversation.",
                },
            },
            {
                "round_number": 6,
                "title": "What Is This Actually Going to Do for My Business?",
                "seller_line": "Okay, but at the end of the day, I run a business. I already spend a lot on technology, marketing, coaching, lead generation—you name it. What is PitchProX actually going to do for me that makes another monthly subscription worth paying for?",
                "response_variants": {
                    "A": "That’s the right question. If it doesn’t make your conversations more effective, it isn’t worth paying for. The value is having another layer of intelligence behind you when one response can change where a call goes.",
                    "B": "I wouldn’t justify it by how many features it has. I’d look at the value of the conversations you’re already having. At your production level, one objection handled differently, one opportunity kept alive, or one appointment that otherwise stalls can matter considerably more than the monthly cost of the platform.",
                    "C": "For someone at your level, I think the economics are less about generating more activity and more about improving the quality of the opportunities you already create. If PitchProX helps you recognize resistance earlier, preserve trust through a difficult objection, and keep a viable conversation progressing, the leverage is being applied to moments that already have substantial value.",
                    "D": "And I wouldn’t try to convince you that another subscription is worthwhile just because the technology is impressive. For a producer at your level, the real question is what a missed moment costs. A seller hesitates and you push too early. A buyer raises an objection and the real concern goes unidentified. A prospect is ready to move forward, but the opportunity passes because the conversation never gets there. PitchProX is designed to give you better intelligence in those moments—where a relatively small improvement in the conversation can have a disproportionately meaningful impact on the opportunity.",
                },
            },
            {
                "round_number": 7,
                "title": "I’m Just Not Sure I’d Actually Use It",
                "seller_line": "I’ll be honest, I like the concept, and it sounds more sophisticated than I expected. I’m just not sure how much I’d actually use it. I’ve signed up for plenty of things that sounded great, used them for two weeks, and then forgot about them.",
                "response_variants": {
                    "A": "That’s fair. Then don’t choose a plan based on how much you think you might use someday. Start with the level that fits your calling now, use it in real conversations, and let the value prove itself.",
                    "B": "I think that’s exactly why I’d start practically rather than overcommit. Choose the subscription that matches how you’re working today, put PitchProX behind the calls you’re already making, and see what happens when you actually have that intelligence available in the moment.",
                    "C": "That hesitation makes sense, because the value isn’t theoretical—it has to become evident in your actual conversations. I’d rather have you start at the subscription level that reflects your current activity, experience PitchProX under real conditions, and then decide whether expanding your usage is justified by what you’re seeing.",
                    "D": "And I actually think that’s the right standard. You shouldn’t subscribe because I gave you a convincing explanation of what PitchProX can do; you should subscribe because you want to find out what it does inside your conversations. So I wouldn’t overthink the first decision. Pick the level that makes sense for how you operate today, put it behind your real calls, and judge it the same way you’d judge anything else in your business—by whether it earns the right to stay.",
                },
            },
        ],
        "final_statement": "You know what, that’s fair. I’ll give you credit—this wasn’t the typical software pitch I was expecting, and you clearly know what you’re talking about. I can actually see where I’d use this. Let’s do it. I’ll start with a subscription and put it through some real calls. If it works the way you’re describing, I can definitely see this becoming part of how I operate.",
    },
}
