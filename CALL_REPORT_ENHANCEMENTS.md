"""
Call Report Enhancement: Lead Stage, Status, and Outcome

This shows how to add call classification fields to call_report.py
"""

# ============================================================================
# ENHANCEMENTS TO call_report.py
# ============================================================================

"""
Current call_report.py generates:
1. call_summary
2. call_status
3. outcome
4. key_moments_log
5. performance_metrics
6. conversion_indicators
7. agent_tone_delivery_feedback
8. agent_sentiment_responsiveness

Enhancement: Add fields that should be in JSON report:

Lead Tracking:
├─ Lead Stage: New → Attempted → Contacted → In Conversation → Follow-Up → 
               Qualified → Appointment Set → Archived
├─ Status: Hot, Warm, Cold, Unknown
└─ Outcome: Appointment Set, Follow-Up Required, Information Requested, 
            Not Interested, Not a Fit, No Answer, Voicemail

These should be:
1. Determined during/after the call
2. Set by the AI system based on conversation analysis
3. Stored in the JSON report
4. Accessible via API for CRM integration
"""

# ============================================================================
# IMPLEMENTATION PLAN
# ============================================================================

"""
Step 1: Add enums to call_report.py

from enum import Enum

class LeadStage(str, Enum):
    NEW = "New"
    ATTEMPTED = "Attempted"
    CONTACTED = "Contacted"
    IN_CONVERSATION = "In Conversation"
    FOLLOW_UP = "Follow-Up"
    QUALIFIED = "Qualified"
    APPOINTMENT_SET = "Appointment Set"
    ARCHIVED = "Archived"

class LeadStatus(str, Enum):
    HOT = "Hot"
    WARM = "Warm"
    COLD = "Cold"
    UNKNOWN = "Unknown"

class CallOutcome(str, Enum):
    APPOINTMENT_SET = "Appointment Set"
    FOLLOW_UP_REQUIRED = "Follow-Up Required"
    INFORMATION_REQUESTED = "Information Requested"
    NOT_INTERESTED = "Not Interested"
    NOT_A_FIT = "Not a Fit"
    NO_ANSWER = "No Answer"
    VOICEMAIL = "Voicemail"


Step 2: Modify CallReportGenerator to detect outcome

class CallReportGenerator:
    async def generate(self, transcript, call_context):
        # ... existing code ...
        
        # NEW: Analyze transcript for stage/status/outcome
        lead_stage = await self._determine_lead_stage(transcript)
        lead_status = await self._determine_lead_status(transcript)
        outcome = await self._determine_outcome(transcript)
        
        # Build report with new fields
        report = {
            "call_id": call_context["call_id"],
            "timestamp": call_context["timestamp"],
            
            # NEW: Lead tracking
            "lead_tracking": {
                "stage": lead_stage.value,
                "status": lead_status.value,
                "outcome": outcome.value,
            },
            
            # Existing fields
            "call_summary": summary,
            "call_status": status,
            ...
        }
        
        return report

    async def _determine_lead_stage(self, transcript) -> LeadStage:
        '''Analyze transcript to determine lead stage'''
        
        # Check conversation progression
        if not transcript:
            return LeadStage.ATTEMPTED
        
        # Use LLM to classify
        prompt = f'''
        Based on this call transcript, what is the lead stage?
        
        Stages:
        - New: First contact
        - Attempted: Tried to reach but didn't connect
        - Contacted: Successfully reached
        - In Conversation: Had meaningful discussion
        - Follow-Up: Need follow-up discussion
        - Qualified: Lead meets criteria
        - Appointment Set: Meeting scheduled
        - Archived: No longer pursuing
        
        Transcript:
        {self._format_transcript(transcript)}
        
        Respond with ONLY the stage name.
        '''
        
        response = self.groq.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3,  # Less randomness for classification
        )
        
        stage_name = response.choices[0].message.content.strip()
        
        try:
            return LeadStage[stage_name.upper().replace(" ", "_")]
        except KeyError:
            return LeadStage.IN_CONVERSATION  # Default

    async def _determine_lead_status(self, transcript) -> LeadStatus:
        '''Analyze transcript to determine lead interest level'''
        
        prompt = f'''
        Based on this call transcript, what is the prospect's interest level?
        
        Status levels:
        - Hot: Highly interested, ready to buy, urgent need
        - Warm: Interested, considering options, moderate timeline
        - Cold: Low interest, not convinced, long timeline
        - Unknown: Can't determine interest level
        
        Transcript:
        {self._format_transcript(transcript)}
        
        Respond with ONLY the status (Hot/Warm/Cold/Unknown).
        '''
        
        response = self.groq.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3,
        )
        
        status_name = response.choices[0].message.content.strip()
        
        try:
            return LeadStatus[status_name.upper()]
        except KeyError:
            return LeadStatus.UNKNOWN  # Default

    async def _determine_outcome(self, transcript) -> CallOutcome:
        '''Analyze transcript to determine call outcome'''
        
        prompt = f'''
        Based on this call transcript, what is the call outcome?
        
        Possible outcomes:
        - Appointment Set: Meeting scheduled
        - Follow-Up Required: Need to follow up
        - Information Requested: Lead asked for info to review
        - Not Interested: Lead explicitly said no
        - Not a Fit: Product/service not suitable
        - No Answer: Couldn't reach prospect
        - Voicemail: Left voicemail
        
        Transcript:
        {self._format_transcript(transcript)}
        
        Respond with ONLY the outcome.
        '''
        
        response = self.groq.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3,
        )
        
        outcome_name = response.choices[0].message.content.strip()
        
        try:
            # Replace spaces with underscores for enum lookup
            enum_name = outcome_name.upper().replace(" ", "_")
            return CallOutcome[enum_name]
        except KeyError:
            # If not exact match, return based on keywords
            transcript_lower = self._format_transcript(transcript).lower()
            
            if "appointment" in transcript_lower or "meeting" in transcript_lower:
                return CallOutcome.APPOINTMENT_SET
            elif "follow" in transcript_lower:
                return CallOutcome.FOLLOW_UP_REQUIRED
            elif "send" in transcript_lower or "email" in transcript_lower:
                return CallOutcome.INFORMATION_REQUESTED
            elif "not interested" in transcript_lower or "not right" in transcript_lower:
                return CallOutcome.NOT_INTERESTED
            elif "not a fit" in transcript_lower or "not suitable" in transcript_lower:
                return CallOutcome.NOT_A_FIT
            else:
                return CallOutcome.FOLLOW_UP_REQUIRED  # Default


Step 3: Example JSON output

{
  "call_id": "CA1234567890abcdef",
  "timestamp": "2024-01-15T14:30:00Z",
  
  "lead_tracking": {
    "stage": "Qualified",
    "status": "Hot",
    "outcome": "Appointment Set"
  },
  
  "call_summary": "Customer interested in premium plan...",
  "call_status": "completed",
  "outcome": "Success",
  
  "key_moments_log": [...],
  "performance_metrics": {...},
  "conversion_indicators": {...},
  "agent_tone_delivery_feedback": {...},
  "agent_sentiment_responsiveness": {...}
}
"""

# ============================================================================
# INTEGRATION WITH CRM
# ============================================================================

"""
Now you can sync to CRM with:

POST /api/crm/sync-call-report
{
  "call_id": "CA1234567890abcdef",
  "lead_id": "crm_12345",
  "stage": "Qualified",
  "status": "Hot",
  "outcome": "Appointment Set"
}

Example CRM integrations:
- Salesforce: API to update Lead.LeadStatus, Lead.LeadScore
- HubSpot: API to update hs_lead_status, hs_lead_priority
- Pipedrive: API to update deal.stage, deal.status
- Zoho: API to update Lead Stage and other fields
"""

# ============================================================================
# AUTOMATED WORKFLOW EXAMPLE
# ============================================================================

"""
Once you have lead stage/status/outcome, automate:

If outcome == "Appointment Set":
  → Auto-create calendar event
  → Send confirmation email to client
  → Notify sales manager

If status == "Hot":
  → Flag for immediate follow-up
  → Send to fastest-response team
  → Add 1-hour follow-up task

If status == "Warm":
  → Schedule follow-up email in 3 days
  → Auto-send additional collateral
  → Add weekly follow-up task

If status == "Cold":
  → Archive in CRM
  → Add to nurture campaign
  → Quarterly re-engagement email

If outcome == "No Answer":
  → Auto-retry tomorrow
  → Add voicemail to email
  → Create follow-up task

If outcome == "Not a Fit":
  → Mark as archived
  → Log reason in CRM
  → Remove from active pipeline
"""

# ============================================================================
# DASHBOARD DISPLAY
# ============================================================================

"""
Add to dashboard HTML (twilio_fast.html):

<div id="lead-tracking">
  <h3>Lead Tracking</h3>
  <div class="tracking-row">
    <span class="label">Stage:</span>
    <span id="lead-stage" class="value">—</span>
  </div>
  <div class="tracking-row">
    <span class="label">Status:</span>
    <span id="lead-status" class="value hot">—</span>
  </div>
  <div class="tracking-row">
    <span class="label">Outcome:</span>
    <span id="lead-outcome" class="value">—</span>
  </div>
</div>

<style>
#lead-tracking {
  padding: 15px;
  border: 1px solid #ccc;
  border-radius: 5px;
  margin: 10px 0;
}

.tracking-row {
  display: flex;
  justify-content: space-between;
  padding: 8px 0;
  border-bottom: 1px solid #eee;
}

.tracking-row:last-child {
  border-bottom: none;
}

.value.hot {
  background-color: #ff4444;
  color: white;
  padding: 2px 8px;
  border-radius: 3px;
}

.value.warm {
  background-color: #ffaa00;
  color: white;
  padding: 2px 8px;
  border-radius: 3px;
}

.value.cold {
  background-color: #6699ff;
  color: white;
  padding: 2px 8px;
  border-radius: 3px;
}
</style>

JavaScript to update:

websocket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === "call_report") {
    if (data.lead_tracking) {
      document.getElementById("lead-stage").textContent = data.lead_tracking.stage;
      document.getElementById("lead-status").textContent = data.lead_tracking.status;
      document.getElementById("lead-status").className = 'value ' + data.lead_tracking.status.toLowerCase();
      document.getElementById("lead-outcome").textContent = data.lead_tracking.outcome;
    }
  }
};
"""
