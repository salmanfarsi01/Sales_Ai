"""Place a Twilio call using values from .env."""

import os
from xml.sax.saxutils import escape

from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv()


def required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing {name} in .env")
    return value


def main() -> None:
    stream_url = required("TWILIO_STREAM_URL").rstrip("/")
    if not stream_url.startswith("wss://"):
        raise RuntimeError("TWILIO_STREAM_URL must start with wss://")
    first_phone = required("TWILIO_TO_NUMBER")
    second_phone = required("TWILIO_DIAL_NUMBER")
    from_phone = required("TWILIO_FROM_NUMBER")
    twiml = (
        "<Response><Say>Connecting your call.</Say>"
        f'<Start><Stream url="{escape(stream_url)}" track="both_tracks" /></Start>'
        f"<Dial>{escape(second_phone)}</Dial></Response>"
    )
    call = Client(
        required("TWILIO_ACCOUNT_SID"), required("TWILIO_AUTH_TOKEN")
    ).calls.create(twiml=twiml, to=first_phone, from_=from_phone)
    print(f"Call started: {call.sid}")
    print(f"Media stream: {stream_url}")


if __name__ == "__main__":
    main()
