from dotenv import load_dotenv

load_dotenv()

from copilot.knowledge_all_questions import main  # noqa: E402


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Twilio copilot stopped.")
