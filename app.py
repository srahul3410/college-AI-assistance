
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Current Gemini API model IDs
MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
]

SYSTEM_PROMPT = """
You are CampusMate, a helpful AI assistant for college students.

Help with:
1. DSA, C, C++, Java and Python programming
2. Assignments, practicals, viva and exam preparation
3. Attendance-related guidance
4. Scholarships and college procedures
5. Projects, internships, resumes and placements
6. Career roadmaps and technical interview preparation

Rules:
- Use simple English or Hinglish according to the student's message.
- Explain concepts in beginner-friendly language.
- Give short answers for simple questions.
- For programming questions, provide correct code and sample output.
- Give step-by-step instructions when needed.
- Never claim to know the student's actual attendance or college records.
- Do not invent official college policies or scholarship rules.
- Be friendly, respectful and practical.
"""


# ---------------- GEMINI RESPONSE ----------------

def generate_answer(client, conversation):
    last_error = None

    for model in MODELS:
        # Try each available model up to two times for temporary errors.
        for attempt in range(2):
            try:
                print(f"Trying model: {model}")

                response = client.models.generate_content(
                    model=model,
                    contents=conversation,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.7,
                        max_output_tokens=2048,
                    ),
                )

                if response.text:
                    return response.text

                last_error = "The model returned an empty response."
                break

            except Exception as error:
                last_error = str(error)
                error_text = last_error.upper()

                if (
                    "API_KEY_INVALID" in error_text
                    or "API KEY NOT VALID" in error_text
                ):
                    return (
                        "Your API key is invalid. Check GEMINI_API_KEY "
                        "inside your .env file."
                    )

                if "PERMISSION_DENIED" in error_text or "403" in error_text:
                    return (
                        "Access denied. Check your API key permissions "
                        "and model access in Google AI Studio."
                    )

                if (
                    "404" in error_text
                    or "NOT_FOUND" in error_text
                ):
                    print(f"{model} is unavailable. Trying another model.")
                    break

                temporary_error = any(
                    marker in error_text
                    for marker in (
                        "503",
                        "UNAVAILABLE",
                        "429",
                        "RESOURCE_EXHAUSTED",
                        "500",
                        "502",
                        "504",
                        "INTERNAL",
                        "TIMEOUT",
                    )
                )

                if temporary_error:
                    if attempt == 0:
                        print(
                            "Temporary service problem. "
                            "Retrying in 2 seconds..."
                        )
                        time.sleep(2)
                        continue

                    print(f"{model} is still unavailable. Trying next model.")
                    break

                print(f"API error: {last_error}")
                return (
                    "Could not get an answer. Check the technical "
                    "error printed in the terminal."
                )

    print(f"\nLast API error: {last_error}")

    return (
        "Sorry, Gemini is currently unavailable or your API quota "
        "may have been reached. Please wait a few minutes and try again. "
        "Check the terminal for the exact error."
    )


# ---------------- CHAT APPLICATION ----------------

def main():
    if not API_KEY or API_KEY == "YOUR_ACTUAL_API_KEY":
        print("API key missing!")
        print(f"Please add GEMINI_API_KEY to: {BASE_DIR / '.env'}")
        return

    try:
        client = genai.Client(api_key=API_KEY)
    except Exception as error:
        print(f"Could not initialize Gemini: {error}")
        return

    # Keep recent conversation context during this session.
    history = []

    print("=" * 48)
    print("          WELCOME TO CAMPUSMATE AI")
    print("=" * 48)
    print("Ask questions about college, DSA, Python and careers.")
    print("Commands: 'exit' to quit, 'clear' to clear memory.")
    print("Note: Conversation memory resets when you close the app.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nCampusMate: Goodbye! Keep learning.")
            break

        if not user_input:
            continue

        command = user_input.lower()

        if command in ("exit", "quit", "bye"):
            print("CampusMate: Goodbye! Keep learning.")
            break

        if command == "clear":
            history.clear()
            print("CampusMate: Conversation memory cleared.\n")
            continue

        # Add the new question and retain recent context.
        history.append(
            types.Content(
                role="user",
                parts=[types.Part(text=user_input)],
            )
        )

        # Limit context size to the latest 10 messages.
        recent_history = history[-10:]

        print("\nCampusMate: Thinking...\n")
        answer = generate_answer(client, recent_history)

        print(f"CampusMate: {answer}\n")

        history.append(
            types.Content(
                role="model",
                parts=[types.Part(text=answer)],
            )
        )

        # Prevent unlimited memory growth.
        history = history[-10:]


if __name__ == "__main__":
    main()
