import os
from dotenv import load_dotenv
from openai import OpenAI

def main():
    # Load API key
    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key or not api_key.startswith("sk-proj-") or api_key.strip() != api_key:
        print("API key error: please check your .env file.")
        return
    else:
        print("API key loaded successfully.\n")

    openai = OpenAI()

    system_prompt = "Write in the style of Ernest Hemingway"
    messages = [{"role": "system", "content": system_prompt}]
    
    CONTEXT_WINDOW_SIZE = 5  # Number of user+assistant exchanges to remember

    print("Start chatting with Hemingway Bot (type 'exit' to quit):\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        # Keep only the last N user-assistant exchanges + system prompt
        user_assistant_msgs = [msg for msg in messages if msg["role"] != "system"]
        trimmed_history = user_assistant_msgs[-(CONTEXT_WINDOW_SIZE * 4):]
        current_context = [{"role": "system", "content": system_prompt}] + trimmed_history

        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=current_context
            )
            assistant_reply = response.choices[0].message.content
            print(f"Hemingway Bot: {assistant_reply}\n")

            messages.append({"role": "assistant", "content": assistant_reply})

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()

