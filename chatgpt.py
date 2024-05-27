from openai import OpenAI
import os
import toml

secrets = "/Users/suryaganesan/Documents/GitHub/Replicate/secrets.toml"
github_secrets = "secrets.toml"

os.environ["OPENAI_API_KEY"] = toml.load(github_secrets)["OPENAI_API_KEY"]
client = OpenAI()

def query_llm_gpt4(user_msg, client=client):
    system_msg = f"""You are an AI python programmer. You are adept in writing and reviewing code. Assume that openpyxl library is already installed when writing code.
    """

    messages = [
        {
            "role": "system",
            "content": system_msg
        }
    ]

    messages.append(
        {
            "role": "user",
            "content": user_msg
        }
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    ).choices[0].message

    return response

response = query_llm_gpt4(
"""Write me a VBA script that will iterate through every row of the excel sheet, excluing the first row and apply conditional formatting to each row to highlight duplicate values.
I want to apply this always to the current open worksheet.
I am going to copy paste this exactly so make sure its readily executable."""
)

print(response.content)