def parse_chat_log(file_path):
    conversations = []
    current_role = None
    current_content = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped_line = line.strip()

            # Check if the line indicates a role switch
            if stripped_line in ["user", "assistant"]:
                # Save the previous message if it exists
                if current_role and current_content:
                    # join lines and strip trailing newline added by the process
                    content_str = "".join(current_content).rstrip("\n")
                    conversations.append(
                        {"role": current_role, "content": content_str}
                    )

                # Reset for the new role
                current_role = stripped_line
                current_content = []
            else:
                # If we are currently tracking a role, append the line
                if current_role is not None:
                    current_content.append(line)

        # Don't forget to append the very last message after the loop ends
        if current_role and current_content:
            content_str = "".join(current_content).rstrip("\n")
            conversations.append({"role": current_role, "content": content_str})

    return conversations

import json
# --- Example Usage ---
data = parse_chat_log("hhhh")
pretty_json = json.dumps(data, indent=4)
print(pretty_json)