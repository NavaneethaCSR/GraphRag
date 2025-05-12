import re

# File paths
input_file = r"C:\Users\LENOVO\GRAPHRAG\hybrid.graphml"
output_file = r"C:\Users\LENOVO\GRAPHRAG\hybrid.graphml"

# Read file content
with open(input_file, "r", encoding="utf-8") as file:
    content = file.read()

# Fix double quotes in attributes (e.g., id=""FREE TOOL"" → id="FREE TOOL")
content = re.sub(r'(\w+)=""(.*?)""', r'\1="\2"', content)

# Fix double quotes in text (e.g., <data key=""d0"">TEXT</data> → <data key="d0">TEXT</data>)
content = re.sub(r'(<data key="[^"]+>)""(.*?)""(</data>)', r'\1\2\3', content)

# Save the cleaned file
with open(output_file, "w", encoding="utf-8") as file:
    file.write(content)

print(f"Cleaned file saved as: {output_file}")
