import requests
import os
import sys

files = os.listdir("docs_pdf")
for file in files:
    if os.path.exists(f"docs_2/{file.split('.')[0]}.md"):
        print(f"Skipping {file}, already converted.")
        continue
        
    try:
        response = requests.post(
            "http://144.22.182.228:8000/ingest/pdf",
            data={"strategy": "recursive", "embed_model_provider": "gemini"},
            files={"file": (file, open(f"docs_pdf/{file}", "rb"), "application/pdf")}
        )
        response.raise_for_status()  # Raise an exception for bad status codes
        
        md_content = response.json().get("md")
        if md_content is None:
            print(f"Error: No markdown content received for {file}")
            print(f"Response content: {response.json()}")
            continue
            
        with open(f"docs_2/{file.split('.')[0]}.md", "w", encoding="utf-8") as f:
            f.write(md_content)
            print(f"Doc {len(os.listdir('docs_2'))} written: {file}")
            
    except requests.exceptions.RequestException as e:
        print(f"Error processing {file}: {str(e)}", file=sys.stderr)
    except Exception as e:
        print(f"Unexpected error processing {file}: {str(e)}", file=sys.stderr)