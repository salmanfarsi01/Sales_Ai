# PDF knowledge

Install the PDF dependency once:

```powershell
python -m pip install -r requirements-pdf.txt
```

Start the normal application:

```powershell
python run_call_copilot.py
```

Open `http://127.0.0.1:8000`, choose a PDF in the header, and select **Upload PDF**. The server validates a maximum 20 MB PDF, extracts selectable text, stores it under `knowledge/`, and rebuilds the local index immediately. A restart is not required.

Password-protected PDFs are rejected. Image-only scanned PDFs require OCR and are not supported yet. When a matching passage is found, the dashboard displays its source. When no passage matches, the assistant still generates a best-effort answer from model knowledge and does not claim that it came from an uploaded file.
