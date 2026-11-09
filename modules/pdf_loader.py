import fitz
from pathlib import Path


class PDFLoader:
    def __init__(self, data_folder):
        self.data_folder = Path(data_folder)

    def load_documents(self):
        documents = []

        pdf_files = sorted(self.data_folder.glob("*.pdf"))

        if not pdf_files:
            raise FileNotFoundError("No PDF files found in data folder.")

        for pdf in pdf_files:
            doc = fitz.open(pdf)

            for page_num, page in enumerate(doc, start=1):
                text = page.get_text().strip()

                if text:
                    documents.append({
                        "source": pdf.name,
                        "page": page_num,
                        "text": text
                    })

            doc.close()

        return documents