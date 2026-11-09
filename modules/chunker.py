from langchain_text_splitters import RecursiveCharacterTextSplitter
import re


class TextChunker:

    def __init__(self, chunk_size=700, chunk_overlap=150):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                "; ",
                ", ",
                " ",
                ""
            ]
        )

    def clean_text(self, text):

        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", " ", text)

        # Remove common PDF metadata / copyright noise
        remove_patterns = [
            r"ScienceDirect",
            r"Available online.*?(?=\n|$)",
            r"Published by Elsevier.*?(?=\n|$)",
            r"CC BY[-\w]*",
            r"creativecommons\.org\S*",
            r"Peer[- ]review.*?(?=\n|$)",
            r"Corresponding author.*?(?=\n|$)",
            r"Tel\..*?(?=\n|$)",
            r"fax.*?(?=\n|$)",
            r"E-mail.*?(?=\n|$)"
        ]

        for pattern in remove_patterns:
            text = re.sub(
                pattern,
                " ",
                text,
                flags=re.IGNORECASE
            )

        # Remove repeated whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove repeated punctuation artifacts
        text = re.sub(r"\[\s*[,;.\s]*\]", " ", text)

        # Remove isolated figure/table markers
        text = re.sub(
            r"\b(Fig|Figure|Table)\s*\.\s*",
            " ",
            text,
            flags=re.IGNORECASE
        )

        # Remove standalone citation brackets like [1], [23]
        text = re.sub(
            r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]",
            " ",
            text
        )

        # Clean excessive punctuation
        text = re.sub(r"\.{2,}", ".", text)
        text = re.sub(r",{2,}", ",", text)

        return text.strip()

    def create_chunks(self, documents):

        chunks = []

        chunk_id = 1

        for doc in documents:

            cleaned_text = self.clean_text(doc["text"])

            # Skip extremely small pieces of text
            if len(cleaned_text.strip()) < 50:
                continue

            split_text = self.splitter.split_text(cleaned_text)

            for chunk in split_text:

                chunks.append({
                    "chunk_id": chunk_id,
                    "source": doc["source"],
                    "page": doc["page"],
                    "text": chunk
                })

                chunk_id += 1

        return chunks