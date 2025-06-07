# Import necessary modules
import json

from dotenv import load_dotenv

from gsw_memory.memory import GSWProcessor

# Load environment variables from .env file
load_dotenv()


def main():
    with open(
        "/mnt/SSD1/nlp/episodic-memory-benchmark/epbench/data/chapter_book.json",
        "r",
    ) as f:
        chapters = json.load(f)

    processor = GSWProcessor(
        model_name="gpt-4o",
        enable_coref=True,
        enable_chunking=True,
        enable_context=True,
        chunk_size=3,
        overlap=0,
        enable_spacetime=True,
    )

    book = []
    for chapter, text in chapters.items():
        book.append(text)

    book = book[:5]
    print("Processing book of length", len(book))
    gsw_structures = processor.process_documents(book, output_dir="output")

    print("GSW structures generated successfully!")


if __name__ == "__main__":
    main()
