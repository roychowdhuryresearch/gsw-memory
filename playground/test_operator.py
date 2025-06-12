# Import necessary modules
import json
import os

os.environ["CURATOR_DISABLE_CACHE"] = "true"
from dotenv import load_dotenv
from gsw_memory.memory import GSWProcessor, reconcile_gsw_outputs

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

    book = book[:2]
    print("Processing book of length", len(book))
    gsw_structures = processor.process_documents(book, output_dir="output")
    print(f"Generated GSW structures for {len(gsw_structures)} chapters")

    # NEW: Reconcile using the integration function with saving
    print("\n--- Reconciling with LOCAL strategy ---")
    reconciled_chapters = reconcile_gsw_outputs(
        gsw_structures, 
        strategy="local",
        output_dir="reconciled_output",
        save_statistics=True,
        enable_visualization=False  # Set to True if you have NetworkX installed
    )

    print(f"Reconciled {len(reconciled_chapters)} chapters:")
    for i, chapter_gsw in enumerate(reconciled_chapters):
        print(
            f"  Chapter {i}: {len(chapter_gsw.entity_nodes)} entities, "
            f"{len(chapter_gsw.verb_phrase_nodes)} verb phrases"
        )

    print("\nGSW processing and reconciliation completed successfully!")


if __name__ == "__main__":
    main()
