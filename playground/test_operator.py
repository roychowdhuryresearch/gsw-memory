# Import necessary modules
import json
import os

os.environ["CURATOR_DISABLE_CACHE"] = "true"
from dotenv import load_dotenv
from gsw_memory.memory import GSWProcessor, reconcile_gsw_outputs
from gsw_memory.memory.aggregators import EntitySummaryAggregator

# Load environment variables from .env file
load_dotenv()


def main():
    with open(
        "/mnt/SSD1/nlp/episodic-memory-benchmark/epbench/data/chapter_book.json",
        "r",
    ) as f:
        chapters = json.load(f)

    with open(
        "/mnt/SSD1/nlp/episodic-memory-benchmark/epbench/data/qa_list_20_allinfo.json",
        "r",
    ) as f:
        qa_data = json.load(f)

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
        enable_visualization=False,  # Set to True if you have NetworkX installed
    )

    print(f"Reconciled {len(reconciled_chapters)} chapters:")
    for i, chapter_gsw in enumerate(reconciled_chapters):
        print(
            f"  Chapter {i}: {len(chapter_gsw.entity_nodes)} entities, "
            f"{len(chapter_gsw.verb_phrase_nodes)} verb phrases"
        )

    print("\nGSW processing and reconciliation completed successfully!")

    # Test EntitySummaryAggregator
    print("\n=== Testing EntitySummaryAggregator ===")

    # Use the first reconciled chapter for testing
    test_gsw = reconciled_chapters[0]
    print(f"Testing with GSW containing {len(test_gsw.entity_nodes)} entities")

    # Create aggregator with same LLM config
    llm_config = {
        "model_name": "gpt-4o",
        "generation_params": {"temperature": 0.0, "max_tokens": 500},
    }
    aggregator = EntitySummaryAggregator(test_gsw, llm_config)

    # Get first 3 entities for testing
    # test_entity_ids = [entity.id for entity in test_gsw.entity_nodes[:3]]
    # print(f"Testing with entities: {[test_gsw.entity_nodes[i].name for i in range(min(3, len(test_gsw.entity_nodes)))]}")

    # Generate summaries (static generation)
    print("\nGenerating summaries...")
    summaries = aggregator.precompute_summaries(include_space_time=True)

    # Display results
    print(f"\nGenerated {len(summaries)} summaries:")
    for entity_id, summary_data in summaries.items():
        print(f"\n--- {summary_data['entity_name']} ---")
        print(summary_data["summary"])

    # Show detailed example for first entity
    print("\n=== DETAILED EXAMPLE ===")
    first_entity = test_gsw.entity_nodes[0]
    print(f"Entity: {first_entity.name} (ID: {first_entity.id})")

    # Show raw chronological data
    print("\n--- Raw Chronological Data ---")
    chronological_data = aggregator._aggregate_entity_data(
        first_entity, include_space_time=True
    )
    print(f"Found data for {len(chronological_data)} chunks:")
    for chunk_id, data in list(chronological_data.items())[:2]:  # Show first 2 chunks
        print(
            f"  {chunk_id}: {len(data['roles_states'])} roles, {len(data['actions'])} actions, {len(data['space_time'])} space/time"
        )

    # Show formatted timeline text
    print("\n--- Formatted Timeline (First 500 chars) ---")
    timeline_text = aggregator._format_data_for_prompt(
        first_entity.name, first_entity.id, chronological_data, include_space_time=True
    )
    print(timeline_text[:500] + "..." if len(timeline_text) > 500 else timeline_text)

    print("\n--- Final Summary ---")
    print(summaries[first_entity.id]["summary"])

    print("\nEntitySummaryAggregator testing completed!")


if __name__ == "__main__":
    main()
