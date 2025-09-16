#!/usr/bin/env python3
"""
Test the new factual extraction prompt on the specific failing cases.

Tests Ermengarde of Tours and Edith Carlmar documents that were failing before.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from gsw_memory import GSWProcessor
from gsw_memory.prompts.operator_prompts import PromptType

# Load environment variables
load_dotenv()

# Disable cache for testing
os.environ["CURATOR_DISABLE_CACHE"] = "true"




def analyze_ermengarde_extraction(gsw_data):
    """Analyze if Ermengarde extraction captures the death date relationship."""
    print("\n=== ANALYZING ERMENGARDE OF TOURS ===")
    
    # Check if death date entity exists
    death_date_entities = [e for e in gsw_data["entity_nodes"] if "20 March 851" in e["name"]]
    print(f"✓ Death date entity found: {len(death_date_entities) > 0}")
    if death_date_entities:
        print(f"  Entity: {death_date_entities[0]['name']} (ID: {death_date_entities[0]['id']})")
    
    # Check if Ermengarde entity exists
    ermengarde_entities = [e for e in gsw_data["entity_nodes"] if "Ermengarde" in e["name"]]
    print(f"✓ Ermengarde entity found: {len(ermengarde_entities) > 0}")
    if ermengarde_entities:
        print(f"  Entity: {ermengarde_entities[0]['name']} (ID: {ermengarde_entities[0]['id']})")
    
    # Check for "died on" verb phrase relationships
    died_on_vps = [vp for vp in gsw_data["verb_phrase_nodes"] if "died" in vp["phrase"].lower()]
    print(f"✓ 'Died on' verb phrases found: {len(died_on_vps)}")
    
    for vp in died_on_vps:
        print(f"  Verb phrase: '{vp['phrase']}' (ID: {vp['id']})")
        for q in vp["questions"]:
            print(f"    Q: {q['text']} → A: {q['answers']}")
    
    # Check if the connection exists
    if ermengarde_entities and death_date_entities and died_on_vps:
        ermengarde_id = ermengarde_entities[0]["id"]
        death_date_id = death_date_entities[0]["id"]
        
        connection_found = False
        for vp in died_on_vps:
            for q in vp["questions"]:
                if (ermengarde_id in q["answers"] and "when" in q["text"].lower()) or \
                   (death_date_id in q["answers"] and "who" in q["text"].lower()):
                    connection_found = True
                    print(f"✅ DEATH DATE CONNECTION FOUND: {q['text']} → {q['answers']}")
        
        if not connection_found:
            print("❌ DEATH DATE CONNECTION MISSING")
    
    return len(died_on_vps) > 0 and len(ermengarde_entities) > 0 and len(death_date_entities) > 0


def analyze_edith_extraction(gsw_data):
    """Analyze if Edith Carlmar extraction captures key biographical details."""
    print("\n=== ANALYZING EDITH CARLMAR ===")
    
    # Check birth date
    birth_entities = [e for e in gsw_data["entity_nodes"] if "15 November 1911" in e["name"]]
    print(f"✓ Birth date entity found: {len(birth_entities) > 0}")
    
    # Check death date  
    death_entities = [e for e in gsw_data["entity_nodes"] if "17 May 2003" in e["name"]]
    print(f"✓ Death date entity found: {len(death_entities) > 0}")
    
    # Check films
    film_entities = [e for e in gsw_data["entity_nodes"] if any(film in e["name"] for film in ["Fjols til fjells", "Aldri annet enn bråk", "Ung flukt"])]
    print(f"✓ Film entities found: {len(film_entities)}")
    for film in film_entities:
        print(f"  Film: {film['name']}")
    
    # Check director relationships
    director_vps = [vp for vp in gsw_data["verb_phrase_nodes"] if "direct" in vp["phrase"].lower()]
    print(f"✓ Director verb phrases found: {len(director_vps)}")
    
    # Check born/died relationships
    born_vps = [vp for vp in gsw_data["verb_phrase_nodes"] if "born" in vp["phrase"].lower()]
    died_vps = [vp for vp in gsw_data["verb_phrase_nodes"] if "died" in vp["phrase"].lower()]
    print(f"✓ Birth/Death verb phrases: born={len(born_vps)}, died={len(died_vps)}")
    
    return len(birth_entities) > 0 and len(death_entities) > 0 and len(film_entities) > 0


def test_failing_cases():
    """Test the specific failing cases with the new prompt."""
    print("🧪 Testing Factual Extraction on Original Failing Cases")
    
    # Define the failing case documents
    failing_documents = [
        {
            "title": "Ermengarde of Tours",
            "text": "Ermengarde of Tours (d. 20 March 851) was the daughter of Hugh of Tours, a member of the Etichonen family. In October 821 in Thionville, she married the Carolingian Emperor Lothair I of the Franks (795–855). In 849, two years before her death, she made a donation to the abbey Erstein in the Elsass, in which she is buried. Lothair and Ermengarde had eight children:"
        },
        {
            "title": "Edith Carlmar", 
            "text": "Edith Carlmar (Edith Mary Johanne Mathiesen) (15 November 1911 – 17 May 2003) was an Norwagian actress and Norway's first female film director. She is known for films such as \"Fjols til fjells\" (1957), \"Aldri annet enn bråk\" (1954), and \"Ung flukt\" (1959). Her 1949 film, \"Døden er et kjærtegnDeath is a Caress\"), is considered to be Norway's first film noir. The last film she directed, \"Ung Flukt\", introduced Liv Ullmann, Norway's most famous actor internationally, to the silver screen. Carlmar came from a poor family in the working class districts of East Oslo. However, she did manage to take dancing classes and made her debut on stage at the age of 15. In the theater she met Otto Carlmar whom she married three years later. From 1936 she worked as an actress in various theatres. Here she met the film director Tancred Ibsen who introduced her to the world of cinema. In 1949 she and her husband started Carlmar Film A/S, and began writing scripts, directing and producing films. They made ten feature films over a ten-year period. After a decade of film-making Carlmar retired as a director. In the last part of her life she accepted only minor acting roles in plays and movies."
        }
    ]
    
    # Initialize GSWProcessor with FACTUAL prompt type
    processor = GSWProcessor(
        model_name="gpt-4o",
        enable_coref=False,
        enable_chunking=False,
        chunk_size=1,
        overlap=0,
        enable_context=False,
        enable_spacetime=True,
        prompt_type=PromptType.FACTUAL,
    )
    
    # Extract document texts
    document_texts = [doc["text"] for doc in failing_documents]
    
    print(f"Processing {len(document_texts)} failing case documents...")
    
    # Process documents
    gsw_structures = processor.process_documents(document_texts)
    
    # Analyze results
    results = []
    for i, doc in enumerate(failing_documents):
        print(f"\n{'='*60}")
        print(f"TESTING: {doc['title']}")
        print(f"{'='*60}")
        
        try:
            doc_data = gsw_structures[i]
            if doc_data:
                chunk_key = list(doc_data.keys())[0]
                chunk_data = doc_data[chunk_key]
                
                if chunk_data.get("gsw"):
                    # Convert to dict for analysis
                    gsw_dict = {
                        "entity_nodes": [
                            {
                                "id": entity.id,
                                "name": entity.name,
                                "roles": [
                                    {
                                        "role": role.role,
                                        "states": role.states
                                    } for role in entity.roles
                                ]
                            } for entity in chunk_data["gsw"].entity_nodes
                        ],
                        "verb_phrase_nodes": [
                            {
                                "id": vp.id,
                                "phrase": vp.phrase,
                                "questions": [
                                    {
                                        "id": q.id,
                                        "text": q.text,
                                        "answers": q.answers
                                    } for q in vp.questions
                                ]
                            } for vp in chunk_data["gsw"].verb_phrase_nodes
                        ]
                    }
                    
                    # Analyze based on document type
                    if "Ermengarde" in doc["title"]:
                        success = analyze_ermengarde_extraction(gsw_dict)
                    elif "Edith" in doc["title"]:
                        success = analyze_edith_extraction(gsw_dict)
                    else:
                        success = True
                    
                    results.append({
                        "title": doc["title"],
                        "success": success,
                        "gsw_data": gsw_dict
                    })
                    
                    print(f"✅ GSW EXTRACTION SUCCESSFUL for {doc['title']}")
                else:
                    print(f"❌ GSW EXTRACTION FAILED for {doc['title']}")
                    results.append({"title": doc["title"], "success": False})
            else:
                print(f"❌ NO DOCUMENT DATA for {doc['title']}")
                results.append({"title": doc["title"], "success": False})
        except Exception as e:
            print(f"❌ ERROR processing {doc['title']}: {e}")
            results.append({"title": doc["title"], "success": False, "error": str(e)})
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    successful = sum(1 for r in results if r.get("success", False))
    total = len(results)
    
    for result in results:
        status = "✅ PASS" if result.get("success", False) else "❌ FAIL" 
        print(f"{status}: {result['title']}")
    
    print(f"\nSUCCESS RATE: {successful}/{total} ({successful/total*100:.1f}%)")
    
    return results


if __name__ == "__main__":
    test_failing_cases()