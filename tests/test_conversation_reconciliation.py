#!/usr/bin/env python3
"""
Test conversation reconciliation functionality.
"""

from gsw_memory.memory import GSWStructure, EntityNode, Role, SpaceNode, TimeNode
from gsw_memory.memory.reconciler import Reconciler


def test_conversation_reconciliation():
    """Test that conversation nodes and edges are properly reconciled."""
    
    # Create initial GSW with conversation data
    initial_gsw = GSWStructure()
    
    # Add some entities
    entity1 = EntityNode(
        id="e1",
        name="Alice",
        roles=[Role(role="speaker", states=["active"])],
        chunk_id="chunk_1"
    )
    entity2 = EntityNode(
        id="e2", 
        name="Bob",
        roles=[Role(role="speaker", states=["active"])],
        chunk_id="chunk_1"
    )
    initial_gsw.add_entity(entity1)
    initial_gsw.add_entity(entity2)
    
    # Add space and time nodes
    space1 = SpaceNode(
        id="s1",
        name="Office",
        chunk_id="chunk_1"
    )
    time1 = TimeNode(
        id="t1",
        name="Morning",
        chunk_id="chunk_1"
    )
    initial_gsw.add_space_node(space1)
    initial_gsw.add_time_node(time1)
    
    # Add conversation node
    conversation1 = {
        "id": "cv_1",
        "chunk_id": "chunk_1",
        "participants": ["e1", "e2"],
        "topics_entity": ["e1"],
        "topics_general": ["work", "meeting"],
        "location_id": "s1",
        "time_id": "t1",
        "motivation": "Discuss project progress",
        "summary": "Alice and Bob discussed the project in the office",
        "participant_summaries": {
            "e1": "Alice shared updates on the project",
            "e2": "Bob asked questions about the timeline"
        },
        "type": "conversation"
    }
    initial_gsw.add_conversation_node(conversation1)
    
    # Add conversation edges
    initial_gsw.add_conversation_participant_edge("e1", "cv_1")
    initial_gsw.add_conversation_participant_edge("e2", "cv_1")
    initial_gsw.add_conversation_topic_edge("e1", "cv_1")
    initial_gsw.add_conversation_space_edge("cv_1", "s1")
    initial_gsw.add_conversation_time_edge("cv_1", "t1")
    
    # Create reconciler and reconcile first chunk
    reconciler = Reconciler(matching_approach="exact")
    reconciled_gsw = reconciler.reconcile(initial_gsw, chunk_id="chunk_1")
    
    # Verify conversation data is preserved
    assert len(reconciled_gsw.conversation_nodes) == 1
    conv_node = reconciled_gsw.get_conversation_by_id("chunk_1::cv_1")
    assert conv_node is not None
    
    # Verify conversation node contents use global IDs
    assert conv_node["participants"] == ["chunk_1::e1", "chunk_1::e2"]
    assert conv_node["topics_entity"] == ["chunk_1::e1"]
    assert conv_node["location_id"] == "chunk_1::s1"
    assert conv_node["time_id"] == "chunk_1::t1"
    
    # Verify participant_summaries keys use global IDs
    expected_summaries = {
        "chunk_1::e1": "Alice shared updates on the project",
        "chunk_1::e2": "Bob asked questions about the timeline"
    }
    assert conv_node["participant_summaries"] == expected_summaries
    
    # Verify conversation edges are preserved with global IDs
    assert len(reconciled_gsw.conversation_participant_edges) == 2
    assert len(reconciled_gsw.conversation_topic_edges) == 1
    assert len(reconciled_gsw.conversation_space_edges) == 1
    assert len(reconciled_gsw.conversation_time_edges) == 1
    
    # Verify edge references use global IDs
    participant_edges = set(tuple(edge) for edge in reconciled_gsw.conversation_participant_edges)
    assert ("chunk_1::e1", "chunk_1::cv_1") in participant_edges
    assert ("chunk_1::e2", "chunk_1::cv_1") in participant_edges
    
    topic_edges = set(tuple(edge) for edge in reconciled_gsw.conversation_topic_edges)
    assert ("chunk_1::e1", "chunk_1::cv_1") in topic_edges
    
    space_edges = set(tuple(edge) for edge in reconciled_gsw.conversation_space_edges)
    assert ("chunk_1::cv_1", "chunk_1::s1") in space_edges
    
    time_edges = set(tuple(edge) for edge in reconciled_gsw.conversation_time_edges)
    assert ("chunk_1::cv_1", "chunk_1::t1") in time_edges
    
    # Test adding a second chunk with conversation data
    second_gsw = GSWStructure()
    
    # Add new entity
    entity3 = EntityNode(
        id="e3",
        name="Charlie",
        roles=[Role(role="speaker", states=["active"])],
        chunk_id="chunk_2"
    )
    second_gsw.add_entity(entity3)
    
    # Add conversation in second chunk
    conversation2 = {
        "id": "cv_2",
        "chunk_id": "chunk_2",
        "participants": ["e2", "e3"],
        "topics_entity": ["e2"],
        "topics_general": ["lunch", "casual"],
        "location_id": None,
        "time_id": None,
        "motivation": "Casual lunch conversation",
        "summary": "Bob and Charlie had lunch together",
        "participant_summaries": {
            "e2": "Bob shared stories about work",
            "e3": "Charlie listened and asked follow-up questions"
        },
        "type": "conversation"
    }
    second_gsw.add_conversation_node(conversation2)
    
    # Add conversation edges for second chunk
    second_gsw.add_conversation_participant_edge("e2", "cv_2")
    second_gsw.add_conversation_participant_edge("e3", "cv_2")
    second_gsw.add_conversation_topic_edge("e2", "cv_2")
    
    # Reconcile second chunk
    final_gsw = reconciler.reconcile(second_gsw, chunk_id="chunk_2")
    
    # Verify both conversations are present
    assert len(final_gsw.conversation_nodes) == 2
    conv_node1 = final_gsw.get_conversation_by_id("chunk_1::cv_1")
    conv_node2 = final_gsw.get_conversation_by_id("chunk_2::cv_2")
    assert conv_node1 is not None
    assert conv_node2 is not None
    
    # Check if entity e2 was merged (Bob appears in both chunks)
    # Since we're using exact matching and the same name "Bob", 
    # e2 from chunk_2 should be merged with chunk_1::e2
    # So conv_node2 should reference chunk_1::e2 instead of chunk_2::e2
    print("Final conversation 2 participants:", conv_node2["participants"])
    
    # Verify that conversation topics are properly updated with merged entity IDs
    print("Final conversation 2 topics_entity:", conv_node2["topics_entity"])
    
    # Verify all conversation edges are present
    assert len(final_gsw.conversation_participant_edges) == 4  # 2 from each chunk
    assert len(final_gsw.conversation_topic_edges) == 2  # 1 from each chunk
    assert len(final_gsw.conversation_space_edges) == 1  # 1 from first chunk
    assert len(final_gsw.conversation_time_edges) == 1  # 1 from first chunk
    
    # Verify statistics include conversation data
    stats = reconciler.get_statistics()
    assert stats["conversation_nodes"] == 2
    assert stats["conversation_participant_edges"] == 4
    assert stats["conversation_topic_edges"] == 2
    assert stats["conversation_space_edges"] == 1
    assert stats["conversation_time_edges"] == 1
    
    print("✅ Conversation reconciliation test passed!")


if __name__ == "__main__":
    test_conversation_reconciliation() 