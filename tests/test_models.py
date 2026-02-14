# Create tests/test_models.py
from gsw_memory.memory import EntityNode, GSWStructure, Role


def test_gsw_structure_creation():
    workspace = GSWStructure()

    entity = EntityNode(
        id="test_entity",
        name="Test Person",
        roles=[Role(role="TestRole", states=["active"])],
    )

    workspace.add_entity(entity)
    assert len(workspace.entity_nodes) == 1
    assert workspace.get_entity_by_id("test_entity") is not None


if __name__ == "__main__":
    test_gsw_structure_creation()
    print("✅ Models working correctly!")
