import xml.etree.ElementTree as ET

def is_graphml_file(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        # Check if the root tag is <graphml>
        return root.tag.endswith("graphml")
    except ET.ParseError:
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

# Example usage
file_path = r"C:\Users\LENOVO\GRAPHRAG\ragtest1\output\create_base_extracted_entities.graphml"
if is_graphml_file(file_path):
    print("The file is in GraphML format.")
else:
    print("The file is NOT in GraphML format.")
