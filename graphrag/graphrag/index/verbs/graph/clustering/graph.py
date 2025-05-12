import logging
import xml.etree.ElementTree as ET
import networkx as nx
log = logging.getLogger(__name__)

def load_graphml1(file_path: str) -> ET.Element:
    """Load and parse a GraphML file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            xml_content = file.read().strip()
        
        if not xml_content.startswith('<?xml'):
            raise ValueError("The file does not contain a valid XML declaration.")

        

        # Parse the XML content and return the root element
        return ET.fromstring(xml_content)

    except ET.ParseError as e:
        log.error(f"Error parsing XML: {e}")
    except FileNotFoundError:
        log.error(f"File not found: {file_path}")
    except Exception as e:
        log.error(f"An error occurred: {e}")

    return None  # Return None on failure for clarity

# Load the GraphML file
graphml_file_path = r'C:\Users\LENOVO\GRAPHRAG\ragtest1\output\create_base_extracted_entities.graphml'
graph_element = load_graphml1(graphml_file_path)

if graph_element is not None:
    print(graph_element)
    print("GraphML loaded successfully.")
    
else:
    print("Failed to load GraphML.")
