import xml.etree.ElementTree as ET

graphml_path = r"C:\Users\LENOVO\GRAPHRAG\community_clusters.graphml"

# Parse the GraphML file
tree = ET.parse(graphml_path)
root = tree.getroot()

# Find all edge elements
edges = root.findall(".//{http://graphml.graphdrawing.org/xmlns}edge")

# Print edge count
print("✅ Number of edges found:", len(edges))

# Print first 5 edges if available
for edge in edges[:5]:
    print("Edge:", edge.attrib)
