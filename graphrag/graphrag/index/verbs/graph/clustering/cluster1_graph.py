# Licensed under the MIT License

"""A module containing cluster_graph, apply_clustering, and run_layout methods definition."""
import xml.etree.ElementTree as ET
import logging
from enum import Enum
import os
from random import Random
from typing import Any, cast, Dict, List, Union
import xml.etree.ElementTree as ET
from graphrag.index.verbs.entities.extraction.entity_extract import entity_extract
import networkx as nx
from networkx import edges
import pandas as pd
from datashaper import NoopVerbCallbacks, TableContainer, VerbCallbacks, VerbInput, Workflow, progress_iterable, verb
from graphrag.index.utils import gen_uuid
from datashaper import VerbManager
from graphrag.index.verbs.graph.clustering.custom_typing import Communities
from graphrag.index.verbs.graph.clustering.graph import load_graphml1 

import leidenalg as la

# Set up logging
log = logging.getLogger(__name__)
verb_manager = VerbManager.get()

import igraph as ig


def run_leiden(graph: nx.Graph, strategy: Dict[str, Any]) -> Dict[int, Dict[str, List[str]]]:
    """Runs the Leiden algorithm on the given graph and returns the community structure."""
    # Convert MultiGraph to Graph if necessary
    if isinstance(graph, nx.MultiGraph):
        graph = nx.Graph(graph)  # Convert MultiGraph to Graph

    # Convert NetworkX graph to igraph
    ig_graph = ig.Graph.TupleList(graph.edges(data=True), directed=False)

    # Use leidenalg to find partitions
    partition = la.find_partition(ig_graph, la.ModularityVertexPartition)
    
    # Construct the clusters
    clusters = {0: {}}
    for idx, community in enumerate(partition):
        clusters[0][idx] = [str(node) for node in community]
    print("DEBUG: Clustering output:", clusters)
    return clusters












class VerbWrapper:
    def __init__(self, func):
        self.func = func
        self.treats_input_tables_as_immutable = False  # Set default behavior

def verb(name: str):
    """Decorator to register a function as a verb."""
    def decorator(func):
        if verb_manager.get_verb(name) is not None:
            log.warning(f"Verb '{name}' is already registered. Skipping registration.")
            return func  # Skip registration, but return the function
        wrapped_func = VerbWrapper(func)
        wrapped_func.name = name
        verb_manager.register(wrapped_func)
        log.info(f"Registered verb: {name}")
        return wrapped_func
    return decorator





verb_name = "cluster_graph"


data_frame = pd.read_csv("/workstation/APPGraph/APPGraph/GRAPHRAG/ragtest1/input/appreview - Sheet1.csv")

class VerbInput:
    def __init__(self, input_file):
        if isinstance(input_file, str):
            self.data_frame = pd.read_csv(input_file)
        elif isinstance(input_file, pd.DataFrame):
            self.data_frame = input_file
        else:
            raise ValueError("Expected a CSV file path or DataFrame.")

    def get_input(self):
        return self.data_frame 

my_input = VerbInput(data_frame)
my_callbacks = NoopVerbCallbacks()
strategy = {
    "type": "leiden",
    "max_cluster_size": 10,
    "use_lcc": True,
    "seed": 0xDEADBEEF,
    "levels": [0, 1]
}


verb_function_details = verb_manager.get_verb(verb_name)

# Check if we got a valid VerbDetails object
if verb_function_details is not None:
    # Access the actual function
    verb_function = verb_function_details
    
    # Now you can call the verb function
    result = verb_function(
        input=my_input,
        callbacks=my_callbacks,
        strategy=strategy,
        column='text',  # Ensure this column name matches your DataFrame
        to='clustered_graph',
        level_to='level'
    )
    
else:
    log.error(f"Verb '{verb_name}' not found.")



class GraphCommunityStrategyType(str, Enum):
    """GraphCommunityStrategyType class definition."""
    leiden = "leiden"

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'
#############################################################################################################################

def run_layout(strategy: dict[str, Any], graphml_or_graph: Union[str, nx.Graph]) -> List:
    if isinstance(graphml_or_graph, str):
        # Check if the input is a file path or a GraphML string
        if os.path.isfile(graphml_or_graph):
            log.info(f"Loading graph from file: {graphml_or_graph}")
            graphml_string = load_graphml1(graphml_or_graph)
            # This will return XML content or None
            if graphml_string is None:
                log.error("Failed to load graph from file. Exiting.")
                return []  # Exit if loading fails
        else:
            log.info(f"Received GraphML string: {graphml_or_graph}")
            graphml_string = graphml_or_graph  # Treat as a GraphML string
    else:
        log.info(f"Received NetworkX Graph object: {graphml_or_graph}")
        graph = graphml_or_graph
        if len(graph.nodes) == 0:
            log.warning("Graph has no nodes")
            return []
        return return_clustering(strategy, graph)  # type: ignore
   
    # Proceed with clustering using the graphml_string
    if graphml_string is not None:
        if isinstance(graphml_string, ET.Element):  # Check if it's an Element
            graphml_string = ET.tostring(graphml_string, encoding='unicode')  # Convert to string
        graph = nx.parse_graphml(graphml_string) 
         # Log the type of each graph
    
        print(f"Processing graph ", {type(graph)})  # Log the type of the graph
           # Parse the GraphML string
    else:
        log.error("GraphML string is empty or None.")
        return []

    if len(graph.nodes) == 0:
        log.warning("Graph has no nodes")
        return []

    print("Graph loaded successfully. Proceeding with clustering...")

    # Clustering logic
    clusters: dict[int, dict[str, list[str]]] = {}
    strategy_type = strategy.get("type", GraphCommunityStrategyType.leiden)

    match strategy_type:
        case GraphCommunityStrategyType.leiden:
            
            clusters = run_leiden(graph, strategy)
        case _:
            raise ValueError(f"Unknown clustering strategy {strategy_type}")
    
    results: Communities = []
    
    for level in clusters:
        for cluster_id, nodes in clusters[level].items():
            results.append((level, cluster_id, nodes))
    return results

##########################################################################################################################################

def apply_clustering(
    graphml: str, communities: Communities, level: int = 0, seed: int | None = None
) -> nx.Graph:
    """Apply clustering to a graphml string."""
    random = Random(seed)
    graph = nx.parse_graphml(graphml)
     # Create a mapping from index to actual node names
    node_names = list(graph.nodes())
    
    for community_level, community_id, nodes in communities:
        if level == community_level:
            for node in nodes:
                # Use node's index to get the actual name from the graph
                if node.isdigit():  # Check if node is a digit
                    actual_node = node_names[int(node)]  # Convert to int and get the name
                else:
                    actual_node = node  # Use node as is if it's already a name
                
                # Ensure the actual node exists in the graph
                if actual_node in graph:
                    graph.nodes[actual_node]["cluster"] = community_id
                    graph.nodes[actual_node]["level"] = level
                else:
                    print(f"Warning: Node '{actual_node}' not found in graph.")
    

    for node_degree in graph.degree:
        graph.nodes[str(node_degree[0])]["degree"] = int(node_degree[1])

    for index, node in enumerate(graph.nodes()):
        graph.nodes[node]["human_readable_id"] = index
        graph.nodes[node]["id"] = str(gen_uuid(random))

    for index, edge in enumerate(graph.edges()):
        graph.edges[edge]["id"] = str(gen_uuid(random))
        graph.edges[edge]["human_readable_id"] = index
        graph.edges[edge]["level"] = level

    return graph



@verb(name=verb_name)
def cluster_graph(
    input: VerbInput,
    callbacks: VerbCallbacks,
    strategy: dict[str, Any],
    column: str,
    to: str,
    level_to: str | None = None,
    **_kwargs,
) -> TableContainer:
    """
    Apply a hierarchical clustering algorithm to a graph.
    ...
    """
    graphml_file_path = '/workstation/APPGraph/APPGraph/GRAPHRAG/ragtest1/output/create_base_extracted_entities.graphml'
    output_df = input.get_input()
    
   

       
    results = output_df[column].apply(lambda graph: run_layout(strategy, graphml_file_path))
    print("DEBUG: First 5 results from run_layout:", results.head().tolist())
    community_map_to = "communities"
    output_df[community_map_to] = results

    level_to = level_to or f"{to}_level"
    output_df[level_to] = output_df.apply(
        lambda x: list({level for level, _, _ in x[community_map_to]}), axis=1
    )
    output_df[to] = [None] * len(output_df)

    num_total = len(output_df)
    seed = strategy.get("seed", Random().randint(0, 0xFFFFFFFF))

    graph_level_pairs_column: list[list[tuple[int, str]]] = []
    for _, row in progress_iterable(
        output_df.iterrows(), callbacks.progress, num_total
    ):
        levels = row[level_to]
        graph_level_pairs: list[tuple[int, str]] = []

        for level in levels:
            graph = "\n".join(
                nx.generate_graphml(
                    apply_clustering(
                        cast(str, row[column]),
                        cast(Communities, row[community_map_to]),
                        level,
                        seed=seed,
                    )
                )
            )
            graph_level_pairs.append((level, graph))
        graph_level_pairs_column.append(graph_level_pairs)
    print("DEBUG: Sample graph_level_pairs_column:", graph_level_pairs_column[:5])
   
    output_df[to] = graph_level_pairs_column
    output_df = output_df.explode(to, ignore_index=True)
    print("DEBUG: Shape of output_df before assignment:", output_df.shape)
    print("DEBUG: Type of output_df[to]:", type(output_df[to]))
    print("DEBUG: First 5 values of output_df[to]:", output_df[to].head().tolist())

    expanded_df = pd.DataFrame(output_df[to].tolist(), index=output_df.index)
    print("DEBUG: Shape of expanded DataFrame:", expanded_df.shape)

    try:
        output_df[[level_to, to]] = pd.DataFrame(output_df[to].tolist(), index=output_df.index)
    except ValueError as e:
        log.error("ValueError during assignment: %s", e)
        raise

    output_df.drop(columns=[community_map_to], inplace=True)

    return TableContainer(table=output_df)






