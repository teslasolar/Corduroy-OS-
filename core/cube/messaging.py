"""
KONOMI Core - Cube Messaging
Message passing between cube nodes
"""

from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Cube


def send_message(
    self: 'Cube',
    from_vertex: str,
    to_vertex: str,
    message: str
) -> str:
    """
    Send message from one node to another.

    Message is processed through the receiving LLM.

    Args:
        from_vertex: Source node
        to_vertex: Destination node
        message: Message content

    Returns:
        Processed message from receiving LLM

    Raises:
        ValueError: If no edge exists between nodes
    """
    if to_vertex not in self.edges[from_vertex]:
        raise ValueError(
            f"No edge from {from_vertex} to {to_vertex}. "
            f"Available edges: {self.edges[from_vertex]}"
        )

    # Get receiving LLM
    if to_vertex == 'central':
        receiver = self.c
    else:
        receiver = self.v[to_vertex]

    # Process message through LLM
    response = receiver.p(message)

    # Log message
    self.message_history.append({
        'from': from_vertex,
        'to': to_vertex,
        'message': message,
        'response': response
    })

    return response


def broadcast(
    self: 'Cube',
    from_vertex: str,
    message: str
) -> Dict[str, str]:
    """
    Broadcast message from one node to all connected nodes.

    Args:
        from_vertex: Source node
        message: Message to broadcast

    Returns:
        Dict of {node: response} from each connected node
    """
    responses = {}

    for to_vertex in self.edges[from_vertex]:
        response = send_message(self, from_vertex, to_vertex, message)
        responses[to_vertex] = response

    return responses


def process_at_vertex(self: 'Cube', vertex: str, data: str) -> str:
    """
    Process data at a specific vertex LLM.

    Args:
        vertex: Vertex name or 'central'
        data: Data to process

    Returns:
        Processed result
    """
    if vertex == 'central':
        llm = self.c
    elif vertex in self.v:
        llm = self.v[vertex]
    else:
        raise ValueError(f"Invalid vertex: {vertex}")

    return llm.p(data)
