"""
Pacote de visualiza��o de trajetos do dataset eVED.

Este pacote fornece ferramentas para visualizar anima��es interativas
dos trajetos de ve�culos do dataset eVED usado no FLEVEn.

Exemplo de uso:
    >>> from data.eVED_animation import visualize_trip, EVEDVisualizer
    >>> visualize_trip(client_id=0, trip_id=706.0)

    >>> viz = EVEDVisualizer()
    >>> clientes = viz.get_available_clients('train')
    >>> trips = viz.get_available_trips(0, 'train')
    >>> viz.create_animated_map(0, trip_id=trips[0])
"""

from .eved_vizu import EVEDVisualizer, visualize_trip

__all__ = ['EVEDVisualizer', 'visualize_trip']
__version__ = '1.5.0'
