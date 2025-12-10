"""
Mapeamento de partition-id para client_id válido.

Este módulo gerencia o mapeamento entre partition IDs (0, 1, 2, ..., num_supernodes-1)
e client IDs válidos, pulando clientes que foram movidos para a pasta ruins/.

Clientes removidos (48 total):
- Por NaN/Inf (43): 4, 5, 8, 11, 16, 18, 19, 21, 27, 28, 29, 31, 35, 37, 44, 45,
                     53, 58, 61, 63, 69, 74, 82, 85, 89, 92, 96, 98, 99, 100,
                     124, 133, 135, 136, 139, 142, 143, 144, 151, 155, 174, 180, 209
- Por 1 trip (5): 138, 154, 156, 157, 158
"""

# Lista completa de clientes removidos (em ruins/)
REMOVED_CLIENTS = {
    4, 5, 8, 11, 16, 18, 19, 21, 27, 28, 29, 31, 35, 37, 44, 45,
    53, 58, 61, 63, 69, 74, 82, 85, 89, 92, 96, 98, 99, 100,
    124, 133, 135, 136, 138, 139, 142, 143, 144, 151, 154, 155,
    156, 157, 158, 174, 180, 209
}

# Total de clientes no dataset original
TOTAL_CLIENTS = 232

# Clientes válidos (184 total)
VALID_CLIENTS = sorted([i for i in range(TOTAL_CLIENTS) if i not in REMOVED_CLIENTS])


def get_valid_client_id(partition_id: int) -> int:
    """
    Mapeia partition_id (0, 1, 2, ...) para client_id válido.

    Args:
        partition_id: ID da partição atribuído pelo Flower (0-indexed)

    Returns:
        client_id: ID do cliente válido (pula clientes removidos)

    Raises:
        IndexError: Se partition_id >= número de clientes válidos (184)

    Example:
        >>> get_valid_client_id(0)  # Retorna 0 (client_0 é válido)
        0
        >>> get_valid_client_id(4)  # Retorna 6 (pula client_4 e client_5 que estão em ruins/)
        6
    """
    if partition_id >= len(VALID_CLIENTS):
        raise IndexError(
            f"partition_id {partition_id} está fora do alcance. "
            f"Existem apenas {len(VALID_CLIENTS)} clientes válidos. "
            f"Ajuste 'num-supernodes' em pyproject.toml para <= {len(VALID_CLIENTS)}"
        )

    return VALID_CLIENTS[partition_id]


def get_total_valid_clients() -> int:
    """Retorna o número total de clientes válidos (184)."""
    return len(VALID_CLIENTS)


def is_client_valid(client_id: int) -> bool:
    """Verifica se um client_id é válido (não está em ruins/)."""
    return client_id not in REMOVED_CLIENTS


# Para debug
if __name__ == "__main__":
    print(f"Total de clientes válidos: {get_total_valid_clients()}")
    print(f"Total de clientes removidos: {len(REMOVED_CLIENTS)}")
    print(f"\nPrimeiros 20 mapeamentos:")
    print("partition_id -> client_id")
    print("-" * 30)
    for pid in range(min(20, len(VALID_CLIENTS))):
        cid = get_valid_client_id(pid)
        print(f"{pid:3d}          -> {cid:3d}")
