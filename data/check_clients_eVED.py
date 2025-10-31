"""
Script para verificar quais clientes do eVED têm dados suficientes para treinamento.
Agora usa apenas a pasta train/ com split temporal.
"""
import pandas as pd
from pathlib import Path

def check_client_data(client_id, base_dir, sequence_length=50, prediction_length=10, train_test_split=0.8):
    """
    Verifica se um cliente tem dados suficientes.

    Args:
        client_id: ID do cliente
        base_dir: Diretório base (EVED_Clients/)
        sequence_length: Tamanho da janela de entrada
        prediction_length: Número de passos à frente
        train_test_split: Proporção de trips para treino (padrão: 0.8)
    """
    client_dir = base_dir / "train" / f"client_{client_id}"

    if not client_dir.exists():
        return False, "Diretório não encontrado"

    trip_files = list(client_dir.glob("trip_*.parquet"))
    trip_files = [f for f in trip_files if 'metadata' not in f.name.lower()]

    if not trip_files:
        return False, "Sem arquivos de trip"

    # Ordena para manter ordem temporal
    trip_files = sorted(trip_files, key=lambda x: x.name)

    # Calcula split temporal
    split_idx = int(len(trip_files) * train_test_split)
    if split_idx == 0:
        split_idx = 1
    if split_idx >= len(trip_files):
        split_idx = len(trip_files) - 1

    train_files = trip_files[:split_idx]
    test_files = trip_files[split_idx:]

    try:
        # Carrega dados
        train_dfs = [pd.read_parquet(f) for f in train_files]
        test_dfs = [pd.read_parquet(f) for f in test_files]

        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)

        # Seleciona features
        feature_columns = ['Vehicle Speed[km/h]']
        target_column = 'Energy_Consumption'

        if target_column not in train_df.columns:
            return False, "Target column não encontrado"

        available_features = [col for col in feature_columns if col in train_df.columns]
        if not available_features:
            return False, "Nenhuma feature disponível"

        all_columns = available_features + [target_column]

        # Limpa dados
        train_processed = train_df[all_columns].copy()
        test_processed = test_df[all_columns].copy()

        train_processed = train_processed.replace([float('inf'), float('-inf')], float('nan')).dropna()
        test_processed = test_processed.replace([float('inf'), float('-inf')], float('nan')).dropna()

        # Verifica se há dados suficientes
        min_required = sequence_length + prediction_length

        if len(train_processed) < min_required or len(test_processed) < min_required:
            return False, f"Dados insuficientes (train: {len(train_processed)}, test: {len(test_processed)}, min: {min_required})"

        return True, f"OK ({len(trip_files)} trips: {len(train_files)} treino, {len(test_files)} teste | pontos: train={len(train_processed)}, test={len(test_processed)})"

    except Exception as e:
        return False, f"Erro: {str(e)}"


if __name__ == "__main__":
    base_dir = Path(__file__).parent / "EVED_Clients"

    valid_clients = []
    invalid_clients = []

    print("Verificando clientes do eVED...")
    print("=" * 80)

    # Verifica até 232 clientes (máximo no treino)
    for client_id in range(232):
        is_valid, msg = check_client_data(client_id, base_dir)

        if is_valid:
            valid_clients.append(client_id)
            print(f"[OK] Cliente {client_id:3d}: {msg}")
        else:
            invalid_clients.append((client_id, msg))
            print(f"[FAIL] Cliente {client_id:3d}: {msg}")

    print("\n" + "=" * 80)
    print(f"\nResumo:")
    print(f"   Clientes validos: {len(valid_clients)}")
    print(f"   Clientes invalidos: {len(invalid_clients)}")

    if len(valid_clients) > 0:
        print(f"\nVoce pode usar ate {len(valid_clients)} clientes na simulacao.")
        print(f"   Ajuste no pyproject.toml:")
        print(f"   options.num-supernodes = {len(valid_clients)}  # ou menos")
        print(f"   min-nodes = {min(10, len(valid_clients))}  # ou menos")

    print("\n" + "=" * 80)
