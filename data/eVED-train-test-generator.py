import random
import os
from datasets import (
    load_dataset, 
    concatenate_datasets,
    DatasetDict
)


# Configurações principais
TEST_FRACTION = 0.4 
NUM_CPUS = max(1, os.cpu_count() - 2) # Usa quase todos os núcleos
VEHICLE_TYPES = {
    'ice':  lambda batch: [v == 'ICE'  for v in batch['Vehicle Type']],
    'hev':  lambda batch: [v == 'HEV'  for v in batch['Vehicle Type']],
    'phev': lambda batch: [e == 'PHEV' for e in batch['EnergyType']],
    'ev':   lambda batch: [e == 'EV'   for e in batch['EnergyType']],
}

def create_grouped_split_from_hub(type_vehicles, test_fraction, random_state):
    """
    Função para os lidar a com a divisão de conjuntos de treino e teste nos dados do eVED,
    garantindo que no teste e treino contenham um certo percentual de cada tipo de veículo.
    
    A esolha dos conjuntos de veiculos pro treino e teste é aleatória. O parâmetro `random_state`
    define a semente para a aleatoriedade, garantindo reprodutibilidade.

    Parâmetros:
    - type_vehicles (dict): Dicionário com filtros para cada tipo de veículo.
    - test_fraction (float): Fração do dataset a ser usada como teste.
    - random_state (int): Semente para a aleatoriedade.
    Retorna:
    - output_dir (str): Diretório onde o dataset dividido foi salvo.
    """
    print(f"\n--- 🚀 Iniciando split para RANDOM_STATE = {random_state} ---")
    
    ds_base = load_dataset('jwsouza13/eVED', split='train') # eVED completo no HF
    
    train_datasets = []
    test_datasets = []
    random.seed(random_state) # Garante amostragem reprodutível

    for key, filter_lambda in type_vehicles.items():
        print(f"Processando tipo: {key}...")
        
        # isola o tipo de veículo
        ds_type_specific = ds_base.filter(
            filter_lambda,
            batched=True,
            desc=f"Filtrando tipo {key}",
            num_proc=NUM_CPUS
        )
        
        ids = ds_type_specific.unique('VehId')
        if not ids:
            print(f"  Aviso: Nenhum 'VehId' único encontrado para {key}. Pulando.")
            continue
            
        # Sorteia os IDs que irão para o conjunto de TESTE
        n_samples_test = int(len(ids) * test_fraction)
        ids_test_set = set(random.sample(ids, n_samples_test))
        print(f"  Dividindo {len(ids)} veículos (Teste: {n_samples_test}, Treino: {len(ids) - n_samples_test})")
        
        ds_train_part = ds_type_specific.filter(
            lambda batch: [veh_id not in ids_test_set for veh_id in batch['VehId']],
            batched=True,
            desc=f"Filtrando treino {key}",
            num_proc=NUM_CPUS 
        )
        
        ds_test_part = ds_type_specific.filter(
            lambda batch: [veh_id in ids_test_set for veh_id in batch['VehId']],
            batched=True,
            desc=f"Filtrando teste {key}",
            num_proc=NUM_CPUS
        )
        
        train_datasets.append(ds_train_part)
        test_datasets.append(ds_test_part)
        
        # melhorar o uso de memória
        del ds_type_specific, ds_train_part, ds_test_part, ids, ids_test_set

    # Concatena os datasets finais (única concatenação)
    print("Concatenando datasets finais...")
    ds_train = concatenate_datasets(train_datasets)
    ds_test = concatenate_datasets(test_datasets)
    dataset_hf = DatasetDict({'train': ds_train, 'test': ds_test})
    
    print(dataset_hf)
    
    # Salva o resultado final em disco
    per_frac = int(test_fraction * 100)
    name = f'eVED_train_test_FRAC-TEST_{per_frac}_STATE_{random_state}'
    dataset_hf.save_to_disk(name)
    
    del ds_base, train_datasets, test_datasets, ds_train, ds_test, dataset_hf
    
    print(f"--- ✅ Concluído split para RANDOM_STATE = {random_state} ---")
    return name


# config 1: fraction de teste 0.5 e random_state 42
output_42 = create_grouped_split_from_hub(
    VEHICLE_TYPES, TEST_FRACTION = .5, random_state=42
)

# config 2: Um "state" diferente
output_100 = create_grouped_split_from_hub(
    VEHICLE_TYPES, TEST_FRACTION, random_state=100
)

# output_1234 = create_grouped_split_from_hub(
#     BASE_DATASET_REPO, VEHICLE_TYPE_FILTERS, TEST_FRACTION, random_state=1234
# )

# Para subir o dataset para o HF. 
# from datasets import load_from_disk
# dataset = load_from_disk(output_42)