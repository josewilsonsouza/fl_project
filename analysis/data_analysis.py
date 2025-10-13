"""
Script para testar e validar as configurações do projeto FLEVEn
Execute antes de rodar o treinamento federado.

Uso:
    python data_analysis.py
"""

import sys
from pathlib import Path
import toml
import torch
import torch.nn as nn
import torch.optim as optim
from fleven.utils import Net, load_data 

def load_config():
    """Carrega configurações do pyproject.toml"""
    try:
        path_toml = Path(__file__).parent.parent/'pyproject.toml'
        config = toml.load(path_toml)
        return config["tool"]["flwr"]["app"]["config"]
    except Exception as e:
        print(f"❌ Erro ao carregar pyproject.toml: {e}")
        sys.exit(1)

def test_data_availability(num_clients=3):
    """Verifica se os dados dos clientes existem"""
    print("\n📁 Verificando disponibilidade de dados...")
    
    missing_clients = []
    for client_id in range(1, num_clients+1):
        data_dir = Path(__file__).parent.parent
        data_dir = data_dir/"data"/f'client_{client_id}'
        
        if not data_dir.exists():
            missing_clients.append(client_id)
            print(f"  ❌ Cliente {client_id}: Diretório não encontrado - {data_dir}")
        else:
            csv_files = list(data_dir.glob("*.csv"))
            if not csv_files:
                missing_clients.append(client_id)
                print(f"  ❌ Cliente {client_id}: Nenhum arquivo CSV encontrado")
            else:
                print(f"  ✅ Cliente {client_id}: {len(csv_files)} arquivo(s) CSV encontrado(s)")
    
    if missing_clients:
        print(f"\n⚠️  Clientes sem dados: {missing_clients}")
        return False
    
    print("✅ Todos os clientes têm dados disponíveis")
    return True

def test_data_loading(config, client_id=1):
    """Testa o carregamento de dados com as configurações"""
    print(f"\n🔄 Testando carregamento de dados para cliente {client_id}...")
    
    try:
        sequence_length = int(config.get("sequence-length", 60))
        prediction_length = int(config.get("prediction-length", 10))
        batch_size = int(config.get("batch-size", 32))
        train_test_split = float(config.get("train-test-split", 0.8))
        
        trainloader, testloader, num_features = load_data(
            client_id,
            sequence_length,
            prediction_length,
            batch_size,
            train_test_split
        )
        
        print(f"  ✅ Carregamento bem-sucedido")
        print(f"  📊 Features: {num_features}")
        print(f"  📊 Batches de treino: {len(trainloader)}")
        print(f"  📊 Amostras de treino: {len(trainloader.dataset)}")
        print(f"  📊 Batches de teste: {len(testloader)}")
        print(f"  📊 Amostras de teste: {len(testloader.dataset)}")
        
        for sequences, labels in trainloader:
            print(f"  📊 Shape do batch de entrada: {sequences.shape}")
            print(f"  📊 Shape do batch de labels: {labels.shape}")
            break
        
        return True, num_features
        
    except Exception as e:
        print(f"  ❌ Erro no carregamento: {e}")
        return False, None

def test_model_creation(config, num_features):
    """Testa a criação do modelo com as configurações"""
    print("\n🔧 Testando criação do modelo...")
    
    try:
        input_size = int(config.get("input-size", 6))
        hidden_size = int(config.get("hidden-size", 50))
        prediction_length = int(config.get("prediction-length", 10))
        num_layers = int(config.get("num-layers", 1))
        
        if input_size != num_features:
            print(f"  ⚠️  AVISO: input-size ({input_size}) != num_features dos dados ({num_features})")
            print(f"       O modelo usará {num_features} features automaticamente")
        
        net = Net(
            input_size=num_features,
            hidden_size=hidden_size,
            output_size=prediction_length,
            num_layers=num_layers
        )
        
        total_params = sum(p.numel() for p in net.parameters())
        
        print(f"  ✅ Modelo criado com sucesso")
        print(f"  📊 Input size: {num_features}")
        print(f"  📊 Hidden size: {hidden_size}")
        print(f"  📊 Output size: {prediction_length}")
        print(f"  📊 Número de camadas: {num_layers}")
        print(f"  📊 Total de parâmetros: {total_params:,}")
        
        return True, net
        
    except Exception as e:
        print(f"  ❌ Erro na criação do modelo: {e}")
        return False, None

def test_forward_pass(net, config, client_id=1):
    """Testa um forward pass completo"""
    print("\n🔄 Testando forward pass...")
    
    try:
        sequence_length = int(config.get("sequence-length", 60))
        prediction_length = int(config.get("prediction-length", 10))
        batch_size = int(config.get("batch-size", 32))
        train_test_split = float(config.get("train-test-split", 0.8))
        
        trainloader, _, _ = load_data(
            client_id,
            sequence_length,
            prediction_length,
            batch_size,
            train_test_split
        )
        
        sequences, labels = next(iter(trainloader))
        
        net.eval()
        with torch.no_grad():
            outputs = net(sequences)
        
        print(f"  ✅ Forward pass bem-sucedido")
        print(f"  📊 Input shape: {sequences.shape}")
        print(f"  📊 Output shape: {outputs.shape}")
        print(f"  📊 Label shape: {labels.shape}")
        
        if outputs.shape != labels.shape:
            print(f"  ❌ ERRO: Output shape {outputs.shape} != Label shape {labels.shape}")
            return False
        
        print(f"  ✅ Dimensões estão corretas")
        return True
        
    except Exception as e:
        print(f"  ❌ Erro no forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

def evaluate(net, testloader, device):
    """Avalia o modelo no dataset de teste e retorna a perda média."""
    criterion = nn.MSELoss()
    total_loss = 0.0
    net.eval()
    with torch.no_grad():
        for sequences, labels in testloader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = net(sequences)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(testloader)
    return avg_loss

def test_training_and_evaluation_step(net, config, client_id=1):
    """Testa um passo de treinamento seguido de uma avaliação no conjunto de teste."""
    print("\n🏋️  Testando passo de Treinamento e Avaliação...")
    
    try:
        from fleven.utils import train
        
        sequence_length = int(config.get("sequence-length", 60))
        prediction_length = int(config.get("prediction-length", 10))
        batch_size = int(config.get("batch-size", 32))
        train_test_split = float(config.get("train-test-split", 0.8))
        learning_rate = float(config.get("learning-rate", 1e-5))
        max_grad_norm = float(config.get("max-grad-norm", 1.0))
        
        trainloader, testloader, _ = load_data(
            client_id,
            sequence_length,
            prediction_length,
            batch_size,
            train_test_split
        )
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        
        train_loss = train(net, trainloader, epochs=1, learning_rate=learning_rate, 
                           max_grad_norm=max_grad_norm, device=device)
        
        test_loss = evaluate(net, testloader, device)
        
        print(f"  ✅ Treinamento e avaliação bem-sucedidos")
        print(f"  📊 Train Loss (1 época): {train_loss:.6f}")
        print(f"  📊 Test Loss: {test_loss:.6f}")
        print(f"  📊 Device: {device}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erro no treinamento ou avaliação: {e}")
        import traceback
        traceback.print_exc()
        return False

# <<< ADICIONADO: Nova função para testar o desempenho ao longo de várias épocas >>>
def test_multi_epoch_performance(config, client_id=1, num_epochs=5):
    """Simula um treinamento por várias épocas para observar a queda da perda."""
    print(f"\n📈 Testando Desempenho Multi-Época (por {num_epochs} épocas)...")

    try:
        # Carrega configurações
        sequence_length = int(config.get("sequence-length", 60))
        prediction_length = int(config.get("prediction-length", 10))
        batch_size = int(config.get("batch-size", 32))
        train_test_split = float(config.get("train-test-split", 0.8))
        learning_rate = float(config.get("learning-rate", 1e-5))
        hidden_size = int(config.get("hidden-size", 50))
        num_layers = int(config.get("num-layers", 1))

        # Carrega dados
        trainloader, testloader, num_features = load_data(
            client_id, sequence_length, prediction_length, batch_size, train_test_split
        )

        # Cria uma nova instância do modelo para este teste
        net = Net(
            input_size=num_features,
            hidden_size=hidden_size,
            output_size=prediction_length,
            num_layers=num_layers
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)

        # Configura otimizador e critério de perda
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        print("  " + "="*50)
        # Loop de treinamento e avaliação por época
        for epoch in range(num_epochs):
            net.train()
            running_loss = 0.0
            for sequences, labels in trainloader:
                sequences, labels = sequences.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = net(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            epoch_train_loss = running_loss / len(trainloader)
            epoch_test_loss = evaluate(net, testloader, device) # Usa nossa função de avaliação

            print(f"  Época {epoch+1:02d}/{num_epochs} | Train Loss: {epoch_train_loss:.6f} | Test Loss: {epoch_test_loss:.6f}")
        
        print("  " + "="*50)
        print("  ✅ Teste de desempenho concluído.")
        return True

    except Exception as e:
        print(f"  ❌ Erro no teste de desempenho: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_config_summary(config):
    """Imprime resumo das configurações"""
    print("\n" + "="*60)
    print("RESUMO DAS CONFIGURAÇÕES")
    print("="*60)
    
    # ... (conteúdo da função inalterado) ...
    print("\n📋 Federação:")
    print(f"  • Estratégia: {config.get('strategy', 'fedavg')}")
    print(f"  • Rodadas: {config.get('rounds', 5)}")
    print(f"  • Nós mínimos: {config.get('min-nodes', 3)}")
    
    print("\n🧠 Modelo LSTM:")
    print(f"  • Input size: {config.get('input-size', 6)}")
    print(f"  • Hidden size: {config.get('hidden-size', 50)}")
    print(f"  • Número de camadas: {config.get('num-layers', 1)}")
    
    print("\n📊 Séries Temporais:")
    print(f"  • Sequence length: {config.get('sequence-length', 60)}")
    print(f"  • Prediction length: {config.get('prediction-length', 10)}")
    
    print("\n🎓 Treinamento:")
    print(f"  • Batch size: {config.get('batch-size', 32)}")
    print(f"  • Learning rate: {config.get('learning-rate', 1e-5)}")
    print(f"  • Épocas locais: {config.get('local-epochs', 1)}")
    print(f"  • Max grad norm: {config.get('max-grad-norm', 1.0)}")
    
    print("\n💾 Dados:")
    print(f"  • Train/Test split: {config.get('train-test-split', 0.8)}")
    print(f"  • Checkpoint a cada: {config.get('save-checkpoint-every', 5)} rodadas")


def main():
    """Função principal de teste"""
    print("="*60)
    print("TESTE DE CONFIGURAÇÃO - FEDERATED LEARNING LSTM")
    print("="*60)
    
    config = load_config()
    print_config_summary(config)
    
    tests_passed = []
    
    tests_passed.append(("Disponibilidade de dados", test_data_availability()))
    
    success, num_features = test_data_loading(config)
    tests_passed.append(("Carregamento de dados", success))
    if not success: sys.exit(1)
    
    success, net = test_model_creation(config, num_features)
    tests_passed.append(("Criação do modelo", success))
    if not success: sys.exit(1)
    
    success = test_forward_pass(net, config)
    tests_passed.append(("Forward pass", success))
    if not success: sys.exit(1)
    
    success = test_training_and_evaluation_step(net, config)
    tests_passed.append(("Passo de Treino e Avaliação", success))
    if not success: sys.exit(1)

    # <<< ADICIONADO: Chamada para o novo teste de desempenho >>>
    success = test_multi_epoch_performance(config)
    tests_passed.append(("Desempenho Multi-Época", success))
    
    print("\n" + "="*60)
    print("RESUMO DOS TESTES")
    print("="*60)
    
    all_passed = True
    for test_name, passed in tests_passed:
        status = "✅ PASSOU" if passed else "❌ FALHOU"
        print(f"{status} - {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("Você pode executar o treinamento federado com:")
        print("  flwr run .")
        print("="*60)
        sys.exit(0)
    else:
        print("❌ ALGUNS TESTES FALHARAM")
        print("Corrija os erros antes de executar o treinamento.")
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    main()