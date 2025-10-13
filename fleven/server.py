"""ServerApp para aprendizado federado com FLEVEn."""
import torch
from typing import Iterable, Optional
from pathlib import Path

from flwr.app import Context, ArrayRecord, MetricRecord
from flwr.serverapp import ServerApp, Grid
from flwr.serverapp.strategy import FedAvg, FedAdam, FedYogi, FedAdagrad
from flwr.common import Message

from fleven.utils import Net, set_seed
from fleven.utils import get_model
from fleven.collector import MetricsCollector
from fleven.analysis import create_visualizations, save_detailed_metrics, print_final_summary

STRATEGIES = {
    "fedavg": FedAvg,
    "fedadam": FedAdam,
    "fedyogi": FedYogi,
    "fedadagrad": FedAdagrad,
}


def get_custom_strategy_class(base_strategy_class):
    """Cria dinamicamente uma classe CustomStrategy que herda da estratégia base."""
    
    class CustomStrategy(base_strategy_class):
        def __init__(self, collector: MetricsCollector, **kwargs):
            super().__init__(**kwargs)
            self.collector = collector
            strategy_name = self.__class__.__bases__[0].__name__
            print(f"CustomStrategy (coletando métricas para {strategy_name}) inicializada.")

        def aggregate_train(self, server_round: int, replies: Iterable[Message]) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
            aggregated_arrays, aggregated_metrics = super().aggregate_train(server_round, replies)
            
            if aggregated_metrics:
                individual_losses = {}
                for reply in replies:
                    if reply.has_content() and "metrics" in reply.content:
                        metrics = reply.content["metrics"]
                        client_id = int(metrics.get("client_id", 0))
                        train_loss = float(metrics.get("train_loss", 0.0))
                        print(f"    > Detalhe Cliente {client_id}: Perda de Treino = {train_loss:.6f}")
                        individual_losses[f"client_{client_id}_train_loss"] = train_loss
                
                global_loss = aggregated_metrics.get("train_loss")
                metrics_dict = {"global_train_loss": global_loss}
                metrics_dict.update(individual_losses)
                self.collector.add_train_round(server_round, metrics_dict)

            return aggregated_arrays, aggregated_metrics

        def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]) -> Optional[MetricRecord]:
            aggregated_metrics = super().aggregate_evaluate(server_round, replies)
            
            if aggregated_metrics:
                individual_losses = {}
                for reply in replies:
                    if reply.has_content() and "metrics" in reply.content:
                        metrics = reply.content["metrics"]
                        client_id = int(metrics.get("client_id", 0))
                        eval_loss = float(metrics.get("eval_loss", 0.0))
                        print(f"    > Detalhe Cliente {client_id}: Perda de Avaliação = {eval_loss:.6f}")
                        individual_losses[f"client_{client_id}_eval_loss"] = eval_loss
                
                global_loss = aggregated_metrics.get("eval_loss")
                metrics_dict = {"global_eval_loss": global_loss}
                metrics_dict.update(individual_losses)
                self.collector.add_eval_round(server_round, metrics_dict)

            return aggregated_metrics

    return CustomStrategy

# Cria a aplicação servidor
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Função principal do servidor - lê todas as configurações do Context."""

    seed = int(context.run_config.get("seed", 42))
    set_seed(seed)
    
    # 🔧 configs gerais
    strategy_name = context.run_config.get("strategy", "fedavg").lower()
    num_rounds = int(context.run_config.get("rounds", 5))
    min_nodes = int(context.run_config.get("min-nodes", 3))
    
    # 🔧 Configurações do modelo
    model_type = context.run_config.get("model-type", "lstm")
    input_size = int(context.run_config.get("input-size", 6))
    hidden_size = int(context.run_config.get("hidden-size", 50))
    prediction_length = int(context.run_config.get("prediction-length", 10))
    num_layers = int(context.run_config.get("num-layers", 1))
    sequence_length = int(context.run_config.get("sequence-length", 60))
    
    # 🔧 Caminho para salvar resultados
    results_base_path = context.run_config.get("results-base-path", None)
    if results_base_path:
        output_dir = Path(results_base_path)
    else:
        base_dir = Path(__file__).parent.parent
        output_dir = base_dir / "results"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"SERVIDOR DE APRENDIZADO FEDERADO")
    print(f"{'='*60}")
    print(f"Estratégia: {strategy_name.upper()}")
    print(f"Rodadas: {num_rounds}")
    print(f"Nós mínimos: {min_nodes}")
    print(f"Tamanho da Previsão: {prediction_length}")
    print(f"Tamanho Hidden: {hidden_size}")
    print(f"Número de Camadas do Modelo: {num_layers}")
    print(f"Resultados serão salvos em: {output_dir.absolute()}")
    print(f"{'='*60}\n")
    
    # 🔧 Cria coletor de métricas
    collector = MetricsCollector(strategy_name)

        # 🔧 Cria o dicionário de configuração do modelo
    model_config = {
        "name": model_type,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": prediction_length,
        "num_layers": num_layers,
        "sequence_length": sequence_length
    }
    
    # 🔧 Cria modelo inicial
    #net = Net(
    #    input_size=input_size, 
    #    hidden_size=hidden_size, 
    #    output_size=prediction_length,
    #   num_layers=num_layers
    #)
    net = get_model(model_config)
    initial_arrays = ArrayRecord(net.state_dict())
    
    # 🔧 Parâmetros base para a estratégia
    strategy_params = {
        "fraction_train": 1.0,
        "fraction_evaluate": 1.0,
        "min_available_nodes": min_nodes,
        "min_train_nodes": min_nodes,
        "min_evaluate_nodes": min_nodes,
    }

    # 🔧 Carrega parâmetros específicos da estratégia
    strategy_specific_params = context.run_config.get("strategy-params", {})

    if strategy_name == "fedadam":
        strategy_params["eta"] = float(strategy_specific_params.get("eta", 0.01))
        strategy_params["beta_1"] = float(strategy_specific_params.get("beta_1", 0.9))
        strategy_params["beta_2"] = float(strategy_specific_params.get("beta_2", 0.999))
        print(f"Carregando FedAdam com: eta={strategy_params['eta']}, beta_1={strategy_params['beta_1']}, beta_2={strategy_params['beta_2']}")
    
    elif strategy_name == "fedadagrad":
        strategy_params["eta"] = float(strategy_specific_params.get("eta_adagrad", 0.1))
        strategy_params["initial_accumulator_value"] = float(strategy_specific_params.get("initial_accumulator_value", 0.1))
        print(f"Carregando FedAdagrad com: eta={strategy_params['eta']}, initial_accumulator_value={strategy_params['initial_accumulator_value']}")

    elif strategy_name == "fedyogi":
        strategy_params["eta"] = float(strategy_specific_params.get("eta_yogi", 0.01))
        strategy_params["beta_1"] = float(strategy_specific_params.get("beta_1_yogi", 0.9))
        strategy_params["beta_2"] = float(strategy_specific_params.get("beta_2_yogi", 0.999))
        strategy_params["initial_accumulator_value"] = float(strategy_specific_params.get("initial_accumulator_value_yogi", 1e-6))
        print(f"Carregando FedYogi com: eta={strategy_params['eta']}, beta_1={strategy_params['beta_1']}, beta_2={strategy_params['beta_2']}")

    # 🔧 Instancia a estratégia de forma dinâmica
    BaseStrategyClass = STRATEGIES.get(strategy_name, FedAvg)
    CustomStrategyClass = get_custom_strategy_class(BaseStrategyClass)
    strategy = CustomStrategyClass(collector=collector, **strategy_params)
    
    print("Iniciando servidor FL...")
    
    # 🔧 Inicia o treino federado
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
    )
    
    # 🔧 Imprime informações sobre o resultado final
    print("\n" + "="*60)
    if result.arrays:
        print(f"Modelo final obtido com sucesso!")
        total_params = sum(p.numel() for p in result.arrays.to_torch_state_dict().values())
        print(f"Total de parâmetros: {total_params}")
    
    print(f"Resultados salvos em: {output_dir.absolute()}")
    print("="*60)

    # 🔧 Gera análises e visualizações
    print("\nTREINAMENTO CONCLUÍDO - GERANDO ANÁLISES")
    print("="*60)
    
    try:
        create_visualizations(collector, output_dir)
        save_detailed_metrics(collector, output_dir)
        print_final_summary(collector)
    except Exception as e:
        print(f"AVISO: Erro ao gerar análises: {e}")
        print("O treinamento foi concluído com sucesso, mas as visualizações não foram geradas.")
    
    print("\n" + "="*60)
    print("PROCESSAMENTO FINALIZADO")
    print("="*60)


if __name__ == "__main__":
    print("Servidor pronto para ser executado com Flower 1.22.0")
    print("Use: flwr run .")