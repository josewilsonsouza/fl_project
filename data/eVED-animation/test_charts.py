"""
Script de teste para verificar a geração de animação com gráficos sincronizados
"""

from eved_vizu import EVEDVisualizer

# Criar visualizador
viz = EVEDVisualizer()

# Tentar gerar uma animação com o primeiro cliente disponível
try:
    print("🔍 Buscando clientes disponíveis...")
    clients = viz.get_available_clients('train')

    if clients:
        client_id = clients[0]
        print(f"✅ Cliente {client_id} encontrado")

        print("🔍 Buscando trips disponíveis...")
        trips = viz.get_available_trips(client_id, 'train')

        if trips:
            trip_id = trips[0]
            print(f"✅ Trip {trip_id} encontrada")

            print("🎬 Gerando animação com gráficos...")
            viz.create_animated_map(
                client_id=client_id,
                split='train',
                trip_id=trip_id,
                output_file='test_animation_with_charts.html'
            )

            print("\n✅ Sucesso! Abra test_animation_with_charts.html no navegador")
            print("📊 Você deverá ver:")
            print("   - Mapa animado no centro")
            print("   - Painéis de gráficos no lado direito")
            print("   - Controles de animação na parte inferior")
            print("   - Gráficos sincronizados com a posição do veículo")
        else:
            print("❌ Nenhuma trip encontrada")
    else:
        print("❌ Nenhum cliente encontrado")

except Exception as e:
    print(f"❌ Erro: {e}")
    import traceback
    traceback.print_exc()
