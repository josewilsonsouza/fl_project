import os
import json
import folium
import numpy as np
import pandas as pd
from folium import plugins

class EVEDVisualizer:
    """Sistema de visualização de trajetos do dataset eVED"""
    
    def __init__(self, base_path="EVED_Clients"):
        self.base_path = base_path
        self.train_path = os.path.join(base_path, "train")
        self.test_path = os.path.join(base_path, "test")
        
        # Mapeamento de tipos de veículos
        self.vehicle_types = {
            'EV': 'Elétrico',
            'ICE': 'Combustão',
            'HEV': 'Híbrido',
            'PHEV': 'Plug-in Híbrido'
        }
        
        # Cores por tipo de veículo
        self.colors = {
            'EV': '#00ff00',
            'ICE': '#ff0000',
            'HEV': '#ffaa00',
            'PHEV': '#0088ff'
        }
    
    def load_client_data(self, client_id, split='train', trips=None):
        """Carrega dados de um cliente específico"""
        path = self.train_path if split == 'train' else self.test_path
        client_path = os.path.join(path, f"client_{client_id}")
        
        if not os.path.exists(client_path):
            raise ValueError(f"Cliente {client_id} não encontrado em {split}")
        
        trip_files = [f for f in os.listdir(client_path) 
                      if f.startswith('trip_') and f.endswith('.parquet')]
        
        if trips is not None:
            trip_files = [f for f in trip_files 
                         if any(f == f"trip_{t}.parquet" for t in trips)]
        
        dfs = []
        for trip_file in trip_files:
            try:
                df = pd.read_parquet(os.path.join(client_path, trip_file))
                dfs.append(df)
            except Exception as e:
                print(f"Aviso: Falha ao ler {trip_file}: {e}")
                continue
        
        if not dfs:
            raise ValueError(f"Nenhuma viagem válida encontrada para cliente {client_id}")
        
        return pd.concat(dfs, ignore_index=True)
    
    def get_vehicle_type(self, df):
        """Determina o tipo de veículo baseado nas colunas corretas"""
        if 'EngineType' in df.columns:
            veh_type = df['EngineType'].iloc[0]
            if pd.notna(veh_type) and veh_type in ('EV', 'PHEV'):
                return veh_type
        
        if 'Vehicle Type' in df.columns:
            veh_type = df['Vehicle Type'].iloc[0]
            if pd.notna(veh_type) and veh_type in ('ICE', 'HEV'):
                return veh_type
        
        return 'Unknown'
    
    def get_available_clients(self, split='train', vehicle_type=None):
        """Lista clientes disponíveis, opcionalmente filtrados por tipo"""
        path = self.train_path if split == 'train' else self.test_path
        
        if not os.path.exists(path):
            return []
        
        clients = []
        for client_dir in os.listdir(path):
            if not client_dir.startswith('client_'):
                continue
            
            try:
                client_id = int(client_dir.split('_')[1])
                
                if not vehicle_type:
                    clients.append(client_id)
                    continue
                
                metadata_path = os.path.join(path, client_dir, 'metadata.json')
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    if metadata.get('VehicleType') == vehicle_type:
                        clients.append(client_id)
                else:
                    df = self.load_client_data(client_id, split)
                    if self.get_vehicle_type(df) == vehicle_type:
                        clients.append(client_id)
                    
            except Exception as e:
                continue
        
        return sorted(clients)
    
    def get_available_trips(self, client_id, split='train'):
        """Lista os IDs das viagens disponíveis para um cliente"""
        path = self.train_path if split == 'train' else self.test_path
        client_path = os.path.join(path, f"client_{client_id}")
        
        if not os.path.exists(client_path):
            return []
        
        trips = []
        for f in os.listdir(client_path):
            if f.startswith('trip_') and f.endswith('.parquet'):
                try:
                    trip_str = f.replace('trip_', '').replace('.parquet', '')
                    trip_id = float(trip_str)
                    trips.append(trip_id)
                except:
                    continue
        
        return sorted(trips)
    
    def create_animated_map(self, client_id, split='train', trip_id=None, 
                          output_file='trajectory_animation.html'):
        """
        Cria mapa HTML interativo com animação do trajeto usando JavaScript puro
        """
        
        print(f"Carregando dados do cliente {client_id}...")
        
        trips_to_load = [trip_id] if trip_id is not None else None
        df = self.load_client_data(client_id, split, trips_to_load)
        
        df_clean = df.dropna(subset=['Latitude[deg]', 'Longitude[deg]']).copy()
        
        if df_clean.empty:
            raise ValueError("Nenhum dado válido de GPS encontrado")
        
        df_clean = df_clean.sort_values('Timestamp(ms)').reset_index(drop=True)
        
        veh_type = self.get_vehicle_type(df)
        color = self.colors.get(veh_type, '#888888')
        veh_name = self.vehicle_types.get(veh_type, 'Desconhecido')
        
        print(f"Tipo de veículo: {veh_name}")
        print(f"Total de pontos: {len(df_clean)}")
        
        # Reduzir pontos para performance
        step = max(1, len(df_clean) // 300)
        df_animation = df_clean.iloc[::step].reset_index(drop=True)
        
        center_lat = df_animation['Latitude[deg]'].mean()
        center_lon = df_animation['Longitude[deg]'].mean()
        
        # Criar mapa base
        m = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=13,
            tiles='OpenStreetMap'
        )
        
        # Trajeto completo (cinza claro)
        coordinates = df_animation[['Latitude[deg]', 'Longitude[deg]']].values.tolist()
        
        folium.PolyLine(
            coordinates,
            color='#cccccc',
            weight=5,
            opacity=0.5,
            popup=f"Trajeto completo - {veh_name}"
        ).add_to(m)
        
        # Marcadores de início e fim
        folium.CircleMarker(
            coordinates[0],
            radius=10,
            popup=f"<b>INÍCIO</b><br>{veh_name}",
            color='green',
            fill=True,
            fillColor='green',
            fillOpacity=0.8
        ).add_to(m)
        
        folium.CircleMarker(
            coordinates[-1],
            radius=10,
            popup=f"<b>FIM</b><br>{veh_name}",
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.8
        ).add_to(m)
        
        # Preparar dados para JavaScript
        coords_js = [[lat, lon] for lat, lon in coordinates]
        
        # Adicionar JavaScript customizado para animação
        animation_js = f"""
        <script>
        (function() {{
            // Esperar o mapa carregar completamente
            var checkMap = setInterval(function() {{
                if (typeof {m.get_name()} !== 'undefined') {{
                    clearInterval(checkMap);
                    initAnimation();
                }}
            }}, 100);
            
            function initAnimation() {{
                var coords = {json.dumps(coords_js)};
                var currentIndex = 0;
                var isPlaying = false;
                var animationSpeed = 50;
                var intervalId = null;
                
                var pathColor = '{color}';
                var markerColor = '#ffffff';
                
                // Camada para o trajeto percorrido
                var traveledPath = L.polyline([], {{
                    color: pathColor,
                    weight: 5,
                    opacity: 0.9
                }}).addTo({m.get_name()});
                
                // Marcador do carro
                var carMarker = L.circleMarker(coords[0], {{
                    radius: 8,
                    color: pathColor,
                    weight: 3,
                    fillColor: markerColor,
                    fillOpacity: 1
                }}).addTo({m.get_name()});
                
                window.updateAnimation = function() {{
                    if (currentIndex < coords.length) {{
                        carMarker.setLatLng(coords[currentIndex]);
                        var traveled = coords.slice(0, currentIndex + 1);
                        traveledPath.setLatLngs(traveled);
                        
                        document.getElementById('timeSlider').value = currentIndex;
                        document.getElementById('timeDisplay').textContent = 
                            'Ponto: ' + (currentIndex + 1) + ' / ' + coords.length;
                        
                        currentIndex++;
                    }} else {{
                        currentIndex = 0;
                    }}
                }};
                
                window.playAnimation = function() {{
                    if (!isPlaying) {{
                        isPlaying = true;
                        document.getElementById('playBtn').innerHTML = '⏸ Pausar';
                        intervalId = setInterval(window.updateAnimation, animationSpeed);
                    }}
                }};
                
                window.pauseAnimation = function() {{
                    isPlaying = false;
                    document.getElementById('playBtn').innerHTML = '▶ Play';
                    if (intervalId) {{
                        clearInterval(intervalId);
                        intervalId = null;
                    }}
                }};
                
                window.togglePlay = function() {{
                    if (isPlaying) {{
                        window.pauseAnimation();
                    }} else {{
                        window.playAnimation();
                    }}
                }};
                
                window.resetAnimation = function() {{
                    window.pauseAnimation();
                    currentIndex = 0;
                    carMarker.setLatLng(coords[0]);
                    traveledPath.setLatLngs([coords[0]]);
                    document.getElementById('timeSlider').value = 0;
                    document.getElementById('timeDisplay').textContent = 'Ponto: 1 / ' + coords.length;
                }};
                
                window.changeSpeed = function(value) {{
                    animationSpeed = 200 - value;
                    document.getElementById('speedDisplay').textContent = 'Velocidade: ' + value + '%';
                    
                    if (isPlaying) {{
                        clearInterval(intervalId);
                        intervalId = setInterval(window.updateAnimation, animationSpeed);
                    }}
                }};
                
                window.seekTo = function(index) {{
                    currentIndex = parseInt(index);
                    carMarker.setLatLng(coords[currentIndex]);
                    var traveled = coords.slice(0, currentIndex + 1);
                    traveledPath.setLatLngs(traveled);
                    document.getElementById('timeDisplay').textContent = 
                        'Ponto: ' + (currentIndex + 1) + ' / ' + coords.length;
                }};
                
                // Auto-play
                setTimeout(window.playAnimation, 1000);
            }}
        }})();
        </script>
        
        <div style="position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); 
                    background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.3);
                    z-index: 1000; min-width: 400px;">
            <div style="text-align: center; margin-bottom: 10px;">
                <h4 style="margin: 0 0 10px 0;">🚗 {veh_name} - Cliente {client_id}</h4>
                <div id="timeDisplay" style="font-size: 14px; color: #666;">Ponto: 1 / {len(coords_js)}</div>
            </div>
            
            <div style="display: flex; gap: 10px; margin-bottom: 10px; justify-content: center;">
                <button id="playBtn" onclick="togglePlay()" 
                        style="padding: 10px 20px; font-size: 16px; cursor: pointer; border: none; 
                               background: {color}; color: white; border-radius: 5px;">
                    ▶ Play
                </button>
                <button onclick="resetAnimation()" 
                        style="padding: 10px 20px; font-size: 16px; cursor: pointer; border: none;
                               background: #666; color: white; border-radius: 5px;">
                    🔄 Reiniciar
                </button>
            </div>
            
            <div style="margin-bottom: 10px;">
                <input type="range" id="timeSlider" min="0" max="{len(coords_js) - 1}" value="0" 
                       oninput="seekTo(this.value)" 
                       style="width: 100%;">
            </div>
            
            <div>
                <div id="speedDisplay" style="font-size: 12px; color: #666; margin-bottom: 5px;">
                    Velocidade: 100%
                </div>
                <input type="range" id="speedSlider" min="10" max="190" value="100" 
                       oninput="changeSpeed(this.value)" 
                       style="width: 100%;">
            </div>
        </div>
        """
        
        # Adicionar o JavaScript ao mapa
        m.get_root().html.add_child(folium.Element(animation_js))
        
        # Salvar mapa
        m.save(output_file)
        print(f"\n✅ Mapa animado salvo em: {output_file}")
        print(f"📍 Trip ID: {trip_id if trip_id else 'todas as trips'}")
        print(f"🎯 Total de pontos: {len(coords_js)}")
        print(f"\n🌐 Abra o arquivo no navegador - A animação inicia automaticamente!")
        
        return m


# Função auxiliar para uso rápido
def visualize_trip(client_id=0, split='train', trip_id=None):
    """Função rápida para visualizar uma trip"""
    viz = EVEDVisualizer()
    
    if trip_id is None:
        available_trips = viz.get_available_trips(client_id, split)
        if available_trips:
            trip_id = available_trips[0]
            print(f"ℹ️  Usando trip {trip_id} (primeira disponível)")
        else:
            raise ValueError(f"Nenhuma trip encontrada para cliente {client_id}")
    
    trip_id = float(trip_id)
    output_name = f'trip_client_{client_id}_trip_{int(trip_id)}.html'
    
    return viz.create_animated_map(
        client_id=client_id,
        split=split,
        trip_id=trip_id,
        output_file=output_name
    )


# Exemplo de uso
if __name__ == "__main__":
    print("=" * 60)
    print("Sistema de Visualização eVED - Animação JavaScript")
    print("=" * 60)
    
    viz = EVEDVisualizer()
    
    all_clients = viz.get_available_clients('train')
    print(f"\n📋 Total de clientes: {len(all_clients)}")
    print(f"Primeiros 5: {all_clients[:5]}")
    
    if all_clients:
        client_to_viz = all_clients[0]
        trips = viz.get_available_trips(client_to_viz, 'train')
        print(f"\n🚗 Cliente {client_to_viz}")
        print(f"Trips: {trips[:10]}")
        
        if trips:
            print(f"\n🎬 Criando animação...")
            visualize_trip(client_id=client_to_viz, trip_id=trips[0])
    
    print("\n" + "=" * 60)
    print("✅ Pronto! Use: visualize_trip(client_id=0, trip_id=706.0)")
    print("=" * 60)