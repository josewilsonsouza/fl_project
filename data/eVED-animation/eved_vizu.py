import os
import json
import folium
import numpy as np
import pandas as pd
from folium import plugins

class EVEDVisualizer:
    """Sistema de visualização de trajetos do dataset eVED no contexto Fleven"""

    def __init__(self, base_path=None):
        # Se base_path não for fornecido, usa caminho relativo da subpasta
        if base_path is None:
            # Estamos em data/eVED-animation, então EVED_Clients está em ../EVED_Clients
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_path = os.path.join(script_dir, "..", "EVED_Clients")

        self.base_path = os.path.abspath(base_path)
        self.train_path = os.path.join(self.base_path, "train")
        self.test_path = os.path.join(self.base_path, "test")

        # Verifica se o caminho base existe
        if not os.path.exists(self.base_path):
            print(f"⚠️  Aviso: Caminho base não encontrado: {self.base_path}")
            print(f"   Certifique-se de que EVED_Clients existe em {os.path.dirname(self.base_path)}")

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
        Cria mapa HTML interativo com animação do trajeto usando JavaScript puro.
        Se trip_id=None, visualiza todas as trips do cliente com cores diferentes.
        """

        print(f"Carregando dados do cliente {client_id}...")

        trips_to_load = [trip_id] if trip_id is not None else None
        df = self.load_client_data(client_id, split, trips_to_load)

        df_clean = df.dropna(subset=['Latitude[deg]', 'Longitude[deg]']).copy()

        if df_clean.empty:
            raise ValueError("Nenhum dado válido de GPS encontrado")

        df_clean = df_clean.sort_values('Timestamp(ms)').reset_index(drop=True)

        # Detectar se temos múltiplas trips (procurar por Trip_ID, TripID ou similar)
        trip_col = None
        for col in ['Trip_ID', 'TripID', 'trip_id', 'Trip ID']:
            if col in df_clean.columns:
                trip_col = col
                break

        is_multi_trip = trip_id is None and trip_col is not None
        if is_multi_trip:
            unique_trips = df_clean[trip_col].nunique()
            print(f"Múltiplas trips detectadas: {unique_trips} trips")
        
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

        # Preparar dados das variáveis para os gráficos
        variables_data = {}

        # Velocidade
        if 'Vehicle Speed[km/h]' in df_animation.columns:
            variables_data['speed'] = df_animation['Vehicle Speed[km/h]'].fillna(0).tolist()

        # Energia/Combustível
        energy_col = None
        if veh_type in ['EV', 'PHEV']:
            for col in ['Energy_Consumption', 'Energy Consumption', 'energy_consumption']:
                if col in df_animation.columns:
                    energy_col = col
                    break
        else:  # ICE, HEV
            for col in ['Fuel Rate[g/s]', 'Fuel Rate', 'fuel_rate']:
                if col in df_animation.columns:
                    energy_col = col
                    break

        if energy_col:
            variables_data['energy'] = df_animation[energy_col].fillna(0).tolist()
            variables_data['energy_label'] = 'Consumo de Energia' if veh_type in ['EV', 'PHEV'] else 'Taxa de Combustível'

        # Elevação
        if 'Elevation Smoothed[m]' in df_animation.columns:
            variables_data['elevation'] = df_animation['Elevation Smoothed[m]'].fillna(0).tolist()

        # Adicionar JavaScript customizado para animação com gráficos
        animation_js = f"""
        <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
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
                var variablesData = {json.dumps(variables_data)};
                var currentIndex = 0;
                var isPlaying = false;
                var animationSpeed = 50;
                var intervalId = null;

                var pathColor = '{color}';
                var markerColor = '#ffffff';

                // Função para inicializar gráficos
                function initCharts(data) {{
                    var chartsObj = {{}};

                    // Gráfico de Velocidade
                    if (data.speed) {{
                        var speedCtx = document.getElementById('speedChart').getContext('2d');
                        chartsObj.speed = new Chart(speedCtx, {{
                            type: 'line',
                            data: {{
                                labels: Array.from({{length: data.speed.length}}, (_, i) => i),
                                datasets: [{{
                                    label: 'Velocidade (km/h)',
                                    data: data.speed,
                                    borderColor: '#3498db',
                                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                                    borderWidth: 2,
                                    tension: 0.4,
                                    pointRadius: 0
                                }}]
                            }},
                            options: {{
                                responsive: false,
                                maintainAspectRatio: false,
                                animation: false,
                                plugins: {{
                                    legend: {{display: false}}
                                }},
                                scales: {{
                                    x: {{display: false}},
                                    y: {{
                                        beginAtZero: true,
                                        ticks: {{color: '#666', font: {{size: 10}}}}
                                    }}
                                }}
                            }}
                        }});
                    }}

                    // Gráfico de Energia/Combustível
                    if (data.energy) {{
                        var energyCtx = document.getElementById('energyChart').getContext('2d');
                        chartsObj.energy = new Chart(energyCtx, {{
                            type: 'line',
                            data: {{
                                labels: Array.from({{length: data.energy.length}}, (_, i) => i),
                                datasets: [{{
                                    label: data.energy_label || 'Energia',
                                    data: data.energy,
                                    borderColor: '#e74c3c',
                                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                                    borderWidth: 2,
                                    tension: 0.4,
                                    pointRadius: 0
                                }}]
                            }},
                            options: {{
                                responsive: false,
                                maintainAspectRatio: false,
                                animation: false,
                                plugins: {{
                                    legend: {{display: false}}
                                }},
                                scales: {{
                                    x: {{display: false}},
                                    y: {{
                                        beginAtZero: true,
                                        ticks: {{color: '#666', font: {{size: 10}}}}
                                    }}
                                }}
                            }}
                        }});
                    }}

                    // Gráfico de Elevação
                    if (data.elevation) {{
                        var elevCtx = document.getElementById('elevationChart').getContext('2d');
                        chartsObj.elevation = new Chart(elevCtx, {{
                            type: 'line',
                            data: {{
                                labels: Array.from({{length: data.elevation.length}}, (_, i) => i),
                                datasets: [{{
                                    label: 'Elevação (m)',
                                    data: data.elevation,
                                    borderColor: '#2ecc71',
                                    backgroundColor: 'rgba(46, 204, 113, 0.1)',
                                    borderWidth: 2,
                                    tension: 0.4,
                                    pointRadius: 0,
                                    fill: true
                                }}]
                            }},
                            options: {{
                                responsive: false,
                                maintainAspectRatio: false,
                                animation: false,
                                plugins: {{
                                    legend: {{display: false}}
                                }},
                                scales: {{
                                    x: {{display: false}},
                                    y: {{
                                        ticks: {{color: '#666', font: {{size: 10}}}}
                                    }}
                                }}
                            }}
                        }});
                    }}

                    return chartsObj;
                }}

                // Inicializar gráficos
                var charts = initCharts(variablesData);

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

                // Atualizar indicador de posição nos gráficos
                function updateChartIndicators(index) {{
                    Object.keys(charts).forEach(function(key) {{
                        var chart = charts[key];
                        // Criar/atualizar linha vertical no gráfico
                        if (chart.options.plugins.annotation) {{
                            chart.options.plugins.annotation.annotations.line1.xMin = index;
                            chart.options.plugins.annotation.annotations.line1.xMax = index;
                        }} else {{
                            // Atualizar viewport do gráfico
                            var maxVisible = 50; // Mostrar últimos 50 pontos
                            var minX = Math.max(0, index - maxVisible);
                            chart.options.scales.x.min = minX;
                            chart.options.scales.x.max = Math.max(maxVisible, index + 10);
                        }}
                        chart.update('none');
                    }});

                    // Atualizar valores numéricos
                    if (variablesData.speed) {{
                        document.getElementById('speedValue').textContent =
                            variablesData.speed[index].toFixed(1) + ' km/h';
                    }}
                    if (variablesData.energy) {{
                        document.getElementById('energyValue').textContent =
                            variablesData.energy[index].toFixed(2);
                    }}
                    if (variablesData.elevation) {{
                        document.getElementById('elevationValue').textContent =
                            variablesData.elevation[index].toFixed(1) + ' m';
                    }}
                }}

                window.updateAnimation = function() {{
                    // Só atualiza se estiver realmente tocando
                    if (!isPlaying) return;

                    if (currentIndex < coords.length) {{
                        carMarker.setLatLng(coords[currentIndex]);
                        var traveled = coords.slice(0, currentIndex + 1);
                        traveledPath.setLatLngs(traveled);

                        document.getElementById('timeSlider').value = currentIndex;
                        document.getElementById('timeDisplay').textContent =
                            'Ponto: ' + (currentIndex + 1) + ' / ' + coords.length;

                        // Atualizar gráficos
                        updateChartIndicators(currentIndex);

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

                    // Atualizar gráficos ao usar o slider
                    updateChartIndicators(currentIndex);
                }};
                
                // Auto-play
                setTimeout(window.playAnimation, 1000);
            }}
        }})();
        </script>

        <!-- Painéis de Gráficos Laterais -->
        <div style="position: fixed; right: 15px; top: 15px; width: 400px; z-index: 1000; max-height: 90vh; overflow-y: auto;">
            {'<div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin-bottom: 15px; border-left: 5px solid #3498db;"><h4 style="margin: 0 0 10px 0; font-size: 16px; color: #3498db; font-weight: bold;">⚡ Velocidade</h4><div id="speedValue" style="font-size: 32px; font-weight: bold; color: #3498db; margin-bottom: 15px; text-align: center;">0 km/h</div><canvas id="speedChart" width="360" height="120"></canvas></div>' if 'speed' in variables_data else ''}
            {'<div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin-bottom: 15px; border-left: 5px solid #e74c3c;"><h4 style="margin: 0 0 10px 0; font-size: 16px; color: #e74c3c; font-weight: bold;">' + ('🔋 ' if veh_type in ['EV', 'PHEV'] else '⛽ ') + variables_data.get('energy_label', 'Energia') + '</h4><div id="energyValue" style="font-size: 32px; font-weight: bold; color: #e74c3c; margin-bottom: 15px; text-align: center;">0</div><canvas id="energyChart" width="360" height="120"></canvas></div>' if 'energy' in variables_data else ''}
            {'<div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); border-left: 5px solid #2ecc71;"><h4 style="margin: 0 0 10px 0; font-size: 16px; color: #2ecc71; font-weight: bold;">⛰️ Elevação</h4><div id="elevationValue" style="font-size: 32px; font-weight: bold; color: #2ecc71; margin-bottom: 15px; text-align: center;">0 m</div><canvas id="elevationChart" width="360" height="120"></canvas></div>' if 'elevation' in variables_data else ''}
        </div>

        <!-- Controles de Animação -->
        <div style="position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
                    background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.3);
                    z-index: 1000; min-width: 400px;">
            <div style="text-align: center; margin-bottom: 10px;">
                <h4 style="margin: 0 0 10px 0;">🚗 {veh_name} - Cliente {client_id}{'<br><small style="color: #888;">Todas as Trips</small>' if trip_id is None else ''}</h4>
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