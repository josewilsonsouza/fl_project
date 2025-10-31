"""
Aplicação Streamlit para visualização interativa de trajetos eVED.

Execute com: streamlit run app_streamlit.py
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import os

# Configuração da página
st.set_page_config(
    page_title="eVED Trajectory Viewer",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importar visualizador
from eved_vizu import EVEDVisualizer

# Funções auxiliares
def get_energy_column(df, veh_type):
    """
    Determina a coluna de energia/consumo baseada no tipo de veículo.
    EVs e PHEVs usam Energy_Consumption, ICE e HEV usam Fuel Rate.
    """
    if veh_type in ['EV', 'PHEV']:
        candidates = ['Energy_Consumption', 'Energy Consumption', 'energy_consumption']
    else:  # ICE, HEV
        candidates = ['Fuel Rate[g/s]', 'Fuel Rate', 'fuel_rate', 'Energy_Consumption']

    for col in candidates:
        if col in df.columns:
            return col
    return None

def get_numeric_columns(df):
    """Retorna todas as colunas numéricas do DataFrame, exceto GPS e timestamp."""
    exclude = ['Latitude[deg]', 'Longitude[deg]', 'Timestamp(ms)', 'Unnamed: 0']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in exclude]

def get_column_display_name(col):
    """Retorna um nome mais amigável para a coluna."""
    friendly_names = {
        'Vehicle Speed[km/h]': 'Velocidade',
        'Energy_Consumption': 'Consumo de Energia',
        'Fuel Rate[g/s]': 'Taxa de Combustível',
        'Gradient': 'Inclinação',
        'OAT[DegC]': 'Temperatura Externa',
        'Air Conditioning Power[Watts]': 'Potência do Ar Condicionado',
        'Heater Power[Watts]': 'Potência do Aquecedor',
        'Elevation Smoothed[m]': 'Elevação',
        'Accel Pedal Rate[%/s]': 'Taxa do Acelerador',
        'Brake Pedal Status': 'Status do Freio',
    }
    return friendly_names.get(col, col)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .vehicle-tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .ev { background-color: #00ff00; color: black; }
    .ice { background-color: #ff0000; color: white; }
    .hev { background-color: #ffaa00; color: black; }
    .phev { background-color: #0088ff; color: white; }
</style>
""", unsafe_allow_html=True)

# Inicializar session state
if 'viz' not in st.session_state:
    st.session_state.viz = EVEDVisualizer()
    st.session_state.current_map = None
    st.session_state.current_data = None

viz = st.session_state.viz

# Header
st.markdown('<div class="main-header">🚗 eVED Trajectory Viewer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Sistema Interativo de Visualização de Trajetos Veiculares</div>', unsafe_allow_html=True)

# Sidebar - Seleção de dados
st.sidebar.header("⚙️ Configurações")

# Escolher split (train/test)
split = st.sidebar.selectbox(
    "📊 Conjunto de Dados",
    options=['train', 'test'],
    index=0,
    help="Selecione o conjunto de dados para explorar"
)

# Filtro por tipo de veículo
vehicle_filter = st.sidebar.selectbox(
    "🚙 Filtrar por Tipo de Veículo",
    options=['Todos', 'EV', 'ICE', 'HEV', 'PHEV'],
    index=0
)

vehicle_type = None if vehicle_filter == 'Todos' else vehicle_filter

# Obter clientes disponíveis
try:
    available_clients = viz.get_available_clients(split, vehicle_type)

    if not available_clients:
        st.error(f"❌ Nenhum cliente encontrado para o filtro selecionado!")
        st.stop()

    st.sidebar.success(f"✅ {len(available_clients)} clientes disponíveis")

    # Selecionar cliente
    client_id = st.sidebar.selectbox(
        "🔢 Selecionar Cliente",
        options=available_clients,
        index=0,
        help="Escolha um cliente para visualizar"
    )

    # Obter trips do cliente selecionado
    available_trips = viz.get_available_trips(client_id, split)

    if not available_trips:
        st.error(f"❌ Cliente {client_id} não possui trips disponíveis!")
        st.stop()

    st.sidebar.info(f"📍 {len(available_trips)} trips disponíveis")

    # Opção para visualizar todas as trips
    view_mode = st.sidebar.radio(
        "🎯 Modo de Visualização",
        options=["Trip Individual", "Todas as Trips do Cliente"],
        index=0,
        help="Escolha entre visualizar uma trip ou todas as trips do cliente"
    )

    # Selecionar trip apenas se modo individual
    if view_mode == "Trip Individual":
        trip_id = st.sidebar.selectbox(
            "🛣️ Selecionar Trip",
            options=available_trips,
            index=0,
            help="Escolha uma viagem específica"
        )
    else:
        trip_id = None  # Todas as trips
        st.sidebar.info(f"🗺️ Visualizando todas as {len(available_trips)} trips")

except Exception as e:
    st.error(f"❌ Erro ao carregar dados: {e}")
    st.stop()

# Geração automática da animação (sem botão)
st.sidebar.markdown("---")
st.sidebar.success("✅ Animação será gerada automaticamente")
generate_map = True  # Sempre gerar automaticamente

# Opções avançadas
with st.sidebar.expander("🔧 Opções Avançadas"):
    show_statistics = st.checkbox("Mostrar Estatísticas Detalhadas", value=True)
    show_plots = st.checkbox("Mostrar Gráficos Analíticos", value=True)
    show_data_table = st.checkbox("Mostrar Tabela de Dados", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Sobre")
st.sidebar.info("""
**eVED Trajectory Viewer** permite explorar e visualizar trajetos de veículos do dataset eVED de forma interativa.

**Tipos de Veículos:**
- 🔋 **EV**: Elétrico
- ⛽ **ICE**: Combustão
- 🔋⛽ **HEV**: Híbrido
- 🔌 **PHEV**: Plug-in Híbrido
""")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Visualização", "📊 Análise de Dados", "📈 Estatísticas", "ℹ️ Informações"])

with tab1:
    # Título dinâmico baseado no modo
    if trip_id is not None:
        st.header(f"🗺️ Visualização do Trajeto - Cliente {client_id}, Trip {trip_id}")
    else:
        st.header(f"🗺️ Visualização de Todas as Trips - Cliente {client_id}")

    # Carregar dados
    try:
        with st.spinner("Carregando dados do trajeto..."):
            # Carregar uma trip específica ou todas
            if trip_id is not None:
                df = viz.load_client_data(client_id, split, trips=[trip_id])
            else:
                df = viz.load_client_data(client_id, split, trips=None)

            df_clean = df.dropna(subset=['Latitude[deg]', 'Longitude[deg]']).copy()
            df_clean = df_clean.sort_values('Timestamp(ms)').reset_index(drop=True)

            st.session_state.current_data = df_clean

            # Informações básicas
            veh_type = viz.get_vehicle_type(df)
            veh_name = viz.vehicle_types.get(veh_type, 'Desconhecido')

            # Detectar coluna de energia/combustível
            energy_col = get_energy_column(df, veh_type)

            # Guardar no session state para uso em outras tabs
            st.session_state.veh_type = veh_type
            st.session_state.veh_name = veh_name
            st.session_state.energy_col = energy_col

            # Badges informativos
            if trip_id is not None:
                col1, col2, col3, col4 = st.columns(4)
            else:
                col1, col2, col3, col4, col5 = st.columns(5)
                with col5:
                    st.metric("🛣️ Trips Carregadas", f"{len(available_trips)}")

            with col1:
                st.metric("🚗 Tipo de Veículo", veh_name)
            with col2:
                st.metric("📍 Pontos GPS", f"{len(df_clean):,}")
            with col3:
                if 'Vehicle Speed[km/h]' in df.columns:
                    avg_speed = df['Vehicle Speed[km/h]'].mean()
                    st.metric("⚡ Velocidade Média", f"{avg_speed:.1f} km/h")
                else:
                    st.metric("⚡ Velocidade Média", "N/A")
            with col4:
                if energy_col and energy_col in df.columns:
                    total_value = df[energy_col].sum()
                    if veh_type in ['ICE', 'HEV']:
                        st.metric("⛽ Combustível Total", f"{total_value:.2f} g")
                    else:
                        st.metric("🔋 Energia Total", f"{total_value:.2f}")
                else:
                    st.metric("🔋/⛽ Energia/Combust.", "N/A")

            st.markdown("---")

            # Mapa animado interativo
            st.subheader("🗺️ Mapa Interativo do Trajeto")

            if generate_map:
                with st.spinner("Gerando mapa animado com Folium..."):
                    # Nome do arquivo baseado no modo
                    if trip_id is not None:
                        output_file = f'trip_cliente_{client_id}_trip_{int(trip_id)}.html'
                    else:
                        output_file = f'trip_cliente_{client_id}_all_trips.html'

                    viz.create_animated_map(
                        client_id=client_id,
                        split=split,
                        trip_id=trip_id,
                        output_file=output_file
                    )
                    st.session_state.current_map = output_file
                    st.success(f"✅ Mapa animado gerado com sucesso!")

            # Mostrar a animação se já foi gerada
            if 'current_map' in st.session_state and st.session_state.current_map is not None:
                output_file = st.session_state.current_map
                if os.path.exists(output_file):
                    # Ler o HTML gerado
                    with open(output_file, 'r', encoding='utf-8') as f:
                        html_content = f.read()

                    # Exibir a animação diretamente no Streamlit
                    st.info("🎮 Use os controles para animação. 📊 Gráficos sincronizados aparecem no lado direito.")
                    components.html(html_content, height=800, scrolling=True)

                    # Botão para download
                    st.download_button(
                        label="📥 Download Mapa HTML",
                        data=html_content,
                        file_name=output_file,
                        mime='text/html',
                        width="stretch"
                    )
                else:
                    st.warning(f"⚠️ Arquivo {output_file} não encontrado. Gere novamente a visualização.")
            else:
                st.info("👆 Clique em '🎬 Gerar Visualização Interativa' na sidebar para criar a animação")

    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {e}")

with tab2:
    st.header("📊 Análise de Dados do Trajeto")

    if st.session_state.current_data is not None:
        df_clean = st.session_state.current_data

        # Obter todas as colunas numéricas disponíveis
        numeric_cols = get_numeric_columns(df_clean)

        if not numeric_cols:
            st.warning("Nenhuma variável numérica encontrada nos dados.")
        else:
            # Sidebar para seleção de variáveis
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 📊 Seleção de Variáveis")

            # Variáveis padrão
            default_var1 = 'Vehicle Speed[km/h]' if 'Vehicle Speed[km/h]' in numeric_cols else numeric_cols[0]
            energy_col = st.session_state.get('energy_col')
            default_var2 = energy_col if energy_col and energy_col in numeric_cols else (numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0])

            var1 = st.sidebar.selectbox(
                "Variável 1 (Principal)",
                options=numeric_cols,
                index=numeric_cols.index(default_var1) if default_var1 in numeric_cols else 0,
                help="Variável para gráficos de linha, histograma e box plot"
            )

            var2 = st.sidebar.selectbox(
                "Variável 2 (Secundária)",
                options=numeric_cols,
                index=numeric_cols.index(default_var2) if default_var2 in numeric_cols else (1 if len(numeric_cols) > 1 else 0),
                help="Variável para gráfico de correlação com Variável 1"
            )

            # Mostrar todas as variáveis disponíveis
            with st.sidebar.expander("📋 Todas as Variáveis"):
                for col in numeric_cols:
                    display_name = get_column_display_name(col)
                    st.text(f"• {display_name}")
                    st.caption(f"  {col}")

            if show_plots:
                # Gráfico de linha - Variável 1
                st.subheader(f"📈 {get_column_display_name(var1)} ao Longo do Trajeto")

                fig_line = px.line(
                    df_clean.reset_index(),
                    x='index',
                    y=var1,
                    labels={'index': 'Ponto do Trajeto', var1: get_column_display_name(var1)},
                    title=f'Variação de {get_column_display_name(var1)}'
                )
                fig_line.update_traces(line_color='#1f77b4')
                st.plotly_chart(fig_line, width="stretch")

                # Histograma e Box Plot
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader(f"📊 Distribuição de {get_column_display_name(var1)}")
                    fig_hist = px.histogram(
                        df_clean,
                        x=var1,
                        nbins=50,
                        labels={var1: get_column_display_name(var1)},
                        title=f'Histograma de {get_column_display_name(var1)}'
                    )
                    st.plotly_chart(fig_hist, width="stretch")

                with col2:
                    st.subheader(f"📦 Box Plot de {get_column_display_name(var1)}")
                    fig_box = px.box(
                        df_clean,
                        y=var1,
                        labels={var1: get_column_display_name(var1)},
                        title=f'Dispersão de {get_column_display_name(var1)}'
                    )
                    st.plotly_chart(fig_box, width="stretch")

                # Gráfico de linha - Variável 2
                st.subheader(f"📈 {get_column_display_name(var2)} ao Longo do Trajeto")

                fig_line2 = px.line(
                    df_clean.reset_index(),
                    x='index',
                    y=var2,
                    labels={'index': 'Ponto do Trajeto', var2: get_column_display_name(var2)},
                    title=f'Variação de {get_column_display_name(var2)}'
                )
                fig_line2.update_traces(line_color='#ff7f0e')
                st.plotly_chart(fig_line2, width="stretch")

                # Scatter: Correlação entre Var1 e Var2
                st.subheader(f"🔗 Relação {get_column_display_name(var1)} × {get_column_display_name(var2)}")

                try:
                    fig_scatter = px.scatter(
                        df_clean,
                        x=var1,
                        y=var2,
                        labels={
                            var1: get_column_display_name(var1),
                            var2: get_column_display_name(var2)
                        },
                        title=f'{get_column_display_name(var1)} vs {get_column_display_name(var2)}',
                        opacity=0.6,
                        trendline="lowess"
                    )
                    st.plotly_chart(fig_scatter, width="stretch")
                except Exception as e:
                    # Se falhar com trendline, tenta sem
                    fig_scatter = px.scatter(
                        df_clean,
                        x=var1,
                        y=var2,
                        labels={
                            var1: get_column_display_name(var1),
                            var2: get_column_display_name(var2)
                        },
                        title=f'{get_column_display_name(var1)} vs {get_column_display_name(var2)}',
                        opacity=0.6
                    )
                    st.plotly_chart(fig_scatter, width="stretch")
                    st.caption(f"⚠️ Trendline não disponível: {str(e)}")

        # Tabela de dados
        if show_data_table:
            st.subheader("📋 Tabela de Dados Brutos")
            st.dataframe(df_clean, width="stretch", height=400)

            # Download CSV
            csv = df_clean.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f'trip_cliente_{client_id}_trip_{int(trip_id)}.csv',
                mime='text/csv',
                width="stretch"
            )
    else:
        st.info("Selecione um cliente e trip, depois clique em 'Carregar Dados' na tab de Visualização.")

with tab3:
    st.header("📈 Estatísticas Detalhadas")

    if st.session_state.current_data is not None and show_statistics:
        df_clean = st.session_state.current_data

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🌍 Estatísticas de GPS")
            stats_gps = {
                "Total de Pontos": f"{len(df_clean):,}",
                "Latitude Mínima": f"{df_clean['Latitude[deg]'].min():.6f}°",
                "Latitude Máxima": f"{df_clean['Latitude[deg]'].max():.6f}°",
                "Longitude Mínima": f"{df_clean['Longitude[deg]'].min():.6f}°",
                "Longitude Máxima": f"{df_clean['Longitude[deg]'].max():.6f}°",
            }
            st.json(stats_gps)

        with col2:
            if 'Vehicle Speed[km/h]' in df_clean.columns:
                st.subheader("⚡ Estatísticas de Velocidade")
                stats_speed = {
                    "Média": f"{df_clean['Vehicle Speed[km/h]'].mean():.2f} km/h",
                    "Mediana": f"{df_clean['Vehicle Speed[km/h]'].median():.2f} km/h",
                    "Desvio Padrão": f"{df_clean['Vehicle Speed[km/h]'].std():.2f} km/h",
                    "Mínima": f"{df_clean['Vehicle Speed[km/h]'].min():.2f} km/h",
                    "Máxima": f"{df_clean['Vehicle Speed[km/h]'].max():.2f} km/h",
                    "Velocidade > 80 km/h": f"{(df_clean['Vehicle Speed[km/h]'] > 80).sum()} pontos"
                }
                st.json(stats_speed)

        # Estatísticas de energia
        if 'Energy_Consumption' in df_clean.columns:
            st.subheader("🔋 Estatísticas de Energia")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Consumo Total", f"{df_clean['Energy_Consumption'].sum():.4f}")
            with col2:
                st.metric("Consumo Médio", f"{df_clean['Energy_Consumption'].mean():.6f}")
            with col3:
                st.metric("Consumo Máximo", f"{df_clean['Energy_Consumption'].max():.6f}")
            with col4:
                st.metric("Consumo Mínimo", f"{df_clean['Energy_Consumption'].min():.6f}")

        # Informações da trip
        st.subheader("ℹ️ Informações da Trip")
        info = {
            "Cliente ID": client_id,
            "Trip ID": trip_id,
            "Split": split,
            "Tipo de Veículo": f"{veh_name} ({veh_type})",
            "Total de Colunas": df_clean.shape[1],
            "Memória Utilizada": f"{df_clean.memory_usage(deep=True).sum() / 1024:.2f} KB"
        }
        st.json(info)

    else:
        st.info("Ative 'Mostrar Estatísticas Detalhadas' na sidebar e carregue os dados.")

with tab4:
    st.header("ℹ️ Informações do Sistema")

    st.markdown("""
    ### 🚗 eVED Trajectory Viewer

    Sistema interativo de visualização de trajetos veiculares do dataset eVED.

    #### 📚 Funcionalidades:

    - **Visualização Interativa**: Mapas interativos com Plotly
    - **Animação Folium**: Gere mapas HTML com animação JavaScript
    - **Análise de Dados**: Gráficos e estatísticas detalhadas
    - **Filtros Avançados**: Filtre por tipo de veículo e conjunto de dados
    - **Export de Dados**: Download de mapas HTML e dados em CSV

    #### 🚙 Tipos de Veículos:

    - 🔋 **EV** (Electric Vehicle): Veículos totalmente elétricos
    - ⛽ **ICE** (Internal Combustion Engine): Veículos a combustão
    - 🔋⛽ **HEV** (Hybrid Electric Vehicle): Veículos híbridos
    - 🔌 **PHEV** (Plug-in Hybrid Electric Vehicle): Híbridos plug-in

    #### 📊 Datasets:

    - **Train**: Conjunto de treinamento para modelos FL
    - **Test**: Conjunto de teste para validação

    #### 🛠️ Tecnologias:

    - Streamlit
    - Plotly
    - Folium
    - Pandas
    - Python 3.11+
    """)

    # Estatísticas gerais
    st.subheader("📊 Estatísticas Gerais do Dataset")

    try:
        total_train = len(viz.get_available_clients('train'))
        total_test = len(viz.get_available_clients('test'))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Clientes (Train)", total_train)
        with col2:
            st.metric("Clientes (Test)", total_test)
        with col3:
            st.metric("Total de Clientes", total_train + total_test)

        # Por tipo de veículo
        st.subheader("🚙 Distribuição por Tipo de Veículo (Train)")

        vehicle_counts = {}
        for vtype in ['EV', 'ICE', 'HEV', 'PHEV']:
            count = len(viz.get_available_clients('train', vehicle_type=vtype))
            vehicle_counts[vtype] = count

        df_vehicles = pd.DataFrame.from_dict(
            vehicle_counts,
            orient='index',
            columns=['Quantidade']
        ).reset_index()
        df_vehicles.columns = ['Tipo', 'Quantidade']

        fig_pie = px.pie(
            df_vehicles,
            values='Quantidade',
            names='Tipo',
            title='Distribuição de Tipos de Veículos',
            color='Tipo',
            color_discrete_map={
                'EV': '#00ff00',
                'ICE': '#ff0000',
                'HEV': '#ffaa00',
                'PHEV': '#0088ff'
            }
        )
        st.plotly_chart(fig_pie, width="stretch")

    except Exception as e:
        st.error(f"Erro ao carregar estatísticas: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>FLEVEn</p>
</div>
""", unsafe_allow_html=True)
