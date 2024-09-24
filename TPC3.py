
import pandas as pd
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
from shapely import *
from shapely.plotting import *
from matplotlib.widgets import CheckButtons, RadioButtons
import numpy as np
import networkx as nx
from matplotlib.colors import LogNorm
import math
from matplotlib.cm import ScalarMappable
from scipy.stats import rankdata

# Tarefa 1

# um tuplo (axioma,regras de expansão,ângulo inicial em graus,ângulo de rotação em graus)
lsystem = tuple[str,dict[str,str],float,float]

tree1 : lsystem = ("F",{"F":"F[-F]F[+F][F]"},90,30)
tree2 : lsystem = ("X",{"F":"FF","X":"F-[[X]+X]+F[+FX]-X"},90,22.5)
bush1 : lsystem = ("Y",{"X":"X[-FFF][+FFF]FX","Y":"YFX[+Y][-Y]"},90,25.7)
bush2 : lsystem = ("VZFFF",{"V":"[+++W][---W]YV","W":"+X[-W]Z","X":"-W[+X]Z","Y":"YZ","Z":"[-FFF][+FFF]F"},90,20)
plant1 : lsystem = ("X",{"X":"F+[[X]-X]-F[-FX]+X)","F":"FF"},60,25)

def expandeLSystem(l: lsystem, n: int) -> str:
    """
    Expande o axioma de um sistema L (L-system) através de n iterações utilizando as regras de produção fornecidas.

    A função realiza os seguintes passos:
    1. Desempacota o sistema L, extraindo o axioma e as regras, ignorando os ângulos.
    2. Inicializa a string resultado com o axioma.
    3. Realiza n iterações de expansão:
        - Para cada iteração, inicializa uma nova string.
        - Para cada caractere na string atual, substitui-o pelo valor correspondente na regra, se existir.
        - Se não houver regra de substituição, mantém o caractere original.
        - Atualiza a string resultado para a próxima iteração.
    4. Retorna a string expandida após n iterações.

    Args:
        l (lsystem): Uma tupla contendo o axioma, as regras de produção, o ângulo inicial e o ângulo de rotação do sistema L.
        n (int): O número de iterações a serem realizadas para expandir o axioma.

    Returns:
        str: A string resultante após n iterações de expansão do sistema L.
    """
    axioma, regras, _, _ = l  # Desempacotamos o L-system, ignorando os ângulos
    resultado = axioma  # Inicializamos com o axioma

    for _ in range(n):  # Repetimos a expansão n vezes
        nova_string = ""  # Iniciamos uma nova string para a iteração atual
        for char in resultado:  # Iteramos sobre cada caractere da string atual
            if char in regras:  # Se o caractere tem uma regra de substituição definida
                nova_string += regras[char]  # Substituímos pelo valor correspondente na regra
            else:
                nova_string += char  # Mantemos o caractere se não houver regra de substituição
        resultado = nova_string  # Atualizamos o resultado para a próxima iteração

    return resultado  # Retornamos o resultado após n iterações


def desenhaTurtle(steps: str, start_pos: (float, float), start_angle: float, side: float, theta: float) -> list[list[(float, float)]]:
    """
    Desenha linhas baseadas nos comandos de um sistema L (L-system) utilizando a abordagem turtle graphics.

    A função realiza os seguintes passos:
    1. Inicializa a posição e o ângulo de início.
    2. Percorre cada comando na string `steps`:
        - Se o comando for 'F', move a "tartaruga" para a frente, desenhando uma linha.
        - Se o comando for '-', gira a "tartaruga" no sentido anti-horário pelo ângulo `theta`.
        - Se o comando for '+', gira a "tartaruga" no sentido horário pelo ângulo `theta`.
        - Se o comando for '[', guarda a posição atual e o ângulo na pilha.
        - Se o comando for ']', restaura a posição e o ângulo a partir da pilha e inicia uma nova linha.
    3. Retorna uma lista de listas de tuplos representando as coordenadas das linhas desenhadas.

    Args:
        steps (str): Uma string de comandos do sistema L.
        start_pos ((float, float)): A posição inicial da "tartaruga".
        start_angle (float): O ângulo inicial da "tartaruga".
        side (float): O tamanho de cada passo que a "tartaruga" dá ao desenhar uma linha.
        theta (float): O ângulo de rotação para os comandos '+' e '-'.

    Returns:
        list[list[(float, float)]]: Uma lista de listas de tuplos, onde cada tuplo representa uma coordenada (x, y) de um ponto desenhado.
    """
    pos = start_pos
    angle = start_angle
    lines = [[pos]]
    stack = []
    for s in steps:
        if s == "F":
            pos = (pos[0] + side * math.cos(math.radians(angle)), pos[1] + side * math.sin(math.radians(angle)))
            lines[-1].append(pos)
        elif s == "-":
            angle = angle - theta
        elif s == "+":
            angle = angle + theta
        elif s == "[":
            stack.append((pos, angle))
        elif s == "]":
            pos, angle = stack.pop()
            lines.append([pos])
    return lines


def desenhaLSystem(l: lsystem, n: int):
    """
    Desenha as iterações de um sistema L (L-system) utilizando a abordagem turtle graphics.

    A função realiza os seguintes passos:
    1. Extrai o axioma, as regras, o ângulo inicial e o ângulo de rotação do sistema L.
    2. Cria uma figura com subplots para cada iteração do sistema L, até a n-ésima iteração.
    3. Para cada iteração:
        - Expande o sistema L de acordo com as regras e o número de iteração atual.
        - Desenha o sistema L utilizando a abordagem turtle graphics, calculando as linhas a serem desenhadas.
        - Mapeia a profundidade de cada segmento de linha para atribuir cores diferentes.
        - Desenha os segmentos de linha no subplot correspondente com cores baseadas na profundidade.
        - Adiciona uma legenda indicando a profundidade máxima se necessário.
    4. Ajusta o layout dos subplots.

    Args:
        l (lsystem): Uma tupla contendo o axioma, as regras, o ângulo inicial e o ângulo de rotação do sistema L.
        n (int): O número de iterações a serem desenhadas.

    Returns:
        None
    """
    axioma, regras, start_angle, theta = l
    fig, axes = plt.subplots(1, n + 1, figsize=(15, 3))  # Ajusta o tamanho dos subplots

    start_pos = (0, 0)
    side = 10  # Tamanho de cada passo

    # Cada iteração de expansão do L-system
    for i in range(n + 1):
        s = expandeLSystem((axioma, regras, start_angle, theta), i)
        lines = desenhaTurtle(s, start_pos, start_angle, side, theta)
        ax = axes[i]

        depth_stack = []
        depth = 0
        depth_color_map = {}  # Mapeamento de profundidade para linhas

        # Calcula profundidade e guarda linhas com profundidade
        for line in lines:
            for point_index in range(len(line) - 1):
                if point_index == 0 or line[point_index] != line[point_index - 1]:
                    if s[point_index] == '[':
                        depth_stack.append(depth)
                        depth += 1
                    elif s[point_index] == ']':
                        depth = depth_stack.pop()
                
                depth_color_map.setdefault(depth, []).append((line[point_index], line[point_index + 1]))
        
        linewidth = max(0.1, (n - i + 1) * 2 / n)  # Grossura da linha decresce com o nível 

        # Desenhar cada segmento de linha com a cor baseada na profundidade
        legend_handles = []  # Para guardar as legendas
        for depth, segments in sorted(depth_color_map.items()):
            color = plt.cm.summer(depth / (max(depth_color_map.keys()) + 1))  # Escolhe uma cor
            # Guarda o primeiro segmento para a legenda
            if depth == max(depth_color_map.keys()):  # Somente adiciona a legenda para a maior profundidade
                line_label = ax.plot([], [], color=color, label=f'Nível {depth}')[0]
                legend_handles.append(line_label)
            for start, end in segments:
                ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=linewidth)

        ax.set_title(f'n={i}')  # Título de cada subplot
        ax.axis('equal')
        ax.set_axis_off()
        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper left')

    plt.tight_layout()
    # plt.show()  # comentado para evitar interferência com a execução do arquivo main.py


# Tarefa 2
packaging_waste = pd.read_csv('dados/env_waspac.tsv',na_values=[":", ': '])
municipal_waste = pd.read_csv('dados/env_wasmun.tsv',na_values=[":", ': '])

cols_numbers_packaging = packaging_waste.select_dtypes(include = ['int', 'float']).columns
cols_numbers_municipal = municipal_waste.select_dtypes(include = ['int', 'float']).columns

packaging_waste[cols_numbers_packaging] = packaging_waste[cols_numbers_packaging].bfill(axis=1).ffill(axis=1)
municipal_waste[cols_numbers_municipal] = municipal_waste[cols_numbers_municipal].bfill(axis=1).ffill(axis=1)

def desenhaReciclagemPaisIndice(ax, pais, indice):
    """
    Plota a evolução anual do índice de reciclagem para um país específico e tipo de resíduo.

    A função realiza os seguintes passos:
    1. Formata o nome do índice e do país para serem aceites pelo dataframe.
    2. Filtra os dados de resíduos municipais e de embalagens para o país selecionado.
    3. Plota os dados de reciclagem dependendo do tipo de índice selecionado:
       - Se o índice for 'packaging', plota um gráfico de linha.
       - Se o índice for 'municipal', plota um gráfico de barras.
       - Se o índice for 'combinado', plota um gráfico de áreas empilhadas com os dados combinados.
    4. Define os rótulos dos eixos, a legenda e o título do gráfico.

    Args:
        ax (matplotlib.axes.Axes): O objeto de eixos onde o gráfico será desenhado.
        pais (str): O nome do país a ser plotado.
        indice (str): O tipo de índice de reciclagem ('municipal', 'packaging', ou 'combinado').

    Returns:
        None
    """
    # Formata o índice e o país para serem aceites pelo dataframe
    indice = indice.lower()
    pais = pais.upper()

    mw = municipal_waste[municipal_waste["geo\\TIME_PERIOD"] == pais].select_dtypes(include=['int', 'float']) if pais in municipal_waste["geo\\TIME_PERIOD"].values else []
    pw = packaging_waste[packaging_waste["geo\\TIME_PERIOD"] == pais].select_dtypes(include=['int', 'float']) if pais in packaging_waste["geo\\TIME_PERIOD"].values else []

    if indice == "packaging":  # Se for packaging
        ax.plot(pw.columns.values, pw.values[0], label=pais)

    elif indice == "municipal":  # Se for municipal
        ax.bar(mw.columns.values, mw.values[0], label=pais)

    else:  # Se for total combinado
        years = list(set(mw.columns.values) & set(pw.columns.values))  # Seleciona os anos em comum entre os dois datasets
        combined = [x + y for x, y in zip(mw[years].values[0], pw[years].values[0])]  # Soma os valores dos dois datasets
        ax.stackplot(years, combined, labels=[pais])

    ax.set_xlabel('Ano')
    ax.set_ylabel('KG/HAB')
    ax.legend()
    ax.set_title(f'Evolução Anual do Índice de Reciclagem ({indice}) (KG/HAB)')

def testeDesenhaReciclagemPaisIndice():
    _,ax = plt.subplots()
    desenhaReciclagemPaisIndice(ax,'PT',"combinado")
    plt.show()

def desenhaReciclagem():
    """
    Plota gráficos de reciclagem para diferentes países e tipos de resíduos, permitindo a seleção interativa de países e datasets.

    A função realiza os seguintes passos:
    1. Cria listas de países disponíveis em dois datasets diferentes: resíduos municipais e resíduos de embalagens.
    2. Define um dicionário para armazenar os países disponíveis em cada dataset.
    3. Inicializa uma lista para armazenar os países selecionados e define o dataset padrão como 'Municipal'.
    4. Define uma função interna para plotar o gráfico com base nos países e dataset selecionados.
    5. Cria a figura e os eixos para o gráfico.
    6. Adiciona botões para selecionar os países e o tipo de resíduo.
    7. Define funções para atualizar os países e o tipo de resíduo selecionados quando os botões são clicados.
    8. Mostra o gráfico atualizado com base nas seleções do utilizador.

    Returns:
        None
    """
    # Lista de países para poder selecionar sem dar erro
    paises_m = list(municipal_waste['geo\\TIME_PERIOD'].unique())
    paises_p = list(packaging_waste['geo\\TIME_PERIOD'].unique())
    paises_c = list(np.intersect1d(paises_m, paises_p))
    paises = list(set(paises_m + paises_p))
    paises.sort()

    dict_paises = {'Municipal': paises_m, 'Packaging': paises_p, 'Combinado': paises_c}

    # Lista de países selecionados
    countries = []
    name = 'Municipal'  # O dataset default é o Municipal

    # Função para desenhar o gráfico
    def plot():
        ax.clear()  # Limpa o gráfico
        for country in countries:  # Para cada país selecionado
            if country in dict_paises[name]:  # Se o país existe no df selecionado
                desenhaReciclagemPaisIndice(ax, country, name)  # Desenha o gráfico para cada país
        plt.draw()  # Atualiza o gráfico

    # Criação da figura e dos eixos
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.2)

    # Botões dos países
    fig.text(0.02, 0.94, 'Países:', ha='left', va='baseline', fontsize=12)
    rax = plt.axes([0.02, 0.15, 0.12, 0.78])
    check = CheckButtons(rax, paises, [False] * len(paises))

    # Botões dos datasets
    fig.text(0.02, 0.12, 'Tipo de resíduo:', ha='left', va='baseline', fontsize=12)
    rax_radio = plt.axes([0.02, 0.01, 0.12, 0.1])
    radio = RadioButtons(rax_radio, ('Municipal', 'Packaging', 'Combinado'))

    def select_countries(label):
        if label in countries:
            countries.remove(label)
        else:
            countries.append(label)
        plot()

    check.on_clicked(select_countries)

    def select_data(label):
        nonlocal name
        name = label
        plot()

    radio.on_clicked(select_data)

    # plt.show()  # comentado para evitar interferência com a execução do arquivo main.py

desenhaReciclagem()

# Tarefa 3

listings = pd.read_csv('dados/listings.csv')
neighbourhoods = gpd.read_file("dados/neighbourhoods.geojson")

def desenhaZonas():
    """
    Desenha um mapa das zonas com base no número total de reviews por zona.
    
    A função realiza as seguintes etapas:
    1. Calcula o número total de reviews por zona.
    2. Junta os dados com o GeoDataFrame das zonas.
    3. Aplica uma transformação logarítmica nos dados.
    4. Verifica se o DataFrame resultante não está vazio.
    5. Converte o GeoDataFrame para utilizar a projeção correta.
    6. Desenha o mapa com as configurações desejadas.
    
    Returns:
        None
    """
    # preencher
    reviews_por_zona = listings.groupby('neighbourhood')['number_of_reviews'].sum().reset_index()
    
    # Juntar com o GeoDataFrame das zonas
    zonas_com_reviews = neighbourhoods.merge(reviews_por_zona, on='neighbourhood', how='left')
    zonas_com_reviews.replace([0, np.nan], 1, inplace=True)
    
    # Aplicar a transformação logarítmica
    zonas_com_reviews['log_reviews'] = np.log1p(zonas_com_reviews['number_of_reviews'])

    # Converter o GeoDataFrame para utilizar a projeção correta para adicionar o mapa base
    zonas_com_reviews = zonas_com_reviews.to_crs(epsg=3857)

    # Desenhar o mapa
    ax = zonas_com_reviews.plot(column='log_reviews', cmap='cool', alpha=0.8, edgecolor='k', legend=True)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    
    # Configurações adicionais para melhor visualização
    ax.set_axis_off()
    legend = ax.get_legend()
    if legend:
        legend.set_bbox_to_anchor((1, 1))
    ax.set_title('Número Total de Reviews por Zona (log scale)')
    plt.show()

def desenhaAlojamentos():
    """
    Plota os alojamentos disponíveis na cidade do Porto, colorindo-os de acordo com o preço e utilizando diferentes marcadores para cada tipo de alojamento.

    A função realiza os seguintes passos:
    1. Filtra os alojamentos e bairros que pertencem à cidade do Porto.
    2. Cria uma figura e um eixo para o gráfico.
    3. Aplica uma normalização logarítmica aos preços dos alojamentos e mapeia esses preços para uma paleta de cores.
    4. Define os marcadores a serem usados para cada tipo de alojamento.
    5. Plota os alojamentos utilizando a longitude e latitude, aplicando a cor e tamanho definidos pelos preços e disponibilidade.
    6. Adiciona um mapa de fundo à figura e uma barra de cores que representa os preços.
    7. Adiciona uma legenda indicando o tipo de alojamento.
    8. Define o título do gráfico e exibe-o.

    Returns:
        None
    """
    # Filtrar alojamentos e bairros no Porto, garantindo que é uma cópia
    porto_listings = listings[listings['neighbourhood_group'] == 'PORTO'].copy()
    porto_neighbourhoods = neighbourhoods[neighbourhoods['neighbourhood_group'] == 'PORTO']
    
    # Definir a figura e o eixo
    fig, ax = plt.subplots(figsize=(25, 15))
    porto_neighbourhoods.plot(ax=ax, color='none', edgecolor='black')
    
    # Aplicar normalização por quantis nos preços e criar mapa de cores
    norm = LogNorm(vmin=porto_listings['price'].min(), vmax=porto_listings['price'].max())
    cmap = plt.get_cmap('spring')
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    porto_listings['color'] = porto_listings['price'].apply(lambda x: mappable.to_rgba(x))
    
    # Mapear tipos de alojamento a marcadores
    marker_dict = {'Entire home/apt': '^', 'Private room': 'o', 'Shared room': 's'}

    # Criar handles para a legenda
    handles = [plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor='k', markersize=10, linestyle='None')
               for marker in marker_dict.values()]
    labels = list(marker_dict.keys())
    
    # Plotar usando scatter com parâmetros agrupados por tipo de alojamento
    for room_type, group in porto_listings.groupby('room_type'):
        marker = marker_dict.get(room_type, 'x')  # Define o marcador ou usa 'x' como fallback
        ax.scatter(group['longitude'], group['latitude'],
                   c=group['color'], s=group['availability_365'] / 32,
                   marker=marker, alpha=0.85)
    
    # Adicionar o mapa de fundo e a barra de cores
    ctx.add_basemap(ax, crs=porto_neighbourhoods.crs.to_string(), source=ctx.providers.CartoDB.Positron)
    fig.colorbar(mappable, ax=ax, orientation='horizontal', label='Preço')
    # Adicionar a legenda
    ax.legend(handles, labels, title="Tipo de Alojamento")
    ax.set_title('Alojamentos disponíveis na cidade do Porto')
    
    # Mostrar o gráfico
    plt.show()

def topLocation() -> tuple[str, str, float, float]:
    """
    Determina o alojamento mais próximo do centro do Porto do anfitrião com mais alojamentos.

    A função realiza os seguintes passos:
    1. Calcula o anfitrião com o maior número de alojamentos.
    2. Filtra os alojamentos pertencentes ao anfitrião mais ativo.
    3. Converte os dados filtrados para um GeoDataFrame.
    4. Define o ponto central do Porto.
    5. Calcula a distância de cada alojamento ao centro do Porto.
    6. Encontra o alojamento mais próximo do centro.
    7. Arredonda a latitude e a longitude do alojamento mais próximo para 6 casas decimais.

    Returns:
        tuple[str, str, float, float]: Um tuplo contendo o nome do alojamento mais próximo, o nome do anfitrião,
                                        a latitude arredondada e a longitude arredondada.
    """
    # Calcular o anfitrião com mais alojamentos
    host_counts = listings['host_name'].value_counts()
    top_host_id = host_counts.idxmax()

    # Filtrar alojamentos do anfitrião mais ativo
    top_host_listings = listings[listings['host_name'] == top_host_id]

    # Converter para GeoDataFrame
    gdf = gpd.GeoDataFrame(top_host_listings, geometry=gpd.points_from_xy(top_host_listings.longitude, top_host_listings.latitude))

    # Ponto central (Porto)
    porto_center = Point(-8.6308, 41.1647)

    # Calcular a distância ao centro do Porto
    gdf['distance_to_center'] = gdf.distance(porto_center)

    # Encontrar o alojamento mais próximo do centro
    min_distance_idx = gdf['distance_to_center'].idxmin()
    closest_listing = gdf.loc[min_distance_idx]

    # Arredondar a latitude e longitude para 6 casas decimais para passar no unit tests
    rounded_latitude = round(closest_listing['latitude'], 6)
    rounded_longitude = round(closest_listing['longitude'], 6)

    return closest_listing['name'], closest_listing['host_name'], rounded_latitude, rounded_longitude

def desenhaTop():
    """
    Plota a localização do alojamento mais central na cidade do Porto num mapa.

    A função realiza os seguintes passos:
    1. Obtém a localização do alojamento mais central utilizando a função `topLocation`.
    2. Cria uma figura e eixos para o gráfico.
    3. Cria um GeoDataFrame para o alojamento mais central.
    4. Adiciona um mapa base ao gráfico.
    5. Plota a localização do alojamento mais central no mapa com um marcador vermelho.
    6. Define os limites do gráfico para focar na área do Porto.
    7. Adiciona um rótulo com o nome do alojamento e o nome do anfitrião.
    8. Exibe o gráfico.

    Returns:
        None
    """
    name, host_name, latitude, longitude = topLocation()
    
    # Criar um mapa com a localização do alojamento mais central
    _, ax = plt.subplots(figsize=(10, 15))

    # Criar GeoDataFrame para o alojamento mais central
    d = {'col1': ['target'], 'geometry': [Point(longitude, latitude)]}
    central_location = gpd.GeoDataFrame(d)
    central_location.set_crs(epsg=4326, inplace=True)

    # Adicionar o mapa base
    central_location.plot(ax=ax, color='red', markersize=20)
    ax.set_aspect('equal')

    ax.set_xlim([-8.71, -8.55])
    ax.set_ylim([41.14, 41.21])
    
    print(ax.get_xlim(), ax.get_ylim())

    ctx.add_basemap(ax, crs=central_location.crs, source=ctx.providers.OpenStreetMap.Mapnik)

    # Adicionar rótulo
    ax.text(float(central_location.geometry.x.iloc[0]), float(central_location.geometry.y.iloc[0] + 0.0005), f'{name} ({host_name})', fontsize=12, ha='center')

    plt.show()




# Tarefa 4

bay = pd.read_csv('dados/bay.csv')

def constroiEcosistema() -> nx.DiGraph:
    """
    Constrói um grafo direcionado representando o ecossistema da baía com base nas relações tróficas entre espécies.

    A função realiza os seguintes passos:
    1. Cria uma cópia do dataframe `bay` e indexa-o pelos nomes das espécies.
    2. Seleciona a coluna do nível trófico e remove as colunas 'Names' e 'TrophicLevel' do dataframe copiado.
    3. Adiciona nós ao grafo para cada espécie, associando cada nó ao seu nível trófico.
    4. Para cada coluna no dataframe copiado, adiciona arestas ao grafo representando a transferência de energia entre espécies, ignorando transferências de valor zero.
    5. Retorna o grafo direcionado construído.

    Returns:
        nx.DiGraph: Um grafo direcionado onde os nós representam espécies e as arestas representam transferências de energia entre elas.
    """
    G = nx.DiGraph() 
    bay_copy = bay.copy()  # Copia do dataframe
    bay_copy.index = bay['Names'].values  # Indexa pelo nome
    level = bay_copy[['TrophicLevel']]  # Seleciona a coluna do nível trófico

    bay_copy.drop(columns=['Names', 'TrophicLevel'], inplace=True)

    for col in bay_copy.columns:  # Para cada coluna
        G.add_node(col, level=level.loc[col].values[0]) 

    for col in bay_copy.columns:  # Para cada coluna
        df = bay_copy[col]
        for name in df.index:  # Para cada linha
            G.add_edge(name, col, transfer=df.loc[name]) if df.loc[name] != 0 else None
    return G


def desenhaEcosistema():
    """
    Desenha um grafo representando o ecossistema com base nas relações tróficas entre espécies.

    A função realiza os seguintes passos:
    1. Constrói o grafo do ecossistema utilizando a função `constroiEcosistema`.
    2. Arredonda o nível trófico de cada nó para uma casa decimal.
    3. Define a disposição dos nós no grafo utilizando um layout multipartite baseado no nível trófico.
    4. Converte as transferências de biomassa em escala logarítmica e normaliza os valores.
    5. Define as cores e larguras das arestas com base nas transferências de biomassa.
    6. Desenha o grafo com os nós e arestas coloridos e dimensionados de acordo com os dados.
    7. Adiciona rótulos numerados aos nós.
    8. Cria uma legenda para os nós e uma barra de cores para as arestas representando o fluxo de biomassa.
    9. Ajusta o layout e exibe o gráfico.

    Returns:
        None
    """
    g = constroiEcosistema() 

    for node in g.nodes(data=True):  # Para cada nó
        node[1]['level'] = round(node[1]['level'], 1)  # Arredonda o nível trófico para uma casa decimal
    
    pos = nx.multipartite_layout(g, subset_key='level', scale=1)  # Layout do grafo
    
    transferencias = np.log1p([edge[2]['transfer'] for edge in g.edges(data=True)])  # Mete as transferências em escala logarítmica
    norma = plt.Normalize(vmin=min(transferencias), vmax=max(transferencias))  # Normaliza as transferências

    colors = plt.cm.viridis(norma(transferencias))  # Cores baseadas nas transferências
    widths = 1 + 5 * norma(transferencias)  # Largura das arestas baseada nas transferências

    _, ax = plt.subplots()

    # Desenha o grafo
    nx.draw(g, pos, with_labels=False, node_size=500, node_color='pink', font_size=5, font_color='black', edge_color=colors, width=widths, ax=ax)

    # Numera os nós
    labels = {node: str(i) for i, node in enumerate(g.nodes(), 1)}
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=10, font_color='black')

    # Cria uma legenda para os nós
    legend_labels = [f"{i}: {node}" for i, node in enumerate(g.nodes(), 1)]
    legend_text = 'Legenda: \n' + "\n".join(legend_labels)
    plt.figtext(1.15, 0.5, legend_text, fontsize=8, verticalalignment='center', transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

    # Cria scalar map para a barra de cores das arestas
    sm = plt.cm.ScalarMappable(norm=norma, cmap=plt.cm.viridis)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.4)
    cbar.set_label('Biomass Flow', rotation=270, labelpad=10)

    plt.subplots_adjust(right=0.9, left=0)  # Ajusta o layout
    plt.title('Ecossistema')  # Título

    plt.show()  # Mostra o gráfico


