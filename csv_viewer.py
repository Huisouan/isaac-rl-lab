import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# 读取两个CSV文件
file1 = 'algo_obs1.csv'
file2 = 'recorddata/algo_obs.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# 检查两个DataFrame的列是否相同
if list(df1.columns) != list(df2.columns):
    raise ValueError("两个CSV文件的列不匹配")

# 初始化 Dash 应用
app = Dash(__name__)

# 创建一个字典来存储每个子图的 figure
figures = {}

for i, column in enumerate(df1.columns):
    fig = go.Figure()
    
    trace1 = go.Scatter(x=df1.index, y=df1[column], mode='lines', name=f'{column} - isaac')
    trace2 = go.Scatter(x=df2.index, y=df2[column], mode='lines', name=f'{column} - mujoco')
    
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    
    fig.update_layout(
        title_text=f'Comparison of {column}',
        showlegend=True,
        height=800
    )
    
    figures[i] = fig

# 定义应用的布局
app.layout = html.Div([
    dcc.Dropdown(
        id='dropdown',
        options=[{'label': f'Comparison of {col}', 'value': i} for i, col in enumerate(df1.columns)],
        value=0
    ),
    dcc.Graph(id='graph')
])

# 定义回调函数来更新图形
@app.callback(
    Output('graph', 'figure'),
    [Input('dropdown', 'value')]
)
def update_graph(selected_value):
    return figures[selected_value]

# 运行应用
if __name__ == '__main__':
    app.run_server(debug=True)