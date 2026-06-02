from graphviz import Digraph

dot = Digraph()

dot.node('A', 'Raw Data (Credit Card Transactions)')
dot.node('B', 'Data Preprocessing')
dot.node('C', 'Feature Engineering')
dot.node('D', 'XGBoost Model')
dot.node('E', 'Fraud Probability (0–1)')
dot.node('F', 'Streamlit Dashboard')

dot.edges(['AB', 'BC', 'CD', 'DE', 'EF'])

dot.render('reports/screenshots/architecture', format='png', cleanup=True)

print("Architecture diagram generated!")
