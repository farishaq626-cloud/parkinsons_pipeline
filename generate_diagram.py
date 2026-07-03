from diagrams import Diagram, Edge
from diagrams.generic.compute import Compute
from diagrams.generic.database import Database
from diagrams.generic.blank import Blank
from diagrams.programming.framework import FastAPI

# Set output format to PDF for journal quality
graph_attr = {
    "fontsize": "20",
    "bgcolor": "white"
}

with Diagram("Parkinson's Pipeline Architecture", show=False, filename="pipeline_architecture", outformat="pdf", graph_attr=graph_attr):
    
    # Define Nodes
    data_sources = Database("PPMI/Kaggle\nData Sources")
    etl = Compute("ETL Pipeline\n(Pandas)")
    model = Compute("Random Forest\nClassifier")
    api = FastAPI("FastAPI\nBackend")
    dashboard = Blank("Streamlit\nDashboard")

    # Define Workflow
    data_sources >> etl >> model
    
    # Model connects to both API and Dashboard
    model >> api
    model >> dashboard
    
    # API connects to dashboard
    api >> Edge(label="fetches\npredictions") >> dashboard