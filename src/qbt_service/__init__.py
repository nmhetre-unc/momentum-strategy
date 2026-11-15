"""
The backtest service: an HTTP API that queues backtest jobs in
PostgreSQL and workers that run them with qbt.runner. Installed with the
`api` extra (pip install -e ".[api]"); the qbt library, CLI and Streamlit
dashboard never import this package.
"""
