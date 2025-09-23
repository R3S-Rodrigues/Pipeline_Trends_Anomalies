# Pipeline_Trends_Anomalies
Portfolio project that implements a pipeline for detecting trends and anomalies in oil production data (m³), aiming to improve operational efficiency and predict failures in the E&amp;P industry.

Phase 2 - Time series trend detection 
Ingestion and pre-processing

This initial stage of the pipeline aims to extract, clean, and organise time series data directly from the MySQL Data Warehouse. It represents the basis of the time series modelling architecture and prepares the data for statistical decomposition and trend analysis.
Implemented features:

Secure connection to the database using SQLAlchemy and parameterised authentication.

Execution of SQL queries to retrieve production records (production_date, volume_m3) with temporal filters and by operating unit.

Data pre-processing, including:
Removal of duplicates
Conversion of the date column to datetime format
Chronological sorting of the series

Export of standardised data to a .csv file, ensuring traceability and reuse in the modelling stage.

This phase ensures that the time series is clean, structured and ready for decomposition, maintaining compliance with principles of governance, reproducibility and operational impact
