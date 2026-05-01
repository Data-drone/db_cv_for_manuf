# Databricks notebook source
# MAGIC %md
# MAGIC # Create Vector Search Endpoint
# MAGIC
# MAGIC Creates a Databricks Vector Search endpoint for image similarity search.
# MAGIC **Idempotent** — skips creation if the endpoint already exists.

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "brian_gen_ai")
dbutils.widgets.text("schema", "cv_manufacturing")
dbutils.widgets.text("vs_endpoint_name", "cv-vector-search")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VS_ENDPOINT_NAME = dbutils.widgets.get("vs_endpoint_name")

print(f"Catalog:  {CATALOG}")
print(f"Schema:   {SCHEMA}")
print(f"Endpoint: {VS_ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create / verify the Vector Search endpoint

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

existing = [ep["name"] for ep in vsc.list_endpoints().get("endpoints", [])]

if VS_ENDPOINT_NAME in existing:
    ep = vsc.get_endpoint(VS_ENDPOINT_NAME)
    status = ep.get("endpoint_status", {}).get("state", "UNKNOWN")
    print(f"Endpoint '{VS_ENDPOINT_NAME}' already exists (state: {status})")
else:
    print(f"Creating endpoint '{VS_ENDPOINT_NAME}'...")
    vsc.create_endpoint(name=VS_ENDPOINT_NAME, endpoint_type="STANDARD")
    print(f"Endpoint '{VS_ENDPOINT_NAME}' creation initiated.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wait for ONLINE status

# COMMAND ----------

import time

timeout_minutes = 20
poll_interval = 30
deadline = time.time() + timeout_minutes * 60

while time.time() < deadline:
    ep = vsc.get_endpoint(VS_ENDPOINT_NAME)
    state = ep.get("endpoint_status", {}).get("state", "UNKNOWN")
    print(f"  {VS_ENDPOINT_NAME}: {state}")
    if state == "ONLINE":
        break
    time.sleep(poll_interval)
else:
    print(f"Warning: endpoint did not reach ONLINE within {timeout_minutes} minutes. "
          f"It may still be provisioning — check the UI.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

ep = vsc.get_endpoint(VS_ENDPOINT_NAME)
state = ep.get("endpoint_status", {}).get("state", "UNKNOWN")
print(f"Vector Search endpoint: {VS_ENDPOINT_NAME}")
print(f"State: {state}")
print(f"Ready for index creation against {CATALOG}.{SCHEMA}")
