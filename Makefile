# cv-manuf-inspect deploy automation
#
# Usage:
#   make validate                                    # syntax check, no side effects
#   make deploy CATALOG=foo                          # deploy to dev
#   make deploy CATALOG=foo TARGET=prod              # deploy to prod
#   make grant-catalog CATALOG=foo                   # grant USE_CATALOG to app SP (run once after first deploy)
#   make post-deploy CATALOG=foo                     # write SP id to variable-overrides.json + grant catalog
#   make sync                                        # quick re-sync app source via direct API (no Terraform)
#   make logs-url                                    # print the live log viewer URL

PROFILE ?= fe-vm-ramcar-motolite
TARGET  ?= dev
APP     ?= cv-manuf-inspect-$(TARGET)
CATALOG ?=
SCHEMA  ?= cv_manufacturing
CATALOG_VAR := $(if $(CATALOG),--var "catalog=$(CATALOG)",)

CLI_MIN_VERSION := 0.280.0

.PHONY: help check-cli check-catalog validate deploy grant-catalog post-deploy sync logs-url

help:
	@grep -E '^[a-zA-Z_-]+:.*?##' Makefile | awk 'BEGIN{FS=":.*?##"}{printf "  %-20s %s\n",$$1,$$2}'

check-cli:
	@v=$$(databricks --version 2>/dev/null | awk '{print $$NF}' | sed 's/^v//'); \
	if [ -z "$$v" ]; then echo "ERROR: databricks CLI not found"; exit 1; fi; \
	echo "databricks CLI v$$v"

check-catalog:
	@if [ -z "$(CATALOG)" ]; then \
	  echo "ERROR: CATALOG is required, e.g. make $(MAKECMDGOALS) CATALOG=ramcar_motolite_catalog"; exit 2; \
	fi

validate: ## Validate the bundle config (no side effects)
	databricks bundle validate --target $(TARGET) $(CATALOG_VAR) --profile $(PROFILE)

deploy: check-cli ## Deploy bundle: schemas, volumes, app, grants. Override default catalog with CATALOG=...
	databricks bundle deploy --target $(TARGET) $(CATALOG_VAR) --profile $(PROFILE)

grant-catalog: check-catalog ## Grant USE_CATALOG to the app SP (one-time)
	@SP=$$(databricks apps get $(APP) --profile $(PROFILE) -o json | python3 -c "import json,sys; print(json.load(sys.stdin).get('service_principal_client_id',''))"); \
	if [ -z "$$SP" ]; then echo "ERROR: could not resolve service_principal_client_id for app $(APP)"; exit 3; fi; \
	echo "Granting USE_CATALOG on $(CATALOG) to $$SP"; \
	databricks grants update CATALOG $(CATALOG) \
	  --json "{\"changes\":[{\"principal\":\"$$SP\",\"add\":[\"USE_CATALOG\"]}]}" \
	  --profile $(PROFILE) | tail -10

post-deploy: check-catalog ## Write SP id into variable-overrides.json + grant USE_CATALOG
	@SP=$$(databricks apps get $(APP) --profile $(PROFILE) -o json | python3 -c "import json,sys; print(json.load(sys.stdin).get('service_principal_client_id',''))"); \
	if [ -z "$$SP" ]; then echo "ERROR: could not resolve service_principal_client_id"; exit 3; fi; \
	mkdir -p .databricks/bundle/$(TARGET); \
	printf '{\n  "app_service_principal_id": "%s",\n  "catalog": "%s",\n  "schema": "%s"\n}\n' "$$SP" "$(CATALOG)" "$(SCHEMA)" \
	  > .databricks/bundle/$(TARGET)/variable-overrides.json; \
	echo "wrote .databricks/bundle/$(TARGET)/variable-overrides.json"; \
	$(MAKE) grant-catalog CATALOG=$(CATALOG) PROFILE=$(PROFILE)

sync: ## Re-sync app source code only (skips Terraform, uses direct Apps API)
	databricks workspace import-dir ./app /Workspace/Users/$$(databricks current-user me --profile $(PROFILE) -o json | python3 -c "import json,sys; print(json.load(sys.stdin)['userName'])")/$(APP) \
	  --overwrite --profile $(PROFILE)
	databricks apps deploy $(APP) \
	  --source-code-path /Workspace/Users/$$(databricks current-user me --profile $(PROFILE) -o json | python3 -c "import json,sys; print(json.load(sys.stdin)['userName'])")/$(APP) \
	  --profile $(PROFILE) -o json | python3 -c "import json,sys; d=json.load(sys.stdin); print('deploy:', d.get('status',{}).get('state'))"

logs-url:
	@echo "https://$$(databricks apps get $(APP) --profile $(PROFILE) -o json | python3 -c \"import json,sys; print(json.load(sys.stdin)['url'].split('//')[1])\")/logz"
