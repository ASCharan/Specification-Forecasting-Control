.PHONY: install test data forecast diagnostic ablation control sensitivity \
        informativeness bootstrap figures all clean

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -q

data:
	python scripts/01_build_dataset.py

# All six city-horizon combinations under ONE lag convention and ONE set of
# scoring rows. Do not vary --lag-offset or --align-test-rows between these.
forecast:
	python scripts/02_run_forecasting.py --city Delhi   --horizon 1  --seeds 0 1 2
	python scripts/02_run_forecasting.py --city Delhi   --horizon 24 --seeds 0 1 2 3 4
	python scripts/02_run_forecasting.py --city Mumbai  --horizon 1  --seeds 0 1 2
	python scripts/02_run_forecasting.py --city Mumbai  --horizon 24 --seeds 0 1 2 3 4
	python scripts/02_run_forecasting.py --city Kolkata --horizon 1  --seeds 0 1 2
	python scripts/02_run_forecasting.py --city Kolkata --horizon 24 --seeds 0 1 2 3 4

# Diagnostic only. Its output belongs in the lag-convention table and nowhere
# else; see CRITICAL_ANALYSIS.md section S1.
diagnostic:
	python scripts/02_run_forecasting.py --city Delhi --horizon 1 --seeds 0 1 2 \
	    --lag-offset 0 --skip-sarima --tag diagnostic

ablation:
	python scripts/06_ablation.py --city Delhi --horizon 1 --seeds 0 1 2

control:
	python scripts/03_train_ppo.py --seeds 0 1 2 --episodes 300

sensitivity:
	python scripts/04_reward_sensitivity.py
	python scripts/05_dispersion_sensitivity.py
	python scripts/09_reward_diagnostics.py

informativeness:
	python scripts/10_traffic_informativeness.py --city Delhi --horizon 1
	python scripts/10_traffic_informativeness.py --city Delhi --horizon 24


bootstrap:
	python scripts/07_paired_bootstrap.py --city Delhi   --horizon 1  --all
	python scripts/07_paired_bootstrap.py --city Delhi   --horizon 24 --all
	python scripts/07_paired_bootstrap.py --city Mumbai  --horizon 24 --all
	python scripts/07_paired_bootstrap.py --city Kolkata --horizon 24 --all

figures:
	python scripts/08_make_figures.py --outdir figures

all: data forecast diagnostic ablation control sensitivity informativeness \
     bootstrap figures

clean:
	rm -rf results/*.json results/predictions checkpoints/*/ figures \
	       __pycache__ .pytest_cache urbanclimate/__pycache__
