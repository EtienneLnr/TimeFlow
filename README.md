# TimeFlow – Continuous Time Series Modeling with Implicit Neural Representations  

**Paper:** “Time Series Continuous Modeling for Imputation and Forecasting with Implicit Neural Representations”  
📄 Published in **TMLR 2024**

<p align="center">
  <img src="INR_explain.png" alt="INR Explanation" width="40%">
</p>

---

## 1. Overview  

**TimeFlow** is a **time-continuous neural model** for **time series imputation and forecasting**.  
It leverages **Implicit Neural Representations (INRs)** and a **meta-learning framework** to model complex temporal dynamics without discretization constraints.

This repository provides:
1. The full implementation of the model described in the TMLR 2024 paper.  
2. Scripts for **imputation**, and **forecasting** experiments.  
3. Datasets example.

---

## 2. Repository Structure  

```bash 
TimeFlow/
├── data/                # Example datasets (Electricity subset for demo)
│   ├── Imputation/      # Data for imputation experiments
│   └── Forecasting/     # Data for forecasting experiments
│
├── experiments/         # Experiment scripts and configurations
│   ├── training/        # Training scripts (.py and .sh)
│   ├── inference/       # Inference scripts (.py and .sh)
│   └── config/          # Experiment configuration (YAML)
│
├── src/                 # Core source code
│   ├── network.py       # INR architecture
│   ├── film_conditionning.py
│   ├── utils.py
│   └── metalearning/    # Meta-learning losses and algorithms
│
├── save_models/         # Saved trained models
├── requirements.txt     # Dependencies
├── INR_explain.png      # Model illustration
└── README.md
```

---

## 3. Running Experiments  

All experiments are run via shell scripts located in the `experiments/` folder.  
**GPU usage is strongly recommended** for faster training.

### Imputation  

**Goal:** Fill missing values in irregularly sampled time series.  

#### Training  

- Without Slurm
```bash 
cd experiments/training
bash inr_imputation.sh
```

- With Slurm
```bash 
cd experiments/training
sbatch inr_imputation.sh
```

#### Adjustable parameters:
- `draw_ratio ∈ {0.05, 0.10, 0.20, 0.30, 0.50}` – percentage of observed 
- `data version ∈ {0, 1}` – dataset version
- *Tip*: Lower draw_ratio allows for higher sample_ratio_batch (faster training).
  
The trained model will be saved in `save_models/`.

#### Inference  

- Without Slurm
```bash 
cd ../inference
bash inference_imputation.sh
```

- With Slurm
```bash 
cd ../inference
sbatch inference_imputation.sh
```

### Forecasting
**Goal:** Predict future values of a time series given its past observations.

#### Training  

- Without Slurm
```bash 
cd experiments/training
bash inr_forecast.sh
```

- With Slurm
```bash 
cd experiments/training
sbatch inr_forecast.sh
```

#### Adjustable parameters:
- `horizon ∈ {96, 192, 336, 720}` – forecasting horizon
- `version ∈ {0, 1}` – dataset version
- *Tip*: For long horizons, you can decrease horizon_ratio to speed up training

#### Inference  

- Without Slurm
```bash 
cd ../inference
bash inference_forecast.sh
```

- With Slurm
```bash 
cd ../inference
sbatch inference_forecast.sh
```

## 4. Data Format

The `data/` folder includes a subset of the **Electricity dataset**, preprocessed for the paper’s experiments.
If you wish to apply *TimeFlow* to your own data, ensure your tensors match the same structure:
- Inputs structured as *(value, gridpoint)* pairs
- Gridpoints scaled to the range [0, 1]

## 5. Dependencies

```bash 
pip install -r requirements.txt
```

## 6. Citation

If you use this work in your research, please cite the paper:


```bibtex 
@article{le2024timeflow,
  title={Time Series Continuous Modeling for Imputation and Forecasting with Implicit Neural Representations},
  author={Le Naour, Etienne and Serrano, Louis and Migus, Léon and Yin, Yuan and Agoua, Ghislain and Baskiotis, Nicolas and Gallinari, Patrick and Guigue, Vincent},
  journal={Transactions on Machine Learning Research (TMLR)},
  year={2024}
}
```

## 7. Contact

📧 Etienne Le Naour — etienne.le-naour@edf.fr
If you find this repository useful, please consider starring ⭐ the repo or citing our work!